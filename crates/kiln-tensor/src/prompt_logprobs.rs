//! Compact device-side prompt-logprob selection results.

#[cfg(any(feature = "cuda", feature = "rocm", test))]
use crate::{Error, Result};

/// One selected prompt-logprob candidate, ranked from the original logits.
#[derive(Debug, Clone, PartialEq)]
pub struct DevicePromptLogprobCandidate {
    /// Vocabulary token ID.
    pub token_id: u32,
    /// Original F32-comparison logit.
    pub logit: f32,
    /// Stable F32 log-softmax value.
    pub logprob: f32,
}

/// Compact result for one vocabulary row.
#[derive(Debug, Clone, PartialEq)]
pub struct DevicePromptLogprobRow {
    /// Stable row maximum used by log-softmax.
    pub row_max: f32,
    /// `ln(sum(exp(logit - row_max)))` for the row.
    pub log_sum_exp_shifted: f32,
    /// Original observed-token logit.
    pub observed_logit: f32,
    /// Stable observed-token log-probability.
    pub observed_logprob: f32,
    /// Count of vocabulary logits greater than or equal to the observed logit.
    pub observed_full_rank: usize,
    /// Top-K candidates in descending-logit, ascending-token-ID tie order.
    pub candidates: Vec<DevicePromptLogprobCandidate>,
}

#[allow(clippy::too_many_arguments)]
#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn finish_device_prompt_logprob_rows(
    operation: &str,
    n_rows: usize,
    n_cols: usize,
    top_k: usize,
    row_maxes: Vec<f32>,
    log_sums: Vec<f32>,
    observed_logits: Vec<f32>,
    observed_ranks: Vec<i64>,
    top_logits: Vec<f32>,
    top_indices: Vec<i64>,
    invalid_kinds: Vec<u32>,
    invalid_columns: Vec<i64>,
    invalid_values: Vec<f32>,
) -> Result<Vec<DevicePromptLogprobRow>> {
    let top_count = n_rows
        .checked_mul(top_k)
        .ok_or_else(|| Error::Msg(format!("{operation}: top-k output length overflow")))?;
    let row_lengths = [
        row_maxes.len(),
        log_sums.len(),
        observed_logits.len(),
        observed_ranks.len(),
        invalid_kinds.len(),
        invalid_columns.len(),
        invalid_values.len(),
    ];
    if row_lengths.iter().any(|&len| len != n_rows) {
        return Err(Error::Msg(format!(
            "{operation}: compact row output lengths {row_lengths:?} did not all equal {n_rows}"
        )));
    }
    if top_logits.len() != top_count || top_indices.len() != top_count {
        return Err(Error::Msg(format!(
            "{operation}: compact top-k output lengths {}/{} did not both equal {top_count}",
            top_logits.len(),
            top_indices.len()
        )));
    }

    for row in 0..n_rows {
        match invalid_kinds[row] {
            0 => {}
            kind @ (1 | 2) => {
                let column = invalid_columns[row];
                let value = invalid_values[row];
                let value_kind = if kind == 1 {
                    "logit"
                } else {
                    "log-probability"
                };
                return Err(Error::Msg(format!(
                    "{operation}: row {row} contained non-finite {value_kind} value {value:?} at token id {column}"
                )));
            }
            kind => {
                return Err(Error::Msg(format!(
                    "{operation}: row {row} returned unknown validation kind {kind}"
                )));
            }
        }
    }

    let mut rows = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let row_max = row_maxes[row];
        let log_sum_exp_shifted = log_sums[row];
        let observed_logit = observed_logits[row];
        let observed_logprob = (observed_logit - row_max) - log_sum_exp_shifted;
        if !row_max.is_finite()
            || !log_sum_exp_shifted.is_finite()
            || !observed_logit.is_finite()
            || !observed_logprob.is_finite()
        {
            return Err(Error::Msg(format!(
                "{operation}: row {row} returned non-finite compact normalization output"
            )));
        }

        let observed_full_rank = usize::try_from(observed_ranks[row]).map_err(|_| {
            Error::Msg(format!(
                "{operation}: row {row} returned invalid observed rank {}",
                observed_ranks[row]
            ))
        })?;
        if !(1..=n_cols).contains(&observed_full_rank) {
            return Err(Error::Msg(format!(
                "{operation}: row {row} observed rank {observed_full_rank} was outside 1..={n_cols}"
            )));
        }

        let start = row * top_k;
        let mut candidates: Vec<DevicePromptLogprobCandidate> = Vec::with_capacity(top_k);
        let mut seen = std::collections::HashSet::with_capacity(top_k);
        for rank in 0..top_k {
            let flat = start + rank;
            let token_id = u32::try_from(top_indices[flat]).map_err(|_| {
                Error::Msg(format!(
                    "{operation}: row {row} rank {} returned invalid token id {}",
                    rank + 1,
                    top_indices[flat]
                ))
            })?;
            if token_id as usize >= n_cols {
                return Err(Error::Msg(format!(
                    "{operation}: row {row} rank {} token id {token_id} was outside vocabulary width {n_cols}",
                    rank + 1
                )));
            }
            if !seen.insert(token_id) {
                return Err(Error::Msg(format!(
                    "{operation}: row {row} returned duplicate top-k token id {token_id}"
                )));
            }
            let logit = top_logits[flat];
            let logprob = (logit - row_max) - log_sum_exp_shifted;
            if !logit.is_finite() || !logprob.is_finite() {
                return Err(Error::Msg(format!(
                    "{operation}: row {row} rank {} returned non-finite compact candidate",
                    rank + 1
                )));
            }
            if let Some(previous) = candidates.last() {
                let correctly_ordered = previous.logit > logit
                    || (previous.logit == logit && previous.token_id < token_id);
                if !correctly_ordered {
                    return Err(Error::Msg(format!(
                        "{operation}: row {row} top-k candidates were not in stable rank order at rank {}",
                        rank + 1
                    )));
                }
            }
            candidates.push(DevicePromptLogprobCandidate {
                token_id,
                logit,
                logprob,
            });
        }

        rows.push(DevicePromptLogprobRow {
            row_max,
            log_sum_exp_shifted,
            observed_logit,
            observed_logprob,
            observed_full_rank,
            candidates,
        });
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compact_outputs_reject_unstable_tie_order() {
        let error = finish_device_prompt_logprob_rows(
            "test",
            1,
            4,
            2,
            vec![3.0],
            vec![0.5],
            vec![2.0],
            vec![2],
            vec![3.0, 3.0],
            vec![2, 1],
            vec![0],
            vec![-1],
            vec![0.0],
        )
        .unwrap_err();
        assert!(error.to_string().contains("stable rank order"));
    }

    #[test]
    fn compact_outputs_surface_derived_non_finite_location() {
        let error = finish_device_prompt_logprob_rows(
            "test",
            1,
            4,
            0,
            vec![0.0],
            vec![0.0],
            vec![0.0],
            vec![1],
            vec![],
            vec![],
            vec![2],
            vec![3],
            vec![f32::NEG_INFINITY],
        )
        .unwrap_err();
        assert!(error.to_string().contains("log-probability"));
        assert!(error.to_string().contains("token id 3"));
    }
}
