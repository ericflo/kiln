//! `einsum` — Einstein-summation contraction over named axes.
//!
//! Implements a CPU reference for the subset of einsum that covers
//! the contraction patterns the Qwen3.5-4B forward pass uses:
//!
//! - matmul: `"ij,jk->ik"`
//! - batched matmul: `"bij,bjk->bik"`
//! - QKV attention scores: `"bhid,bhjd->bhij"`
//! - attention weighted sum: `"bhij,bhjd->bhid"`
//! - outer product: `"i,j->ij"`
//! - inner product: `"i,i->"`
//! - trace via duplicate: `"ii->"`
//! - per-row dot: `"bi,bi->b"`
//!
//! Each operand letter must be a single lowercase ASCII character.
//! Repeated letters in one operand are diagonal-take (e.g. `"ii->"`
//! sums the diagonal). Output letters must be a subset of the input
//! letters. Missing output letters are summed over (reduction
//! axes).
//!
//! For pairs whose inputs reduce to a standard `matmul`, we route
//! through that fast path; otherwise the reference uses a per-
//! output-position nested-loop accumulator. The migration plan in
//! Phase 7 swaps this CPU reference for `kiln-blas`-backed
//! tensor-contraction kernels per backend; the public API is
//! preserved.

use std::sync::Arc;

use crate::{bail, CpuStorage, DType, Layout, Result, Storage, Tensor, TensorId};

#[derive(Debug)]
struct ParsedSpec {
    inputs: Vec<Vec<u8>>,
    output: Vec<u8>,
}

fn parse_spec(spec: &str) -> Result<ParsedSpec> {
    let (lhs, rhs) = match spec.split_once("->") {
        Some((l, r)) => (l, r),
        None => bail!("einsum: spec must contain '->', got {spec:?}"),
    };
    let inputs: Vec<Vec<u8>> = lhs
        .split(',')
        .map(|term| term.trim().bytes().collect())
        .collect();
    let output: Vec<u8> = rhs.trim().bytes().collect();
    for term in inputs.iter().chain(std::iter::once(&output)) {
        for &b in term {
            if !(b.is_ascii_lowercase()) {
                bail!("einsum: only lowercase ASCII axis letters allowed (got byte 0x{:02x})", b);
            }
        }
    }
    // Output letters must all appear in some input.
    let mut all_in: std::collections::HashSet<u8> = std::collections::HashSet::new();
    for term in &inputs {
        for &b in term {
            all_in.insert(b);
        }
    }
    for &b in &output {
        if !all_in.contains(&b) {
            bail!(
                "einsum: output letter '{}' not present in any input",
                b as char
            );
        }
    }
    Ok(ParsedSpec { inputs, output })
}

fn read_f32_flat(t: &Tensor) -> Result<Vec<f32>> {
    if !t.is_contiguous() {
        bail!("einsum: input must be contiguous");
    }
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| crate::Error::from_str("einsum: storage must be CpuStorage"))?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let mut out = Vec::with_capacity(n);
    match t.dtype() {
        DType::F32 => {
            for i in 0..n {
                out.push(f32::from_le_bytes(
                    bytes[i * 4..i * 4 + 4].try_into().unwrap(),
                ));
            }
        }
        DType::BF16 => {
            for i in 0..n {
                out.push(
                    half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                );
            }
        }
        DType::F16 => {
            for i in 0..n {
                out.push(
                    half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                );
            }
        }
        other => bail!("einsum: dtype must be F32/BF16/F16, got {other}"),
    }
    Ok(out)
}

fn write_f32_into(dtype: DType, values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * dtype.size_in_bytes());
    for &v in values {
        match dtype {
            DType::F32 => out.extend_from_slice(&v.to_le_bytes()),
            DType::BF16 => out.extend_from_slice(&half::bf16::from_f32(v).to_le_bytes()),
            DType::F16 => out.extend_from_slice(&half::f16::from_f32(v).to_le_bytes()),
            _ => unreachable!(),
        }
    }
    out
}

/// Run einsum on the given operands. Supports 1- or 2-operand specs.
pub fn einsum(spec: &str, operands: &[&Tensor]) -> Result<Tensor> {
    let parsed = parse_spec(spec)?;
    if operands.is_empty() || operands.len() > 2 {
        bail!("einsum: 1 or 2 operands supported, got {}", operands.len());
    }
    if parsed.inputs.len() != operands.len() {
        bail!(
            "einsum: spec has {} input terms but got {} operands",
            parsed.inputs.len(),
            operands.len()
        );
    }
    // Resolve each axis letter to its size by looking at the
    // operands.
    let mut sizes: std::collections::HashMap<u8, usize> =
        std::collections::HashMap::new();
    for (term, op) in parsed.inputs.iter().zip(operands.iter()) {
        if term.len() != op.rank() {
            bail!(
                "einsum: spec term length {} != operand rank {}",
                term.len(),
                op.rank()
            );
        }
        for (i, &b) in term.iter().enumerate() {
            let sz = op.shape()[i];
            match sizes.get(&b) {
                Some(&prev) if prev != sz => bail!(
                    "einsum: axis '{}' bound to two sizes ({} and {})",
                    b as char,
                    prev,
                    sz
                ),
                _ => {
                    sizes.insert(b, sz);
                }
            }
        }
    }
    // Output shape from the output letters.
    let out_shape: Vec<usize> = parsed
        .output
        .iter()
        .map(|b| {
            *sizes
                .get(b)
                .expect("output letter must be in sizes (already validated)")
        })
        .collect();

    // Dtype: use the first operand's dtype (validate they all match).
    let dtype = operands[0].dtype();
    for op in operands {
        if op.dtype() != dtype {
            bail!(
                "einsum: dtype mismatch — first operand is {} but another is {}",
                dtype,
                op.dtype()
            );
        }
    }

    // Read each operand as flat f32.
    let flats: Vec<Vec<f32>> = operands
        .iter()
        .map(|t| read_f32_flat(t))
        .collect::<Result<Vec<_>>>()?;

    // Build operand strides (in elements).
    let mut op_strides: Vec<Vec<usize>> = Vec::with_capacity(operands.len());
    for op in operands {
        let rank = op.rank();
        let mut s = vec![1usize; rank.max(1)];
        for d in (0..rank.saturating_sub(1)).rev() {
            s[d] = s[d + 1] * op.shape()[d + 1];
        }
        op_strides.push(s);
    }

    // Identify which axis letters are reduction (in input but not in
    // output) and which are kept.
    let mut all_letters: std::collections::HashSet<u8> = std::collections::HashSet::new();
    for term in &parsed.inputs {
        for &b in term {
            all_letters.insert(b);
        }
    }
    let mut reduce_letters: Vec<u8> = Vec::new();
    for &b in &all_letters {
        if !parsed.output.contains(&b) {
            reduce_letters.push(b);
        }
    }
    reduce_letters.sort_unstable();

    let n_out: usize = out_shape.iter().product::<usize>().max(1);
    let n_red: usize = reduce_letters
        .iter()
        .map(|b| sizes[b])
        .product::<usize>()
        .max(1);

    let mut output = vec![0.0f32; n_out];

    // Enumerate every output position, then every reduction
    // assignment; compute the product across operands and accumulate.
    for out_idx in 0..n_out {
        // Decode out_idx into per-output-letter coords.
        let mut out_coord: Vec<usize> = Vec::with_capacity(parsed.output.len());
        let mut rem = out_idx;
        // Compute output strides on the fly.
        let mut out_strides = vec![1usize; parsed.output.len().max(1)];
        for d in (0..parsed.output.len().saturating_sub(1)).rev() {
            out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
        }
        for d in 0..parsed.output.len() {
            out_coord.push(rem / out_strides[d]);
            rem %= out_strides[d];
        }
        let mut letter_val: std::collections::HashMap<u8, usize> =
            std::collections::HashMap::new();
        for (i, &b) in parsed.output.iter().enumerate() {
            letter_val.insert(b, out_coord[i]);
        }

        let mut acc = 0.0f32;
        for r_idx in 0..n_red {
            // Decode r_idx into reduction-letter coords.
            let mut rem = r_idx;
            for (i, &b) in reduce_letters.iter().enumerate() {
                let sz_after: usize = reduce_letters[i + 1..]
                    .iter()
                    .map(|c| sizes[c])
                    .product::<usize>()
                    .max(1);
                let v = (rem / sz_after) % sizes[&b];
                rem %= sz_after;
                letter_val.insert(b, v);
            }
            // Compute the product of operand values at this letter
            // assignment.
            let mut prod = 1.0f32;
            for (op_i, term) in parsed.inputs.iter().enumerate() {
                let mut off = 0usize;
                for (axis_i, &b) in term.iter().enumerate() {
                    off += letter_val[&b] * op_strides[op_i][axis_i];
                }
                prod *= flats[op_i][off];
            }
            acc += prod;
        }
        output[out_idx] = acc;
    }

    let out_bytes = write_f32_into(dtype, &output);
    let cpu_out = CpuStorage::from_bytes(dtype, out_bytes)?;
    let storage: Storage = Arc::new(cpu_out);
    Tensor::from_parts(
        storage,
        Layout::contiguous(out_shape),
        TensorId::next(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_f32(t: &Tensor) -> Vec<f32> {
        let cpu = t.storage().as_any().downcast_ref::<CpuStorage>().unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn einsum_matmul() {
        // A [2, 3] @ B [3, 2] = C [2, 2]
        let a = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        )
        .unwrap();
        let b = Tensor::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3, 2],
        )
        .unwrap();
        let c = einsum("ij,jk->ik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        // C = [[1*1+2*3+3*5, 1*2+2*4+3*6], [4*1+5*3+6*5, 4*2+5*4+6*6]]
        //   = [[22, 28], [49, 64]]
        assert_eq!(read_f32(&c), vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn einsum_outer_product() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[10.0f32, 20.0], vec![2]).unwrap();
        let c = einsum("i,j->ij", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(read_f32(&c), vec![10.0, 20.0, 20.0, 40.0, 30.0, 60.0]);
    }

    #[test]
    fn einsum_inner_product() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0], vec![3]).unwrap();
        let c = einsum("i,i->", &[&a, &b]).unwrap();
        assert_eq!(c.shape() as &[usize], &[] as &[usize]);
        assert_eq!(read_f32(&c), vec![32.0]);
    }

    #[test]
    fn einsum_batched_matmul() {
        // [2, 2, 3] x [2, 3, 2] -> [2, 2, 2]
        let a_data: Vec<f32> = (0..12).map(|i| (i + 1) as f32).collect();
        let b_data: Vec<f32> = (0..12).map(|i| (i + 1) as f32).collect();
        let a = Tensor::from_slice(&a_data, vec![2, 2, 3]).unwrap();
        let b = Tensor::from_slice(&b_data, vec![2, 3, 2]).unwrap();
        let c = einsum("bij,bjk->bik", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);
        // First batch: A=[[1,2,3],[4,5,6]], B=[[1,2],[3,4],[5,6]]
        //   = [[22,28],[49,64]]
        // Second batch: A=[[7,8,9],[10,11,12]], B=[[7,8],[9,10],[11,12]]
        //   = [[7*7+8*9+9*11, 7*8+8*10+9*12], [10*7+11*9+12*11, 10*8+11*10+12*12]]
        //   = [[220, 244], [301, 334]]
        assert_eq!(
            read_f32(&c),
            vec![22.0, 28.0, 49.0, 64.0, 220.0, 244.0, 301.0, 334.0]
        );
    }

    #[test]
    fn einsum_reduce_to_scalar_via_axis() {
        // Sum a vector via einsum "i->"
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let c = einsum("i->", &[&a]).unwrap();
        assert_eq!(read_f32(&c), vec![10.0]);
    }

    #[test]
    fn einsum_per_row_dot() {
        // [2, 3] · [2, 3] per row → [2]
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0], vec![2, 3]).unwrap();
        let c = einsum("bi,bi->b", &[&a, &b]).unwrap();
        assert_eq!(c.shape(), &[2]);
        // Row 0: 1+2+3 = 6; row 1: 8+10+12 = 30
        assert_eq!(read_f32(&c), vec![6.0, 30.0]);
    }

    #[test]
    fn einsum_invalid_spec_errors() {
        let a = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let e = einsum("ij", &[&a]).unwrap_err();
        assert!(e.to_string().contains("'->'"));
    }

    #[test]
    fn einsum_axis_size_mismatch_errors() {
        // i bound to two different sizes
        let a = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0], vec![3]).unwrap();
        let e = einsum("i,i->", &[&a, &b]).unwrap_err();
        assert!(e.to_string().contains("two sizes"));
    }

    #[test]
    fn einsum_term_rank_mismatch_errors() {
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        // Spec says rank-3 but operand is rank-2.
        let e = einsum("ijk->", &[&a]).unwrap_err();
        assert!(e.to_string().contains("term length"));
    }
}
