use super::*;

/// State for Gated DeltaNet linear attention layers.
///
/// Each linear attention layer maintains:
/// - A recurrent state matrix S of shape `[batch, num_value_heads, key_head_dim, value_head_dim]`
/// - A conv1d sliding window buffer of shape `[batch, conv_dim, kernel_size - 1]`
///
/// This state is O(1) in sequence length — it does not grow with the number of tokens processed.
pub struct LinearAttentionState {
    /// Per-layer recurrent state S. Length = number of linear attention layers.
    pub recurrent_states: Vec<Tensor>,
    /// Per-layer conv1d sliding window buffers. Length = number of linear attention layers.
    pub conv_states: Vec<Tensor>,
}

/// Vulkan's tensor-keyed GDN registry normalizes recurrent and convolution
/// state to f32 regardless of the logical tensor dtype. Batch row transfers
/// must therefore use the resident representation's stride, not the host
/// tensor's BF16/F16 stride.
pub(super) fn resident_gdn_f32_row_bytes(tensor: &Tensor, label: &'static str) -> Result<u64> {
    tensor
        .elem_count()
        .checked_mul(std::mem::size_of::<f32>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .with_context(|| format!("linear-attention {label} resident row byte count overflow"))
}

/// Build a linear-attention snapshot without releasing partially submitted
/// device-copy destinations on either an error or a panic.
///
/// The destination vectors live outside the unwind boundary and are only
/// borrowed by `build`. That lets this helper retain every completed tensor
/// before resuming the panic into the serving layer's ownership fence, which
/// quarantines the backend and converts it into a request error.
pub(super) fn build_linear_attention_snapshot<T>(
    mut recurrent_states: Vec<T>,
    mut conv_states: Vec<T>,
    build: impl FnOnce(&mut Vec<T>, &mut Vec<T>) -> Result<()>,
) -> Result<(Vec<T>, Vec<T>)> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        build(&mut recurrent_states, &mut conv_states)
    }));
    match result {
        Ok(Ok(())) => Ok((recurrent_states, conv_states)),
        Ok(Err(error)) => {
            std::mem::forget(recurrent_states);
            std::mem::forget(conv_states);
            Err(error)
        }
        Err(payload) => {
            std::mem::forget(recurrent_states);
            std::mem::forget(conv_states);
            std::panic::resume_unwind(payload)
        }
    }
}

impl LinearAttentionState {
    /// Create fresh zero-initialized state for all linear attention layers.
    pub fn new(config: &kiln_core::config::ModelConfig, device: &Device) -> Result<Self> {
        Self::new_with_batch_and_recurrent_dtype(
            config,
            1,
            device,
            Self::training_recurrent_dtype(config, device),
        )
    }

    /// Create fresh inference state for all linear attention layers.
    ///
    /// CUDA/Metal inference and explicitly named Vulkan inference use the same
    /// dtype as the model weights so decode does not cast every GDN recurrent
    /// state into and back out of the hot kernel dtype on every token. `new`
    /// keeps the training/test default.
    pub fn new_for_inference(
        config: &kiln_core::config::ModelConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_batch_for_inference(config, 1, device)
    }

    /// Create fresh inference state for `batch` independent decode rows.
    pub fn new_with_batch_for_inference(
        config: &kiln_core::config::ModelConfig,
        batch: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_batch_for_inference_backend(config, batch, device, None)
    }

    /// Create fresh inference state for `batch` decode rows, allowing callers
    /// whose accelerator is not represented by Candle's `Device` enum to name
    /// the backend explicitly.
    pub fn new_with_batch_for_inference_backend(
        config: &kiln_core::config::ModelConfig,
        batch: usize,
        device: &Device,
        backend_name: Option<&str>,
    ) -> Result<Self> {
        let policy =
            InferenceRecurrentStatePolicy::for_backend(backend_name.unwrap_or_default(), *device);
        Self::new_with_batch_for_inference_policy(config, batch, device, policy)
    }

    /// Create fresh inference state for `batch` decode rows using the active
    /// backend's capability snapshot as the dtype policy authority.
    pub fn new_with_batch_for_inference_runtime(
        config: &kiln_core::config::ModelConfig,
        batch: usize,
        device: &Device,
        backend: &dyn BackendRuntime,
    ) -> Result<Self> {
        let policy = BackendCapabilityQueries::backend_capabilities(backend)
            .gdn
            .inference_recurrent_state;
        Self::new_with_batch_for_inference_policy(config, batch, device, policy)
    }

    fn new_with_batch_for_inference_policy(
        config: &kiln_core::config::ModelConfig,
        batch: usize,
        device: &Device,
        policy: InferenceRecurrentStatePolicy,
    ) -> Result<Self> {
        Self::new_with_batch_and_recurrent_dtype(
            config,
            batch,
            device,
            Self::inference_recurrent_dtype(config, device, policy),
        )
    }

    /// Create fresh zero-initialized state for all linear attention layers and
    /// `batch` independent decode rows.
    pub fn new_with_batch(
        config: &kiln_core::config::ModelConfig,
        batch: usize,
        device: &Device,
    ) -> Result<Self> {
        Self::new_with_batch_and_recurrent_dtype(
            config,
            batch,
            device,
            Self::training_recurrent_dtype(config, device),
        )
    }

    fn training_recurrent_dtype(config: &kiln_core::config::ModelConfig, device: &Device) -> DType {
        match (device, config.dtype) {
            (Device::Metal(_), kiln_core::config::DType::BF16) => DType::BF16,
            (Device::Metal(_), kiln_core::config::DType::FP16) => DType::F16,
            _ => DType::F32,
        }
    }

    fn inference_recurrent_dtype(
        config: &kiln_core::config::ModelConfig,
        device: &Device,
        policy: InferenceRecurrentStatePolicy,
    ) -> DType {
        match config.dtype {
            kiln_core::config::DType::BF16
                if matches!(
                    policy.bf16,
                    Support::Native | Support::NativeWithConstraints
                ) =>
            {
                DType::BF16
            }
            kiln_core::config::DType::FP16
                if matches!(policy.f16, Support::Native | Support::NativeWithConstraints) =>
            {
                DType::F16
            }
            _ => Self::training_recurrent_dtype(config, device),
        }
    }

    fn new_with_batch_and_recurrent_dtype(
        config: &kiln_core::config::ModelConfig,
        batch: usize,
        device: &Device,
        recurrent_dtype: DType,
    ) -> Result<Self> {
        anyhow::ensure!(batch > 0, "LinearAttentionState batch must be positive");
        let num_linear_layers = config.num_layers - config.num_full_attention_layers;
        let nv = config.linear_num_value_heads;
        let dk = config.linear_key_head_dim;
        let dv = config.linear_value_head_dim;
        let conv_dim = config.linear_qkv_dim();
        let k_minus_1 = config.linear_conv_kernel_dim.saturating_sub(1);

        let mut recurrent_states = Vec::with_capacity(num_linear_layers);
        let mut conv_states = Vec::with_capacity(num_linear_layers);

        for _ in 0..num_linear_layers {
            // kt `zeros` takes shape as `Into<Vec<usize>>` and `Device` by
            // value (kt Device is Copy) (#1082 forward-flip).
            recurrent_states.push(Tensor::zeros(
                vec![batch, nv, dk, dv],
                recurrent_dtype,
                *device,
            )?);
            conv_states.push(Tensor::zeros(
                vec![batch, conv_dim, k_minus_1],
                DType::F32,
                *device,
            )?);
        }

        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }

    /// Return the shared batch dimension across all recurrent and conv states.
    pub fn batch_size(&self) -> Result<usize> {
        if self.recurrent_states.len() != self.conv_states.len() {
            anyhow::bail!(
                "LinearAttentionState batch_size: recurrent/conv layer count mismatch ({} vs {})",
                self.recurrent_states.len(),
                self.conv_states.len()
            );
        }

        let first = self
            .recurrent_states
            .first()
            .context("LinearAttentionState batch_size: no recurrent states")?;
        let batch = first.dim(0)?;
        for (idx, tensor) in self.recurrent_states.iter().enumerate() {
            anyhow::ensure!(
                tensor.dim(0)? == batch,
                "LinearAttentionState batch_size: recurrent state {idx} batch mismatch"
            );
        }
        for (idx, tensor) in self.conv_states.iter().enumerate() {
            anyhow::ensure!(
                tensor.dim(0)? == batch,
                "LinearAttentionState batch_size: conv state {idx} batch mismatch"
            );
        }
        Ok(batch)
    }

    /// Assemble a batched GDN state from one-row per-request states.
    pub fn from_batch_rows(rows: &[&Self]) -> Result<Self> {
        anyhow::ensure!(
            !rows.is_empty(),
            "LinearAttentionState::from_batch_rows requires at least one row"
        );
        let num_layers = rows[0].recurrent_states.len();
        anyhow::ensure!(
            rows[0].conv_states.len() == num_layers,
            "LinearAttentionState::from_batch_rows row 0 recurrent/conv layer count mismatch"
        );

        for (idx, row) in rows.iter().enumerate() {
            anyhow::ensure!(
                row.recurrent_states.len() == num_layers && row.conv_states.len() == num_layers,
                "LinearAttentionState::from_batch_rows row {idx} layer count mismatch"
            );
            let row_batch = row.batch_size()?;
            anyhow::ensure!(
                row_batch == 1,
                "LinearAttentionState::from_batch_rows row {idx} has batch size {}, expected 1",
                row_batch
            );
        }

        // Defensive dtype normalization: rows must share a dtype for `Tensor::cat`.
        // The canonical recurrent dtype is whatever `new_with_batch` produced for
        // row 0 (F32 on CUDA, BF16/F16 on Metal). If any other row drifted (e.g. a
        // prior decode error left state mid-conversion in BF16), cast it back to
        // row 0's dtype so cat succeeds. Same for conv state.
        let mut recurrent_states = Vec::with_capacity(num_layers);
        let mut conv_states = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let target_recurrent_dtype = rows[0].recurrent_states[layer_idx].dtype();
            let mut recurrent_owned: Vec<Tensor> = Vec::with_capacity(rows.len());
            for (row_idx, row) in rows.iter().enumerate() {
                let t = &row.recurrent_states[layer_idx];
                if t.dtype() != target_recurrent_dtype {
                    tracing::debug!(
                        layer = layer_idx,
                        row = row_idx,
                        from = ?t.dtype(),
                        to = ?target_recurrent_dtype,
                        "from_batch_rows: normalizing recurrent state dtype before cat"
                    );
                    recurrent_owned.push(t.to_dtype(target_recurrent_dtype)?);
                } else {
                    recurrent_owned.push(t.clone());
                }
            }
            let recurrent_refs: Vec<&Tensor> = recurrent_owned.iter().collect();

            let target_conv_dtype = rows[0].conv_states[layer_idx].dtype();
            let mut conv_owned: Vec<Tensor> = Vec::with_capacity(rows.len());
            for (row_idx, row) in rows.iter().enumerate() {
                let t = &row.conv_states[layer_idx];
                if t.dtype() != target_conv_dtype {
                    tracing::debug!(
                        layer = layer_idx,
                        row = row_idx,
                        from = ?t.dtype(),
                        to = ?target_conv_dtype,
                        "from_batch_rows: normalizing conv state dtype before cat"
                    );
                    conv_owned.push(t.to_dtype(target_conv_dtype)?);
                } else {
                    conv_owned.push(t.clone());
                }
            }
            let conv_refs: Vec<&Tensor> = conv_owned.iter().collect();

            // `Tensor::cat` already produces a contiguous output tensor, so
            // the trailing `.contiguous()` was a no-op that nevertheless
            // re-checked the layout on every cat — and on the hot decode
            // path that meant one redundant CPU-side check per (layer, step,
            // state-kind) tuple = 24 GDN layers × 2 states × steps. Removing
            // it shaves a small amount of dispatch overhead off the
            // `batch_state_assemble` stage. The runtime path is identical
            // because cat is the only source feeding these tensors.
            //
            // Phase 7 (#1082): when stable KT routes are enabled and every input is a
            // contiguous CUDA tensor of a supported dtype, route the
            // axis-0 concat through `kiln_tensor::cuda_concat(_, 0)`.
            // Falls through to the candle composite when any
            // precondition fails so behavior is identical with the
            // gate off.
            let recurrent_cat = {
                #[cfg(feature = "cuda")]
                {
                    if let Some(out) = try_kt_cat_dim0(&recurrent_refs)? {
                        out
                    } else {
                        Tensor::cat(&recurrent_refs, 0)?
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Tensor::cat(&recurrent_refs, 0)?
                }
            };
            let conv_cat = {
                #[cfg(feature = "cuda")]
                {
                    if let Some(out) = try_kt_cat_dim0(&conv_refs)? {
                        out
                    } else {
                        Tensor::cat(&conv_refs, 0)?
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Tensor::cat(&conv_refs, 0)?
                }
            };
            recurrent_states.push(recurrent_cat);
            conv_states.push(conv_cat);
        }

        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }

    /// Return a logical prefix of a resident batched state without changing
    /// the tensor IDs used by the backend residency registry.
    ///
    /// The returned tensors share CPU storage and mutation versions with this
    /// state. The backend may therefore keep one maximum-capacity physical
    /// state while decode executes a smaller logical batch. This view is valid
    /// only while `self` remains owned by the caller.
    pub(crate) fn resident_batch_prefix_view(&self, batch: usize) -> Result<Self> {
        let capacity = self.batch_size()?;
        anyhow::ensure!(batch > 0, "resident batch prefix must be non-zero");
        anyhow::ensure!(
            batch <= capacity,
            "resident batch prefix {batch} exceeds capacity {capacity}"
        );
        anyhow::ensure!(
            self.recurrent_states.len() == self.conv_states.len(),
            "resident batch prefix recurrent/conv layer count mismatch"
        );

        let recurrent_states = self
            .recurrent_states
            .iter()
            .map(|state| {
                state
                    .batch_prefix_preserving_identity_for_residency(batch)
                    .map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        let conv_states = self
            .conv_states
            .iter()
            .map(|state| {
                state
                    .batch_prefix_preserving_identity_for_residency(batch)
                    .map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }

    /// Extract one batch row into storage that cannot alias the batch tensor.
    ///
    /// A one-row narrow of a contiguous tensor can itself be contiguous, so
    /// `contiguous()` is not an ownership boundary here.
    pub(super) fn detached_batch_row(tensor: &Tensor, batch_idx: usize) -> Result<Tensor> {
        Ok(tensor.narrow(0, batch_idx, 1)?.copy()?)
    }

    /// Split a batched state into one-row states in batch order.
    pub fn split_batch_rows(&self) -> Result<Vec<Self>> {
        let batch = self.batch_size()?;
        let mut rows = Vec::with_capacity(batch);
        for batch_idx in 0..batch {
            let mut recurrent_states = Vec::with_capacity(self.recurrent_states.len());
            let mut conv_states = Vec::with_capacity(self.conv_states.len());
            for tensor in &self.recurrent_states {
                recurrent_states.push(Self::detached_batch_row(tensor, batch_idx)?);
            }
            for tensor in &self.conv_states {
                conv_states.push(Self::detached_batch_row(tensor, batch_idx)?);
            }
            rows.push(Self {
                recurrent_states,
                conv_states,
            });
        }
        Ok(rows)
    }

    /// Overwrite one-row destination states from the rows of this batched state.
    pub fn scatter_batch_rows(&self, destinations: &mut [&mut Self]) -> Result<()> {
        let rows = self.split_batch_rows()?;
        anyhow::ensure!(
            destinations.len() == rows.len(),
            "LinearAttentionState::scatter_batch_rows destination count mismatch ({} vs {})",
            destinations.len(),
            rows.len()
        );
        for (dst, row) in destinations.iter_mut().zip(rows.iter()) {
            dst.restore_from(row)?;
        }
        Ok(())
    }

    /// Replace one-row destination tensors with independent copies from this
    /// batched state.
    ///
    /// This avoids the extra `restore_from` copies in [`Self::scatter_batch_rows`]
    /// for scheduler-owned decode rows. Graph pointer stability belongs to the
    /// runner-owned batched slot; per-request row state must own its storage.
    pub fn scatter_batch_rows_replace(&self, destinations: &mut [&mut Self]) -> Result<()> {
        let batch = self.batch_size()?;
        anyhow::ensure!(
            destinations.len() == batch,
            "LinearAttentionState::scatter_batch_rows_replace destination count mismatch ({} vs {})",
            destinations.len(),
            batch
        );

        for (row_idx, dst) in destinations.iter_mut().enumerate() {
            anyhow::ensure!(
                dst.recurrent_states.len() == self.recurrent_states.len(),
                "LinearAttentionState::scatter_batch_rows_replace recurrent layer count mismatch for row {row_idx} ({} vs {})",
                dst.recurrent_states.len(),
                self.recurrent_states.len()
            );
            anyhow::ensure!(
                dst.conv_states.len() == self.conv_states.len(),
                "LinearAttentionState::scatter_batch_rows_replace conv layer count mismatch for row {row_idx} ({} vs {})",
                dst.conv_states.len(),
                self.conv_states.len()
            );

            for (dst_tensor, src_tensor) in dst
                .recurrent_states
                .iter_mut()
                .zip(self.recurrent_states.iter())
            {
                *dst_tensor = Self::detached_batch_row(src_tensor, row_idx)?;
            }
            for (dst_tensor, src_tensor) in dst.conv_states.iter_mut().zip(self.conv_states.iter())
            {
                *dst_tensor = Self::detached_batch_row(src_tensor, row_idx)?;
            }
        }

        Ok(())
    }

    /// Assemble backend-resident recurrent row buffers into this batched state.
    ///
    /// The CPU tensors still carry the same shapes/dtypes as the portable
    /// path, but a backend may bind device-resident state buffers to their
    /// tensor IDs so the decode recurrent kernel can avoid re-uploading stale
    /// CPU state.
    pub fn assemble_gdn_recurrent_resident_batch_rows(
        &self,
        backend: &dyn BackendRuntime,
        rows: &[&Self],
    ) -> Result<bool> {
        let batch = self.batch_size()?;
        anyhow::ensure!(
            rows.len() == batch,
            "LinearAttentionState::assemble_gdn_recurrent_resident_batch_rows row count mismatch ({} vs {})",
            rows.len(),
            batch
        );
        let mut assembled_any = false;
        for layer_idx in 0..self.recurrent_states.len() {
            let row_tensors: Vec<&Tensor> = rows
                .iter()
                .map(|row| &row.recurrent_states[layer_idx])
                .collect();
            assembled_any |= ResidencyBackend::runtime_assemble_gdn_recurrent_resident_batch_rows(
                backend,
                &row_tensors,
                &self.recurrent_states[layer_idx],
            )?;
        }
        Ok(assembled_any)
    }

    /// Assemble backend-resident recurrent + conv row buffers into this
    /// batched state using kt tensor IDs only.
    pub fn assemble_gdn_state_resident_batch_rows_kt(
        &self,
        backend: &dyn BackendRuntime,
        rows: &[&Self],
    ) -> Result<bool> {
        let batch = self.batch_size()?;
        anyhow::ensure!(
            rows.len() == batch,
            "LinearAttentionState::assemble_gdn_state_resident_batch_rows_kt row count mismatch ({} vs {})",
            rows.len(),
            batch
        );
        anyhow::ensure!(
            !rows.is_empty(),
            "LinearAttentionState::assemble_gdn_state_resident_batch_rows_kt requires at least one row"
        );
        let mut assembled_any = false;
        for layer_idx in 0..self.recurrent_states.len() {
            let row_keys: Vec<kiln_tensor::TensorId> = rows
                .iter()
                .map(|row| row.recurrent_states[layer_idx].id())
                .collect();
            let batch_key = self.recurrent_states[layer_idx].id();
            let recurrent_row = &rows[0].recurrent_states[layer_idx];
            let conv_row = &rows[0].conv_states[layer_idx];
            let recurrent_row_bytes = resident_gdn_f32_row_bytes(recurrent_row, "recurrent")?;
            let conv_row_bytes = resident_gdn_f32_row_bytes(conv_row, "convolution")?;
            assembled_any |= ResidencyBackend::runtime_assemble_linear_attn_gdn_state_batch_kt(
                backend,
                &row_keys,
                batch_key,
                recurrent_row_bytes,
                conv_row_bytes,
            )?;
        }
        Ok(assembled_any)
    }

    /// Refresh THIS batched state's recurrent + conv tensors *in place*
    /// from the supplied per-row states, preserving device pointers.
    /// Required by the multi-batch CUDA graph replay path: the captured
    /// graph holds the persistent slot's device addresses, so refreshing
    /// must not replace the tensors.
    ///
    /// Uses [`Tensor::slice_set`] per row + per layer, which writes the
    /// source bytes into the destination's existing storage. After this
    /// call, `self.recurrent_states[layer_idx][row]` byte-matches
    /// `rows[row].recurrent_states[layer_idx]`, same for `conv_states`.
    ///
    /// The inverse direction (persistent → per-row, e.g. after a graph
    /// replay) still uses [`Self::scatter_batch_rows_replace_with_backend`]
    /// which is allowed to replace per-row tensors — only the batched
    /// slot's pointers must stay pinned.
    pub fn refresh_batched_state_from_rows_in_place(&mut self, rows: &[&Self]) -> Result<()> {
        let batch = self.batch_size()?;
        anyhow::ensure!(
            rows.len() == batch,
            "refresh_batched_state_from_rows_in_place row count mismatch ({} vs {})",
            rows.len(),
            batch
        );
        anyhow::ensure!(
            rows.iter()
                .all(|r| r.recurrent_states.len() == self.recurrent_states.len()
                    && r.conv_states.len() == self.conv_states.len()),
            "refresh_batched_state_from_rows_in_place: per-row state layer-count mismatch"
        );
        for layer_idx in 0..self.recurrent_states.len() {
            for (row_idx, src) in rows.iter().enumerate() {
                self.recurrent_states[layer_idx]
                    .slice_set(&src.recurrent_states[layer_idx], 0, row_idx)
                    .with_context(|| {
                        format!(
                            "refresh recurrent state row {row_idx} into persistent batched slot at layer {layer_idx}"
                        )
                    })?;
            }
        }
        for layer_idx in 0..self.conv_states.len() {
            for (row_idx, src) in rows.iter().enumerate() {
                self.conv_states[layer_idx]
                    .slice_set(&src.conv_states[layer_idx], 0, row_idx)
                    .with_context(|| {
                        format!(
                            "refresh conv state row {row_idx} into persistent batched slot at layer {layer_idx}"
                        )
                    })?;
            }
        }
        Ok(())
    }

    /// Replace one-row destination tensors, preserving backend-resident
    /// recurrent state when the backend owns a fresher batch buffer.
    pub fn scatter_batch_rows_replace_with_backend(
        &self,
        backend: &dyn BackendRuntime,
        destinations: &mut [&mut Self],
    ) -> Result<()> {
        let batch = self.batch_size()?;
        anyhow::ensure!(
            destinations.len() == batch,
            "LinearAttentionState::scatter_batch_rows_replace_with_backend destination count mismatch ({} vs {})",
            destinations.len(),
            batch
        );

        for (row_idx, dst) in destinations.iter_mut().enumerate() {
            anyhow::ensure!(
                dst.recurrent_states.len() == self.recurrent_states.len(),
                "LinearAttentionState::scatter_batch_rows_replace_with_backend recurrent layer count mismatch for row {row_idx} ({} vs {})",
                dst.recurrent_states.len(),
                self.recurrent_states.len()
            );
            anyhow::ensure!(
                dst.conv_states.len() == self.conv_states.len(),
                "LinearAttentionState::scatter_batch_rows_replace_with_backend conv layer count mismatch for row {row_idx} ({} vs {})",
                dst.conv_states.len(),
                self.conv_states.len()
            );
        }

        for layer_idx in 0..self.recurrent_states.len() {
            let mut dst_tensors: Vec<&mut Tensor> = destinations
                .iter_mut()
                .map(|dst| &mut dst.recurrent_states[layer_idx])
                .collect();
            if !ResidencyBackend::runtime_scatter_gdn_recurrent_resident_batch_rows(
                backend,
                &self.recurrent_states[layer_idx],
                &mut dst_tensors,
            )? {
                for (row_idx, dst_tensor) in dst_tensors.into_iter().enumerate() {
                    *dst_tensor =
                        Self::detached_batch_row(&self.recurrent_states[layer_idx], row_idx)?;
                }
            }
        }

        for layer_idx in 0..self.conv_states.len() {
            for (row_idx, dst) in destinations.iter_mut().enumerate() {
                dst.conv_states[layer_idx] =
                    Self::detached_batch_row(&self.conv_states[layer_idx], row_idx)?;
            }
        }

        Ok(())
    }

    /// Scatter backend-resident batched recurrent + conv buffers back into
    /// per-row resident buffers using kt tensor IDs only.
    pub fn scatter_gdn_state_resident_batch_rows_kt(
        &self,
        backend: &dyn BackendRuntime,
        destinations: &mut [&mut Self],
    ) -> Result<bool> {
        let batch = self.batch_size()?;
        anyhow::ensure!(
            destinations.len() == batch,
            "LinearAttentionState::scatter_gdn_state_resident_batch_rows_kt destination count mismatch ({} vs {})",
            destinations.len(),
            batch
        );
        anyhow::ensure!(
            !destinations.is_empty(),
            "LinearAttentionState::scatter_gdn_state_resident_batch_rows_kt requires at least one destination"
        );
        for (row_idx, dst) in destinations.iter_mut().enumerate() {
            anyhow::ensure!(
                dst.recurrent_states.len() == self.recurrent_states.len(),
                "LinearAttentionState::scatter_gdn_state_resident_batch_rows_kt recurrent layer count mismatch for row {row_idx} ({} vs {})",
                dst.recurrent_states.len(),
                self.recurrent_states.len()
            );
            anyhow::ensure!(
                dst.conv_states.len() == self.conv_states.len(),
                "LinearAttentionState::scatter_gdn_state_resident_batch_rows_kt conv layer count mismatch for row {row_idx} ({} vs {})",
                dst.conv_states.len(),
                self.conv_states.len()
            );
        }

        let mut scattered_any = false;
        for layer_idx in 0..self.recurrent_states.len() {
            let row_keys: Vec<kiln_tensor::TensorId> = destinations
                .iter()
                .map(|dst| dst.recurrent_states[layer_idx].id())
                .collect();
            let batch_key = self.recurrent_states[layer_idx].id();
            let recurrent_row = &destinations[0].recurrent_states[layer_idx];
            let conv_row = &destinations[0].conv_states[layer_idx];
            let recurrent_row_bytes = resident_gdn_f32_row_bytes(recurrent_row, "recurrent")?;
            let conv_row_bytes = resident_gdn_f32_row_bytes(conv_row, "convolution")?;
            scattered_any |= ResidencyBackend::runtime_scatter_linear_attn_gdn_state_batch_kt(
                backend,
                batch_key,
                &row_keys,
                recurrent_row_bytes,
                conv_row_bytes,
            )?;
        }
        Ok(scattered_any)
    }

    pub fn materialize_gdn_recurrent_resident_states(
        &mut self,
        backend: &dyn BackendRuntime,
    ) -> Result<()> {
        for state in &mut self.recurrent_states {
            ResidencyBackend::runtime_materialize_gdn_recurrent_resident_state(backend, state)?;
        }
        Ok(())
    }

    pub fn materialize_gdn_prefill_resident_states(
        &mut self,
        backend: &dyn BackendRuntime,
        owner_id: u64,
    ) -> Result<()> {
        for (layer_idx, state) in self.recurrent_states.iter_mut().enumerate() {
            ResidencyBackend::runtime_materialize_gdn_prefill_resident_state(
                backend, owner_id, layer_idx, state,
            )?;
        }
        Ok(())
    }

    pub fn apply_gdn_prefill_resident_state_boundary(
        &mut self,
        backend: &dyn BackendRuntime,
        owner_id: u64,
    ) -> Result<()> {
        for (layer_idx, state) in self.recurrent_states.iter_mut().enumerate() {
            ResidencyBackend::runtime_apply_gdn_prefill_resident_state_boundary(
                backend, owner_id, layer_idx, state,
            )?;
        }
        Ok(())
    }

    pub fn evict_gdn_recurrent_resident_states(&self, backend: &dyn BackendRuntime) {
        for state in &self.recurrent_states {
            ResidencyBackend::runtime_evict_gdn_recurrent_resident_state(backend, state);
        }
    }

    pub fn has_any_gdn_recurrent_resident_state(&self, backend: &dyn BackendRuntime) -> bool {
        self.recurrent_states
            .iter()
            .any(|state| ResidencyBackend::runtime_has_gdn_recurrent_resident_state(backend, state))
    }

    pub fn has_all_gdn_recurrent_resident_states(&self, backend: &dyn BackendRuntime) -> bool {
        !self.recurrent_states.is_empty()
            && self.recurrent_states.iter().all(|state| {
                ResidencyBackend::runtime_has_gdn_recurrent_resident_state(backend, state)
            })
    }

    pub fn has_any_gdn_state_resident_kt(&self, backend: &dyn BackendRuntime) -> bool {
        self.recurrent_states.iter().any(|state| {
            ResidencyBackend::runtime_has_linear_attn_gdn_state_kt(backend, state.id())
        })
    }

    pub fn has_all_gdn_state_resident_kt(&self, backend: &dyn BackendRuntime) -> bool {
        !self.recurrent_states.is_empty()
            && self.recurrent_states.iter().all(|state| {
                ResidencyBackend::runtime_has_linear_attn_gdn_state_kt(backend, state.id())
            })
    }

    pub fn ensure_gdn_state_resident_kt(&self, backend: &dyn BackendRuntime) -> Result<bool> {
        anyhow::ensure!(
            self.conv_states.len() == self.recurrent_states.len(),
            "LinearAttentionState::ensure_gdn_state_resident_kt recurrent/conv layer count mismatch"
        );
        if self.has_all_gdn_state_resident_kt(backend) {
            return Ok(false);
        }
        let mut seeded_any = false;
        for layer_idx in 0..self.recurrent_states.len() {
            seeded_any |= ResidencyBackend::runtime_seed_linear_attn_gdn_state_kt(
                backend,
                &self.recurrent_states[layer_idx],
                &self.conv_states[layer_idx],
            )?;
        }
        Ok(seeded_any)
    }

    /// Release kt-resident recurrent and convolution buffers owned by this
    /// state. This is separate from the decode-step resident-state eviction:
    /// the kt maps intentionally survive between tokens and are released only
    /// when the request or assembled-batch owner is retired.
    pub fn evict_gdn_state_resident_kt(&self, backend: &dyn BackendRuntime) {
        for state in &self.recurrent_states {
            ResidencyBackend::runtime_evict_linear_attn_gdn_state_kt(backend, state.id());
        }
    }

    /// Capture the current GDN recurrent + conv state into a fresh shadow
    /// `LinearAttentionState`. Used by speculative decoding to preserve the
    /// base model's O(1) GDN state before advancing into a draft: if any
    /// proposed token is rejected, [`Self::restore_from`] puts it back.
    ///
    /// This snapshot allocates new device tensors and issues a
    /// `cudaMemcpyDeviceToDevice` per layer. For Qwen3.5-4B that is
    /// 24 × (recurrent ≈ 2 MiB + conv ≈ 24 KiB) ≈ 49 MiB per snapshot, which
    /// is acceptable for WIP scaffolding. The follow-up PR replaces this with
    /// the ping-pong shadow-slot pattern from the existing KV-cache draft
    /// code path (no per-step alloc, two pre-allocated slots swapped via
    /// index) to bring overhead to zero.
    pub fn snapshot(&self) -> Result<Self> {
        let (recurrent_states, conv_states) = build_linear_attention_snapshot(
            Vec::with_capacity(self.recurrent_states.len()),
            Vec::with_capacity(self.conv_states.len()),
            |recurrent_states, conv_states| {
                for tensor in &self.recurrent_states {
                    recurrent_states.push(tensor.copy().context("snapshot recurrent state")?);
                }
                for tensor in &self.conv_states {
                    conv_states.push(tensor.copy().context("snapshot conv state")?);
                }
                Ok(())
            },
        )?;
        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }

    /// Snapshot for decode rollback.
    ///
    /// Recurrent tensors are replaced on update, so Arc-cloning their handles
    /// preserves the pre-step value without a device copy. Conv state is mutated
    /// in-place by the Metal/CUDA update kernels, so it must still be copied.
    pub fn snapshot_for_decode_rollback(&self) -> Result<Self> {
        self.snapshot_for_decode_rollback_prefix(self.recurrent_states.len())
    }

    /// Snapshot only the linear-attention prefix needed by a draft model.
    ///
    /// Skip-layer drafting runs `model_forward_segment(..., 0, draft_layers)`,
    /// so it never touches GDN states after that layer prefix. Carrying all
    /// 24 Qwen3.5-4B GDN states in the draft snapshot wastes device copies on
    /// every speculative step.
    pub fn snapshot_for_decode_rollback_prefix(&self, num_linear_layers: usize) -> Result<Self> {
        if num_linear_layers > self.recurrent_states.len() {
            anyhow::bail!(
                "LinearAttentionState::snapshot_for_decode_rollback_prefix: requested {} recurrent states, only {} available",
                num_linear_layers,
                self.recurrent_states.len()
            );
        }
        if num_linear_layers > self.conv_states.len() {
            anyhow::bail!(
                "LinearAttentionState::snapshot_for_decode_rollback_prefix: requested {} conv states, only {} available",
                num_linear_layers,
                self.conv_states.len()
            );
        }
        let recurrent_states = self.recurrent_states[..num_linear_layers].to_vec();
        let mut conv_states = Vec::with_capacity(num_linear_layers);
        for t in &self.conv_states[..num_linear_layers] {
            conv_states.push(t.copy().context("snapshot conv state")?);
        }
        Ok(Self {
            recurrent_states,
            conv_states,
        })
    }

    /// Restore this state from a previously captured [`Self::snapshot`].
    ///
    /// Checks that the shapes/counts match — a mismatch indicates the caller
    /// mixed up snapshots from different sessions, which would be a logic bug
    /// in the spec-decode loop. Overwrites the current tensors in place so
    /// downstream GPU pointers (e.g. those captured inside a CUDA graph) stay
    /// valid. The follow-up ping-pong rewrite folds this into a zero-copy
    /// slot swap; this correctness-first copy implementation is the scaffold.
    pub fn restore_from(&mut self, snapshot: &Self) -> Result<()> {
        if self.recurrent_states.len() != snapshot.recurrent_states.len() {
            anyhow::bail!(
                "LinearAttentionState::restore_from: recurrent_states len mismatch ({} vs {})",
                self.recurrent_states.len(),
                snapshot.recurrent_states.len()
            );
        }
        if self.conv_states.len() != snapshot.conv_states.len() {
            anyhow::bail!(
                "LinearAttentionState::restore_from: conv_states len mismatch ({} vs {})",
                self.conv_states.len(),
                snapshot.conv_states.len()
            );
        }
        for (dst, src) in self
            .recurrent_states
            .iter_mut()
            .zip(snapshot.recurrent_states.iter())
        {
            *dst = src.copy().context("restore recurrent state")?;
        }
        for (dst, src) in self.conv_states.iter_mut().zip(snapshot.conv_states.iter()) {
            *dst = src.copy().context("restore conv state")?;
        }
        Ok(())
    }

    /// Restore from [`Self::snapshot_for_decode_rollback`] without recopying
    /// recurrent state. The snapshot owns fresh conv-state copies, so assigning
    /// their tensor handles is enough to restore the old conv buffers as well.
    pub fn restore_from_decode_rollback(&mut self, snapshot: &Self) -> Result<()> {
        if self.recurrent_states.len() != snapshot.recurrent_states.len() {
            anyhow::bail!(
                "LinearAttentionState::restore_from_decode_rollback: recurrent_states len mismatch ({} vs {})",
                self.recurrent_states.len(),
                snapshot.recurrent_states.len()
            );
        }
        if self.conv_states.len() != snapshot.conv_states.len() {
            anyhow::bail!(
                "LinearAttentionState::restore_from_decode_rollback: conv_states len mismatch ({} vs {})",
                self.conv_states.len(),
                snapshot.conv_states.len()
            );
        }
        self.recurrent_states.clone_from(&snapshot.recurrent_states);
        self.conv_states.clone_from(&snapshot.conv_states);
        Ok(())
    }
}
