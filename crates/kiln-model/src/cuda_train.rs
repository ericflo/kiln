//! CUDA-native training tensor shell.
//!
//! This is the first CUDA-specific training-stack boundary. It deliberately
//! stays small: candle CUDA tensors remain the canonical storage, while this
//! wrapper enforces CUDA residency before calling backend-owned training
//! kernels. Higher-level native CUDA training can build on this without
//! accepting CPU tensors by accident.

use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Device, Tensor, TensorId};
use kiln_core::config::ModelConfig;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::forward::{
    GpuAttentionWeights, GpuFullAttentionWeights, GpuLayerWeights, GpuLinearAttentionWeights,
    GpuWeights,
};

/// Monotonic op-id allocator for the future CUDA-native training graph.
static NEXT_CUDA_TRAIN_OP_ID: AtomicU64 = AtomicU64::new(1);

pub fn next_cuda_train_op_id() -> u64 {
    NEXT_CUDA_TRAIN_OP_ID.fetch_add(1, Ordering::Relaxed)
}

/// Backward op interface for the future CUDA-native training graph.
pub trait CudaBackwardOp: Send + Sync + std::fmt::Debug {
    fn op_name(&self) -> &'static str;
    fn input_refs(&self) -> &[CudaTrainTensor];
    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>>;
}

#[derive(Debug, Clone)]
pub struct CudaTrainTensor {
    tensor: Tensor,
    requires_grad: bool,
    param_id: Option<TensorId>,
    op_id: u64,
    grad_fn: Option<Arc<dyn CudaBackwardOp>>,
}

impl CudaTrainTensor {
    pub fn new(tensor: Tensor) -> Result<Self> {
        Self::leaf(tensor, false, None)
    }

    pub fn parameter(tensor: Tensor, param_id: TensorId) -> Result<Self> {
        Self::leaf(tensor, true, Some(param_id))
    }

    fn leaf(tensor: Tensor, requires_grad: bool, param_id: Option<TensorId>) -> Result<Self> {
        ensure!(
            matches!(tensor.device(), Device::Cuda(_)),
            "CudaTrainTensor requires a CUDA tensor, got {:?}",
            tensor.device()
        );
        Ok(Self {
            tensor,
            requires_grad,
            param_id,
            op_id: next_cuda_train_op_id(),
            grad_fn: None,
        })
    }

    pub fn from_op(tensor: Tensor, grad_fn: Option<Arc<dyn CudaBackwardOp>>) -> Result<Self> {
        ensure!(
            matches!(tensor.device(), Device::Cuda(_)),
            "CudaTrainTensor::from_op requires a CUDA tensor, got {:?}",
            tensor.device()
        );
        let requires_grad = grad_fn.is_some();
        Ok(Self {
            tensor,
            requires_grad,
            param_id: None,
            op_id: next_cuda_train_op_id(),
            grad_fn,
        })
    }

    pub fn zeros_like(reference: &Tensor) -> Result<Self> {
        ensure!(
            matches!(reference.device(), Device::Cuda(_)),
            "CudaTrainTensor::zeros_like requires a CUDA tensor, got {:?}",
            reference.device()
        );
        let tensor = Tensor::zeros(reference.shape().clone(), reference.dtype(), reference.device())
            .context("alloc CUDA training tensor")?;
        Self::new(tensor)
    }

    pub fn detach(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            requires_grad: false,
            param_id: None,
            op_id: next_cuda_train_op_id(),
            grad_fn: None,
        }
    }

    pub fn as_tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn dtype(&self) -> DType {
        self.tensor.dtype()
    }

    pub fn dims(&self) -> &[usize] {
        self.tensor.dims()
    }

    pub fn op_id(&self) -> u64 {
        self.op_id
    }

    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn param_id(&self) -> Option<TensorId> {
        self.param_id
    }

    pub fn grad_fn(&self) -> Option<&Arc<dyn CudaBackwardOp>> {
        self.grad_fn.as_ref()
    }

    pub fn num_elements(&self) -> usize {
        self.tensor.dims().iter().product()
    }

    pub fn sgd_step_inplace(&self, grad: &CudaTrainTensor, lr: f32) -> Result<()> {
        ensure!(
            self.tensor.shape() == grad.tensor.shape(),
            "CUDA SGD shape mismatch: param={:?} grad={:?}",
            self.tensor.dims(),
            grad.tensor.dims()
        );
        ensure!(
            self.tensor.dtype() == grad.tensor.dtype(),
            "CUDA SGD dtype mismatch: param={:?} grad={:?}",
            self.tensor.dtype(),
            grad.tensor.dtype()
        );
        kiln_rmsnorm_kernel::sgd_step_inplace(&self.tensor, &grad.tensor, lr)
            .context("CUDA training tensor SGD step")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn adamw_step_inplace(
        &self,
        grad: &CudaTrainTensor,
        first_moment: &CudaTrainTensor,
        second_moment: &CudaTrainTensor,
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<()> {
        for (name, tensor) in [
            ("grad", grad.as_tensor()),
            ("first_moment", first_moment.as_tensor()),
            ("second_moment", second_moment.as_tensor()),
        ] {
            ensure!(
                self.tensor.shape() == tensor.shape(),
                "CUDA AdamW shape mismatch: param={:?} {name}={:?}",
                self.tensor.dims(),
                tensor.dims()
            );
            ensure!(
                self.tensor.dtype() == tensor.dtype(),
                "CUDA AdamW dtype mismatch: param={:?} {name}={:?}",
                self.tensor.dtype(),
                tensor.dtype()
            );
        }
        kiln_rmsnorm_kernel::adamw_step_inplace(
            &self.tensor,
            grad.as_tensor(),
            first_moment.as_tensor(),
            second_moment.as_tensor(),
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
        )
        .context("CUDA training tensor AdamW step")
    }

    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        Ok(self
            .tensor
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }
}

/// Per-parameter gradient store keyed by candle `TensorId`.
#[derive(Debug, Default)]
pub struct CudaGradStore {
    grads: HashMap<TensorId, CudaTrainTensor>,
}

impl CudaGradStore {
    pub fn new() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: TensorId, t: CudaTrainTensor) {
        self.grads.insert(id, t);
    }

    pub fn get(&self, id: TensorId) -> Option<&CudaTrainTensor> {
        self.grads.get(&id)
    }

    pub fn remove(&mut self, id: TensorId) -> Option<CudaTrainTensor> {
        self.grads.remove(&id)
    }

    pub fn into_inner(self) -> HashMap<TensorId, CudaTrainTensor> {
        self.grads
    }

    pub fn iter(&self) -> impl Iterator<Item = (&TensorId, &CudaTrainTensor)> {
        self.grads.iter()
    }

    pub fn len(&self) -> usize {
        self.grads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }
}

#[derive(Debug)]
pub struct CudaTrainArena {
    device: Device,
    allocations: Vec<CudaTrainTensor>,
    allocated_bytes: usize,
}

impl CudaTrainArena {
    pub fn new(device: &Device) -> Result<Self> {
        ensure!(
            matches!(device, Device::Cuda(_)),
            "CudaTrainArena requires a CUDA device, got {device:?}"
        );
        Ok(Self {
            device: device.clone(),
            allocations: Vec::new(),
            allocated_bytes: 0,
        })
    }

    pub fn zeros(&mut self, dims: &[usize], dtype: DType) -> Result<CudaTrainTensor> {
        let tensor = Tensor::zeros(dims.to_vec(), dtype, &self.device)
            .context("CudaTrainArena::zeros")?;
        let train_tensor = CudaTrainTensor::new(tensor)?;
        self.track(train_tensor)
    }

    pub fn track(&mut self, tensor: CudaTrainTensor) -> Result<CudaTrainTensor> {
        ensure!(
            matches!(tensor.as_tensor().device(), Device::Cuda(_)),
            "CudaTrainArena can only track CUDA tensors, got {:?}",
            tensor.as_tensor().device()
        );
        self.allocated_bytes = self
            .allocated_bytes
            .checked_add(approx_tensor_bytes(&tensor)?)
            .context("CudaTrainArena byte accounting overflow")?;
        self.allocations.push(tensor.clone());
        Ok(tensor)
    }

    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    pub fn clear(&mut self) {
        self.allocations.clear();
        self.allocated_bytes = 0;
    }
}

fn approx_tensor_bytes(tensor: &CudaTrainTensor) -> Result<usize> {
    let elems: usize = tensor.dims().iter().product();
    let elem_bytes = match tensor.dtype() {
        DType::U8 => 1,
        DType::U32 | DType::F32 => 4,
        DType::I64 | DType::F64 => 8,
        DType::BF16 | DType::F16 => 2,
        other => anyhow::bail!("CudaTrainArena does not support byte accounting for {other:?}"),
    };
    elems
        .checked_mul(elem_bytes)
        .context("CudaTrainArena tensor byte accounting overflow")
}

/// Walk the CUDA training graph rooted at `loss` and return per-parameter gradients.
pub fn cuda_backward(loss: &CudaTrainTensor) -> Result<CudaGradStore> {
    ensure!(
        loss.num_elements() == 1,
        "cuda_backward: loss must be scalar (got shape {:?})",
        loss.dims()
    );

    let mut order = Vec::new();
    let mut visited = HashSet::new();
    let mut leaves = Vec::new();
    collect_topo(loss, &mut visited, &mut order, &mut leaves);

    let mut grads: HashMap<u64, CudaTrainTensor> = HashMap::new();
    let seed = Tensor::ones(
        loss.as_tensor().shape().clone(),
        loss.dtype(),
        loss.as_tensor().device(),
    )
    .context("cuda_backward: seed ones_like")?;
    grads.insert(loss.op_id(), CudaTrainTensor::new(seed)?);

    for t in order.iter().rev() {
        let Some(grad_at_out) = grads.remove(&t.op_id()) else {
            continue;
        };
        let Some(gf) = t.grad_fn() else {
            continue;
        };
        let input_grads = gf
            .backward(&grad_at_out)
            .with_context(|| format!("cuda_backward: {} bwd", gf.op_name()))?;
        let inputs = gf.input_refs();
        ensure!(
            inputs.len() == input_grads.len(),
            "cuda_backward: {} returned {} grads for {} inputs",
            gf.op_name(),
            input_grads.len(),
            inputs.len()
        );
        for (input, maybe_grad) in inputs.iter().zip(input_grads.into_iter()) {
            let Some(g) = maybe_grad else { continue };
            if !input.requires_grad() && input.grad_fn().is_none() && input.param_id().is_none() {
                continue;
            }
            ensure!(
                g.as_tensor().shape() == input.as_tensor().shape(),
                "cuda_backward: {} produced grad of shape {:?} for input of shape {:?}",
                gf.op_name(),
                g.dims(),
                input.dims()
            );
            match grads.remove(&input.op_id()) {
                Some(existing) => {
                    let summed = (existing.as_tensor() + g.as_tensor())
                        .context("cuda_backward: grad accumulation")?;
                    grads.insert(input.op_id(), CudaTrainTensor::new(summed)?);
                }
                None => {
                    grads.insert(input.op_id(), g);
                }
            }
        }
    }

    let mut store = CudaGradStore::new();
    for leaf in leaves {
        if let Some(pid) = leaf.param_id() {
            if let Some(g) = grads.remove(&leaf.op_id()) {
                store.insert(pid, g);
            }
        }
    }
    Ok(store)
}

pub fn cuda_sgd_step_from_store(
    params: &[CudaTrainTensor],
    grads: &CudaGradStore,
    lr: f32,
) -> Result<usize> {
    let mut updated = 0usize;
    for param in params {
        let Some(param_id) = param.param_id() else {
            continue;
        };
        let Some(grad) = grads.get(param_id) else {
            continue;
        };
        param
            .sgd_step_inplace(grad, lr)
            .with_context(|| format!("cuda_sgd_step_from_store: param {:?}", param_id))?;
        updated += 1;
    }
    Ok(updated)
}

#[derive(Debug, Clone, Copy)]
pub struct CudaAdamWConfig {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
}

impl Default for CudaAdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaAdamWState {
    pub first_moment: CudaTrainTensor,
    pub second_moment: CudaTrainTensor,
    pub step: u32,
}

impl CudaAdamWState {
    pub fn zeros_like(param: &CudaTrainTensor) -> Result<Self> {
        Ok(Self {
            first_moment: CudaTrainTensor::zeros_like(param.as_tensor())?,
            second_moment: CudaTrainTensor::zeros_like(param.as_tensor())?,
            step: 0,
        })
    }
}

pub fn cuda_adamw_step_from_store(
    params: &[CudaTrainTensor],
    grads: &CudaGradStore,
    states: &mut HashMap<TensorId, CudaAdamWState>,
    cfg: CudaAdamWConfig,
) -> Result<usize> {
    let mut updated = 0usize;
    for param in params {
        let Some(param_id) = param.param_id() else {
            continue;
        };
        let Some(grad) = grads.get(param_id) else {
            continue;
        };
        let state = states
            .get_mut(&param_id)
            .with_context(|| format!("cuda_adamw_step_from_store: missing state for {:?}", param_id))?;
        state.step = state.step.saturating_add(1);
        param
            .adamw_step_inplace(
                grad,
                &state.first_moment,
                &state.second_moment,
                cfg.lr,
                cfg.beta1,
                cfg.beta2,
                cfg.eps,
                cfg.weight_decay,
                state.step,
            )
            .with_context(|| format!("cuda_adamw_step_from_store: param {:?}", param_id))?;
        updated += 1;
    }
    Ok(updated)
}

pub fn cuda_add(lhs: &CudaTrainTensor, rhs: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        lhs.as_tensor().shape() == rhs.as_tensor().shape(),
        "cuda_add: shape mismatch lhs={:?} rhs={:?}",
        lhs.dims(),
        rhs.dims()
    );
    ensure!(
        lhs.dtype() == rhs.dtype(),
        "cuda_add: dtype mismatch lhs={:?} rhs={:?}",
        lhs.dtype(),
        rhs.dtype()
    );
    let out = (lhs.as_tensor() + rhs.as_tensor()).context("cuda_add: candle CUDA add")?;
    let needs_grad = lhs.requires_grad()
        || lhs.grad_fn().is_some()
        || lhs.param_id().is_some()
        || rhs.requires_grad()
        || rhs.grad_fn().is_some()
        || rhs.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(AddBackward {
            inputs: vec![lhs.clone(), rhs.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_add_last_dim_bias(
    input: &CudaTrainTensor,
    bias: &CudaTrainTensor,
) -> Result<CudaTrainTensor> {
    ensure!(
        !input.dims().is_empty(),
        "cuda_add_last_dim_bias: expected non-empty input shape"
    );
    ensure!(
        bias.dims().len() == 1,
        "cuda_add_last_dim_bias: expected rank-1 bias, got {:?}",
        bias.dims()
    );
    let last_dim = *input.dims().last().expect("checked non-empty shape");
    ensure!(
        bias.dims()[0] == last_dim,
        "cuda_add_last_dim_bias: bias dim {} must match input last dim {}",
        bias.dims()[0],
        last_dim
    );
    ensure!(
        input.dtype() == bias.dtype(),
        "cuda_add_last_dim_bias: dtype mismatch input={:?} bias={:?}",
        input.dtype(),
        bias.dtype()
    );
    let out = input
        .as_tensor()
        .broadcast_add(bias.as_tensor())
        .context("cuda_add_last_dim_bias: candle CUDA broadcast_add")?;
    let needs_grad = input.requires_grad()
        || input.grad_fn().is_some()
        || input.param_id().is_some()
        || bias.requires_grad()
        || bias.grad_fn().is_some()
        || bias.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(AddLastDimBiasBackward {
            inputs: vec![input.clone(), bias.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_sub(lhs: &CudaTrainTensor, rhs: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        lhs.as_tensor().shape() == rhs.as_tensor().shape(),
        "cuda_sub: shape mismatch lhs={:?} rhs={:?}",
        lhs.dims(),
        rhs.dims()
    );
    ensure!(
        lhs.dtype() == rhs.dtype(),
        "cuda_sub: dtype mismatch lhs={:?} rhs={:?}",
        lhs.dtype(),
        rhs.dtype()
    );
    let out = (lhs.as_tensor() - rhs.as_tensor()).context("cuda_sub: candle CUDA sub")?;
    let needs_grad = lhs.requires_grad()
        || lhs.grad_fn().is_some()
        || lhs.param_id().is_some()
        || rhs.requires_grad()
        || rhs.grad_fn().is_some()
        || rhs.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(SubBackward {
            inputs: vec![lhs.clone(), rhs.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_mul(lhs: &CudaTrainTensor, rhs: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        lhs.as_tensor().shape() == rhs.as_tensor().shape(),
        "cuda_mul: shape mismatch lhs={:?} rhs={:?}",
        lhs.dims(),
        rhs.dims()
    );
    ensure!(
        lhs.dtype() == rhs.dtype(),
        "cuda_mul: dtype mismatch lhs={:?} rhs={:?}",
        lhs.dtype(),
        rhs.dtype()
    );
    let out = (lhs.as_tensor() * rhs.as_tensor()).context("cuda_mul: candle CUDA mul")?;
    let needs_grad = lhs.requires_grad()
        || lhs.grad_fn().is_some()
        || lhs.param_id().is_some()
        || rhs.requires_grad()
        || rhs.grad_fn().is_some()
        || rhs.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(MulBackward {
            inputs: vec![lhs.clone(), rhs.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_mul_last_dim_weight(
    input: &CudaTrainTensor,
    weight: &CudaTrainTensor,
) -> Result<CudaTrainTensor> {
    ensure!(
        !input.dims().is_empty(),
        "cuda_mul_last_dim_weight: expected non-empty input shape"
    );
    ensure!(
        weight.dims().len() == 1,
        "cuda_mul_last_dim_weight: expected rank-1 weight, got {:?}",
        weight.dims()
    );
    let last_dim = *input.dims().last().expect("checked non-empty shape");
    ensure!(
        weight.dims()[0] == last_dim,
        "cuda_mul_last_dim_weight: weight dim {} must match input last dim {}",
        weight.dims()[0],
        last_dim
    );
    ensure!(
        input.dtype() == weight.dtype(),
        "cuda_mul_last_dim_weight: dtype mismatch input={:?} weight={:?}",
        input.dtype(),
        weight.dtype()
    );
    let out = input
        .as_tensor()
        .broadcast_mul(weight.as_tensor())
        .context("cuda_mul_last_dim_weight: candle CUDA broadcast_mul")?;
    let needs_grad = input.requires_grad()
        || input.grad_fn().is_some()
        || input.param_id().is_some()
        || weight.requires_grad()
        || weight.grad_fn().is_some()
        || weight.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(MulLastDimWeightBackward {
            inputs: vec![input.clone(), weight.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_mul_last_dim_broadcast(
    input: &CudaTrainTensor,
    scalar: &CudaTrainTensor,
) -> Result<CudaTrainTensor> {
    ensure!(
        !input.dims().is_empty(),
        "cuda_mul_last_dim_broadcast: expected non-empty input shape"
    );
    ensure!(
        input.dims().len() == scalar.dims().len(),
        "cuda_mul_last_dim_broadcast: rank mismatch input={:?} scalar={:?}",
        input.dims(),
        scalar.dims()
    );
    let rank = input.dims().len();
    ensure!(
        input.dims()[..rank - 1] == scalar.dims()[..rank - 1] && scalar.dims()[rank - 1] == 1,
        "cuda_mul_last_dim_broadcast: scalar shape {:?} must match input {:?} with last dim 1",
        scalar.dims(),
        input.dims()
    );
    ensure!(
        input.dtype() == scalar.dtype(),
        "cuda_mul_last_dim_broadcast: dtype mismatch input={:?} scalar={:?}",
        input.dtype(),
        scalar.dtype()
    );
    let out = input
        .as_tensor()
        .broadcast_mul(scalar.as_tensor())
        .context("cuda_mul_last_dim_broadcast: candle CUDA broadcast_mul")?;
    let needs_grad = input.requires_grad()
        || input.grad_fn().is_some()
        || input.param_id().is_some()
        || scalar.requires_grad()
        || scalar.grad_fn().is_some()
        || scalar.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(MulLastDimBroadcastBackward {
            inputs: vec![input.clone(), scalar.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_div(lhs: &CudaTrainTensor, rhs: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        lhs.as_tensor().shape() == rhs.as_tensor().shape(),
        "cuda_div: shape mismatch lhs={:?} rhs={:?}",
        lhs.dims(),
        rhs.dims()
    );
    ensure!(
        lhs.dtype() == rhs.dtype(),
        "cuda_div: dtype mismatch lhs={:?} rhs={:?}",
        lhs.dtype(),
        rhs.dtype()
    );
    let out = lhs
        .as_tensor()
        .broadcast_div(rhs.as_tensor())
        .context("cuda_div: candle CUDA div")?;
    let needs_grad = lhs.requires_grad()
        || lhs.grad_fn().is_some()
        || lhs.param_id().is_some()
        || rhs.requires_grad()
        || rhs.grad_fn().is_some()
        || rhs.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(DivBackward {
            inputs: vec![lhs.clone(), rhs.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_scale(input: &CudaTrainTensor, scale: f32) -> Result<CudaTrainTensor> {
    let out = input
        .as_tensor()
        .affine(scale as f64, 0.0)
        .context("cuda_scale: candle CUDA scale")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(ScaleBackward {
            inputs: vec![input.clone()],
            scale,
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_causal_mask(input: &CudaTrainTensor, kv_offset: usize) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_causal_mask: expected F32 input, got {:?}",
        input.dtype()
    );
    ensure!(
        input.dims().len() == 3,
        "cuda_causal_mask: expected rank-3 [batch_heads, q_len, kv_len], got {:?}",
        input.dims()
    );
    let q_len = input.dims()[1];
    let kv_len = input.dims()[2];
    if q_len <= 1 {
        return Ok(input.clone());
    }

    let mask: Vec<f32> = (0..q_len)
        .flat_map(|q_idx| {
            let max_kv = kv_offset + q_idx + 1;
            (0..kv_len).map(move |kv_idx| if kv_idx < max_kv { 0.0 } else { -1.0e30 })
        })
        .collect();
    let mask = Tensor::new(mask, input.as_tensor().device())
        .context("cuda_causal_mask: build mask tensor")?
        .reshape((1usize, q_len, kv_len))
        .context("cuda_causal_mask: reshape mask")?;
    let out = input
        .as_tensor()
        .broadcast_add(&mask)
        .context("cuda_causal_mask: add mask")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(CausalMaskBackward {
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_rmsnorm(
    input: &CudaTrainTensor,
    weight: &CudaTrainTensor,
    eps: f32,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32 && weight.dtype() == DType::F32,
        "cuda_rmsnorm: expected F32 input and weight, got input={:?} weight={:?}",
        input.dtype(),
        weight.dtype()
    );
    ensure!(
        !input.dims().is_empty(),
        "cuda_rmsnorm: input must have at least one dimension"
    );
    let hidden = *input.dims().last().expect("checked non-empty dims");
    ensure!(
        weight.dims() == [hidden],
        "cuda_rmsnorm: weight shape {:?} does not match hidden {}",
        weight.dims(),
        hidden
    );

    let variance = input
        .as_tensor()
        .sqr()
        .context("cuda_rmsnorm: square input")?
        .mean_keepdim(D::Minus1)
        .context("cuda_rmsnorm: row variance")?;
    let rms_inv = (variance + eps as f64)
        .context("cuda_rmsnorm: add eps")?
        .sqrt()
        .context("cuda_rmsnorm: sqrt variance")?
        .recip()
        .context("cuda_rmsnorm: reciprocal rms")?;
    let normed = input
        .as_tensor()
        .broadcast_mul(&rms_inv)
        .context("cuda_rmsnorm: normalize input")?;
    let weight_plus_one = (weight.as_tensor().ones_like()? + weight.as_tensor())
        .context("cuda_rmsnorm: weight plus one")?;
    let out = normed
        .broadcast_mul(&weight_plus_one)
        .context("cuda_rmsnorm: apply weight")?;
    let needs_grad = input.requires_grad()
        || input.grad_fn().is_some()
        || input.param_id().is_some()
        || weight.requires_grad()
        || weight.grad_fn().is_some()
        || weight.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(RmsNormBackward {
            eps,
            inputs: vec![input.clone(), weight.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

fn cuda_rope_apply(
    input: &CudaTrainTensor,
    cos: &CudaTrainTensor,
    sin: &CudaTrainTensor,
    rotary_dim: usize,
    inverse: bool,
) -> Result<Tensor> {
    ensure!(
        input.dtype() == DType::F32 && cos.dtype() == DType::F32 && sin.dtype() == DType::F32,
        "cuda_rope: expected F32 input/cos/sin, got input={:?} cos={:?} sin={:?}",
        input.dtype(),
        cos.dtype(),
        sin.dtype()
    );
    ensure!(
        input.dims().len() == 3,
        "cuda_rope: input must be rank-3 [rows, heads, head_dim], got {:?}",
        input.dims()
    );
    let rows = input.dims()[0];
    let head_dim = input.dims()[2];
    ensure!(
        rotary_dim <= head_dim && rotary_dim % 2 == 0,
        "cuda_rope: rotary_dim={rotary_dim} must be <= head_dim={head_dim} and even"
    );
    let half = rotary_dim / 2;
    ensure!(
        cos.dims() == [rows, half],
        "cuda_rope: cos shape {:?} != [{rows}, {half}]",
        cos.dims()
    );
    ensure!(
        sin.dims() == [rows, half],
        "cuda_rope: sin shape {:?} != [{rows}, {half}]",
        sin.dims()
    );

    let x_rot = input
        .as_tensor()
        .narrow(D::Minus1, 0, rotary_dim)
        .context("cuda_rope: narrow rotary dims")?;
    let x_pass = if rotary_dim < head_dim {
        Some(
            input
                .as_tensor()
                .narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)
                .context("cuda_rope: narrow passthrough dims")?,
        )
    } else {
        None
    };
    let x1 = x_rot
        .narrow(D::Minus1, 0, half)
        .context("cuda_rope: narrow first half")?;
    let x2 = x_rot
        .narrow(D::Minus1, half, half)
        .context("cuda_rope: narrow second half")?;
    let cos = cos
        .as_tensor()
        .unsqueeze(1)
        .context("cuda_rope: cos unsqueeze heads")?;
    let sin = sin
        .as_tensor()
        .unsqueeze(1)
        .context("cuda_rope: sin unsqueeze heads")?;

    let x1_cos = x1
        .broadcast_mul(&cos)
        .context("cuda_rope: x1 * cos")?;
    let x2_sin = x2
        .broadcast_mul(&sin)
        .context("cuda_rope: x2 * sin")?;
    let x1_sin = x1
        .broadcast_mul(&sin)
        .context("cuda_rope: x1 * sin")?;
    let x2_cos = x2
        .broadcast_mul(&cos)
        .context("cuda_rope: x2 * cos")?;
    let (r1, r2) = if inverse {
        (
            (x1_cos + x2_sin).context("cuda_rope: inverse first half")?,
            (x2_cos - x1_sin).context("cuda_rope: inverse second half")?,
        )
    } else {
        (
            (x1_cos - x2_sin).context("cuda_rope: forward first half")?,
            (x1_sin + x2_cos).context("cuda_rope: forward second half")?,
        )
    };

    match x_pass {
        Some(pass) => Tensor::cat(&[&r1, &r2, &pass], D::Minus1).context("cuda_rope: cat output"),
        None => Tensor::cat(&[&r1, &r2], D::Minus1).context("cuda_rope: cat output"),
    }
}

pub fn cuda_rope(
    input: &CudaTrainTensor,
    cos: &CudaTrainTensor,
    sin: &CudaTrainTensor,
    rotary_dim: usize,
) -> Result<CudaTrainTensor> {
    let out = cuda_rope_apply(input, cos, sin, rotary_dim, false)?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(RopeBackward {
            cos: cos.clone(),
            sin: sin.clone(),
            rotary_dim,
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_to_dtype(input: &CudaTrainTensor, dtype: DType) -> Result<CudaTrainTensor> {
    let out = input
        .as_tensor()
        .to_dtype(dtype)
        .with_context(|| format!("cuda_to_dtype: convert to {dtype:?}"))?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(CastBackward {
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_sum_all(input: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    let out = input
        .as_tensor()
        .sum_all()
        .context("cuda_sum_all: candle CUDA sum_all")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(SumAllBackward {
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_mean_all(input: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    let out = input
        .as_tensor()
        .mean_all()
        .context("cuda_mean_all: candle CUDA mean_all")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(MeanAllBackward {
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_reshape(input: &CudaTrainTensor, dims: &[usize]) -> Result<CudaTrainTensor> {
    let in_elems = input.num_elements();
    let out_elems: usize = dims.iter().product();
    ensure!(
        in_elems == out_elems,
        "cuda_reshape: element count mismatch input={:?} output={:?}",
        input.dims(),
        dims
    );
    let out = input
        .as_tensor()
        .reshape(dims)
        .context("cuda_reshape: candle CUDA reshape")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(ReshapeBackward {
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_transpose2d(input: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        input.dims().len() == 2,
        "cuda_transpose2d: expected 2D input, got {:?}",
        input.dims()
    );
    let out = input
        .as_tensor()
        .t()
        .context("cuda_transpose2d: candle CUDA transpose")?
        .contiguous()
        .context("cuda_transpose2d: contiguous output")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(Transpose2dBackward {
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_transpose_last_two(input: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        input.dims().len() >= 2,
        "cuda_transpose_last_two: expected rank >= 2 input, got {:?}",
        input.dims()
    );
    let rank = input.dims().len();
    let out = input
        .as_tensor()
        .transpose(rank - 2, rank - 1)
        .context("cuda_transpose_last_two: transpose last two dims")?
        .contiguous()
        .context("cuda_transpose_last_two: contiguous output")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(TransposeLastTwoBackward {
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_matmul(lhs: &CudaTrainTensor, rhs: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        lhs.dims().len() == 2 && rhs.dims().len() == 2,
        "cuda_matmul: expected 2D inputs, got lhs={:?} rhs={:?}",
        lhs.dims(),
        rhs.dims()
    );
    ensure!(
        lhs.dims()[1] == rhs.dims()[0],
        "cuda_matmul: inner-dim mismatch lhs={:?} rhs={:?}",
        lhs.dims(),
        rhs.dims()
    );
    ensure!(
        lhs.dtype() == rhs.dtype(),
        "cuda_matmul: dtype mismatch lhs={:?} rhs={:?}",
        lhs.dtype(),
        rhs.dtype()
    );
    let out = lhs
        .as_tensor()
        .matmul(rhs.as_tensor())
        .context("cuda_matmul: candle CUDA matmul")?;
    let needs_grad = lhs.requires_grad()
        || lhs.grad_fn().is_some()
        || lhs.param_id().is_some()
        || rhs.requires_grad()
        || rhs.grad_fn().is_some()
        || rhs.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(MatmulBackward {
            inputs: vec![lhs.clone(), rhs.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_swiglu_mlp(
    input: &CudaTrainTensor,
    gate_weight: &CudaTrainTensor,
    up_weight: &CudaTrainTensor,
    down_weight: &CudaTrainTensor,
) -> Result<CudaTrainTensor> {
    let gate = cuda_matmul(input, gate_weight)?;
    let up = cuda_matmul(input, up_weight)?;
    let activated = cuda_silu(&gate)?;
    let hidden = cuda_mul(&activated, &up)?;
    cuda_matmul(&hidden, down_weight)
}

pub struct CudaOwnedFullAttentionLayer {
    pub input_norm_weight: CudaTrainTensor,
    pub q_weight: CudaTrainTensor,
    pub k_weight: CudaTrainTensor,
    pub v_weight: CudaTrainTensor,
    pub q_norm_weight: CudaTrainTensor,
    pub k_norm_weight: CudaTrainTensor,
    pub o_weight: CudaTrainTensor,
    pub post_norm_weight: CudaTrainTensor,
    pub gate_weight: CudaTrainTensor,
    pub up_weight: CudaTrainTensor,
    pub down_weight: CudaTrainTensor,
    pub heads_q: usize,
    pub heads_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub attn_output_gate: bool,
}

impl CudaOwnedFullAttentionLayer {
    pub fn as_borrowed<'a>(
        &'a self,
        rope: Option<CudaRopeTables<'a>>,
    ) -> CudaFullAttentionLayer<'a> {
        CudaFullAttentionLayer {
            input_norm_weight: &self.input_norm_weight,
            q_weight: &self.q_weight,
            k_weight: &self.k_weight,
            v_weight: &self.v_weight,
            q_norm_weight: Some(&self.q_norm_weight),
            k_norm_weight: Some(&self.k_norm_weight),
            o_weight: &self.o_weight,
            post_norm_weight: &self.post_norm_weight,
            gate_weight: &self.gate_weight,
            up_weight: &self.up_weight,
            down_weight: &self.down_weight,
            heads_q: self.heads_q,
            heads_kv: self.heads_kv,
            head_dim: self.head_dim,
            eps: self.eps,
            attn_output_gate: self.attn_output_gate,
            rope,
        }
    }
}

pub struct CudaOwnedLinearAttentionLayer {
    pub layer_norm_weight: CudaTrainTensor,
    pub in_proj_qkv_weight: CudaTrainTensor,
    pub in_proj_z_weight: CudaTrainTensor,
    pub in_proj_a_weight: CudaTrainTensor,
    pub in_proj_b_weight: CudaTrainTensor,
    pub conv1d_weight: CudaTrainTensor,
    pub a_log: CudaTrainTensor,
    pub a_log_gates: CudaTrainTensor,
    pub dt_bias: CudaTrainTensor,
    pub gated_norm_weight: CudaTrainTensor,
    pub out_proj_weight: CudaTrainTensor,
    pub heads_k: usize,
    pub heads_v: usize,
    pub head_dim_k: usize,
    pub head_dim_v: usize,
    pub conv_kernel: usize,
    pub eps: f32,
}

pub enum CudaLayerWeights {
    FullAttention(CudaOwnedFullAttentionLayer),
    LinearAttention(CudaOwnedLinearAttentionLayer),
}

pub struct CudaModelWeights {
    pub token_embedding: CudaTrainTensor,
    pub final_norm_weight: CudaTrainTensor,
    pub lm_head_weight: CudaTrainTensor,
    pub layers: Vec<CudaLayerWeights>,
    pub rotary_inv_freq: Vec<f32>,
    pub rotary_dim: usize,
    pub vocab: usize,
    pub hidden: usize,
}

/// One GDN layer's recurrent and conv state for CUDA-native training.
pub struct CudaGdnLayerState {
    /// Recurrent state S, shape [batch, num_value_heads, head_dim_k, head_dim_v].
    pub recurrent_state: CudaTrainTensor,
    pub recurrent_n_elements: usize,
    /// Conv1d sliding window, shape [batch, conv_channels, kernel_size - 1].
    pub conv_state: CudaTrainTensor,
    pub conv_n_elements: usize,
}

/// Whole-model CUDA GDN state, indexed by LinearAttention-layer order.
pub struct CudaLinearAttentionState {
    pub layers: Vec<CudaGdnLayerState>,
}

impl CudaLinearAttentionState {
    /// Create fresh zero-initialized state for `num_gdn_layers` GDN layers.
    pub fn zeros(
        device: &Device,
        num_gdn_layers: usize,
        batch: usize,
        heads_v: usize,
        head_dim_k: usize,
        head_dim_v: usize,
        conv_channels: usize,
        kernel_size: usize,
    ) -> Result<Self> {
        ensure!(
            matches!(device, Device::Cuda(_)),
            "CudaLinearAttentionState requires a CUDA device, got {:?}",
            device
        );
        let recurrent_n = batch * heads_v * head_dim_k * head_dim_v;
        let state_len = kernel_size.saturating_sub(1).max(1);
        let conv_n = batch * conv_channels * state_len;
        let mut layers = Vec::with_capacity(num_gdn_layers);
        for _ in 0..num_gdn_layers {
            layers.push(CudaGdnLayerState {
                recurrent_state: CudaTrainTensor::new(Tensor::zeros(
                    (batch, heads_v, head_dim_k, head_dim_v),
                    DType::F32,
                    device,
                )?)?,
                recurrent_n_elements: recurrent_n,
                conv_state: CudaTrainTensor::new(Tensor::zeros(
                    (batch, conv_channels, state_len),
                    DType::F32,
                    device,
                )?)?,
                conv_n_elements: conv_n,
            });
        }
        Ok(Self { layers })
    }
}

/// Count how many imported CUDA layers are GDN (LinearAttention).
pub fn cuda_count_gdn_layers(weights: &CudaModelWeights) -> usize {
    weights
        .layers
        .iter()
        .filter(|layer| matches!(layer, CudaLayerWeights::LinearAttention(_)))
        .count()
}

fn cuda_frozen_f32_tensor(tensor: &Tensor, name: &str) -> Result<CudaTrainTensor> {
    ensure!(
        matches!(tensor.device(), Device::Cuda(_)),
        "CUDA native model import requires CUDA tensor for {name}, got {:?}",
        tensor.device()
    );
    let tensor = tensor
        .to_dtype(DType::F32)
        .with_context(|| format!("convert CUDA native model weight {name} to f32"))?
        .contiguous()
        .with_context(|| format!("make CUDA native model weight {name} contiguous"))?;
    CudaTrainTensor::new(tensor).with_context(|| format!("wrap CUDA native model weight {name}"))
}

fn cuda_full_attention_from_gpu(
    weights: &GpuFullAttentionWeights,
    layer: &GpuLayerWeights,
    model_config: &ModelConfig,
    layer_idx: usize,
) -> Result<CudaOwnedFullAttentionLayer> {
    let ctx = |name: &str| format!("layer {layer_idx} FullAttention {name}");
    Ok(CudaOwnedFullAttentionLayer {
        input_norm_weight: cuda_frozen_f32_tensor(&layer.input_layernorm, &ctx("input_norm"))?,
        q_weight: cuda_frozen_f32_tensor(&weights.q_proj_t, &ctx("q_proj_t"))?,
        k_weight: cuda_frozen_f32_tensor(&weights.k_proj_t, &ctx("k_proj_t"))?,
        v_weight: cuda_frozen_f32_tensor(&weights.v_proj_t, &ctx("v_proj_t"))?,
        q_norm_weight: cuda_frozen_f32_tensor(&weights.q_norm, &ctx("q_norm"))?,
        k_norm_weight: cuda_frozen_f32_tensor(&weights.k_norm, &ctx("k_norm"))?,
        o_weight: cuda_frozen_f32_tensor(&weights.o_proj_t, &ctx("o_proj_t"))?,
        post_norm_weight: cuda_frozen_f32_tensor(
            &layer.post_attention_layernorm,
            &ctx("post_attention_norm"),
        )?,
        gate_weight: cuda_frozen_f32_tensor(&layer.mlp.gate_proj_t, &ctx("gate_proj_t"))?,
        up_weight: cuda_frozen_f32_tensor(&layer.mlp.up_proj_t, &ctx("up_proj_t"))?,
        down_weight: cuda_frozen_f32_tensor(&layer.mlp.down_proj_t, &ctx("down_proj_t"))?,
        heads_q: model_config.num_attention_heads,
        heads_kv: model_config.num_kv_heads,
        head_dim: model_config.head_dim,
        eps: model_config.rms_norm_eps as f32,
        attn_output_gate: model_config.attn_output_gate,
    })
}

fn cuda_linear_attention_from_gpu(
    weights: &GpuLinearAttentionWeights,
    layer: &GpuLayerWeights,
    model_config: &ModelConfig,
    layer_idx: usize,
) -> Result<CudaOwnedLinearAttentionLayer> {
    let ctx = |name: &str| format!("layer {layer_idx} LinearAttention {name}");
    let conv_kernel = *weights
        .conv1d
        .dims()
        .last()
        .context("CUDA native model import: conv1d must have rank > 0")?;
    Ok(CudaOwnedLinearAttentionLayer {
        layer_norm_weight: cuda_frozen_f32_tensor(&layer.input_layernorm, &ctx("input_norm"))?,
        in_proj_qkv_weight: cuda_frozen_f32_tensor(&weights.in_proj_qkv_t, &ctx("in_proj_qkv_t"))?,
        in_proj_z_weight: cuda_frozen_f32_tensor(&weights.in_proj_z_t, &ctx("in_proj_z_t"))?,
        in_proj_a_weight: cuda_frozen_f32_tensor(&weights.in_proj_a_t, &ctx("in_proj_a_t"))?,
        in_proj_b_weight: cuda_frozen_f32_tensor(&weights.in_proj_b_t, &ctx("in_proj_b_t"))?,
        conv1d_weight: cuda_frozen_f32_tensor(&weights.conv1d, &ctx("conv1d"))?,
        a_log: cuda_frozen_f32_tensor(&weights.a_log, &ctx("a_log"))?,
        a_log_gates: cuda_frozen_f32_tensor(&weights.a_log_gates, &ctx("a_log_gates"))?,
        dt_bias: cuda_frozen_f32_tensor(&weights.dt_bias, &ctx("dt_bias"))?,
        gated_norm_weight: cuda_frozen_f32_tensor(&weights.norm, &ctx("gated_norm"))?,
        out_proj_weight: cuda_frozen_f32_tensor(&weights.out_proj_t, &ctx("out_proj_t"))?,
        heads_k: model_config.linear_num_key_heads,
        heads_v: model_config.linear_num_value_heads,
        head_dim_k: model_config.linear_key_head_dim,
        head_dim_v: model_config.linear_value_head_dim,
        conv_kernel,
        eps: model_config.rms_norm_eps as f32,
    })
}

impl CudaModelWeights {
    pub fn from_gpu_weights(weights: &GpuWeights, model_config: &ModelConfig) -> Result<Self> {
        let token_embedding = cuda_frozen_f32_tensor(&weights.embed_tokens, "embed_tokens")?;
        let lm_head_weight = cuda_frozen_f32_tensor(&weights.embed_tokens_t, "embed_tokens_t")?;
        let final_norm_weight = cuda_frozen_f32_tensor(&weights.final_norm, "final_norm")?;
        let mut layers = Vec::with_capacity(weights.layers.len());
        for (idx, layer) in weights.layers.iter().enumerate() {
            let imported = match &layer.attention {
                GpuAttentionWeights::Full(full) => CudaLayerWeights::FullAttention(
                    cuda_full_attention_from_gpu(full, layer, model_config, idx)?,
                ),
                GpuAttentionWeights::Linear(linear) => CudaLayerWeights::LinearAttention(
                    cuda_linear_attention_from_gpu(linear, layer, model_config, idx)?,
                ),
            };
            layers.push(imported);
        }
        let rotary_inv_freq = weights
            .rotary_inv_freq
            .to_dtype(DType::F32)
            .context("convert CUDA native model rotary_inv_freq to f32")?
            .to_device(&Device::Cpu)
            .context("read CUDA native model rotary_inv_freq to CPU")?
            .flatten_all()
            .context("flatten CUDA native model rotary_inv_freq")?
            .to_vec1::<f32>()
            .context("read CUDA native model rotary_inv_freq values")?;
        Ok(Self {
            token_embedding,
            final_norm_weight,
            lm_head_weight,
            layers,
            rotary_dim: rotary_inv_freq.len() * 2,
            rotary_inv_freq,
            vocab: model_config.vocab_size,
            hidden: model_config.hidden_size,
        })
    }
}

pub struct CudaRopeTables<'a> {
    pub cos: &'a CudaTrainTensor,
    pub sin: &'a CudaTrainTensor,
    pub rotary_dim: usize,
}

pub struct CudaFullAttentionLayer<'a> {
    pub input_norm_weight: &'a CudaTrainTensor,
    pub q_weight: &'a CudaTrainTensor,
    pub k_weight: &'a CudaTrainTensor,
    pub v_weight: &'a CudaTrainTensor,
    pub q_norm_weight: Option<&'a CudaTrainTensor>,
    pub k_norm_weight: Option<&'a CudaTrainTensor>,
    pub o_weight: &'a CudaTrainTensor,
    pub post_norm_weight: &'a CudaTrainTensor,
    pub gate_weight: &'a CudaTrainTensor,
    pub up_weight: &'a CudaTrainTensor,
    pub down_weight: &'a CudaTrainTensor,
    pub heads_q: usize,
    pub heads_kv: usize,
    pub head_dim: usize,
    pub eps: f32,
    pub attn_output_gate: bool,
    pub rope: Option<CudaRopeTables<'a>>,
}

fn cuda_per_head_rmsnorm_flat(
    input: &CudaTrainTensor,
    weight: &CudaTrainTensor,
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dims().len() == 2 && input.dims()[1] == heads * head_dim,
        "cuda_per_head_rmsnorm_flat: expected [rows, heads*head_dim], got {:?} heads={heads} head_dim={head_dim}",
        input.dims()
    );
    let rows = input.dims()[0];
    let flat = cuda_reshape(input, &[rows * heads, head_dim])?;
    let normed = cuda_rmsnorm(&flat, weight, eps)?;
    cuda_reshape(&normed, &[rows, heads * head_dim])
}

fn cuda_apply_rope_to_flat(
    input: &CudaTrainTensor,
    cos: &CudaTrainTensor,
    sin: &CudaTrainTensor,
    heads: usize,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dims().len() == 2 && input.dims()[1] == heads * head_dim,
        "cuda_apply_rope_to_flat: expected [rows, heads*head_dim], got {:?} heads={heads} head_dim={head_dim}",
        input.dims()
    );
    let rows = input.dims()[0];
    let rank3 = cuda_reshape(input, &[rows, heads, head_dim])?;
    let rotated = cuda_rope(&rank3, cos, sin, rotary_dim)?;
    cuda_reshape(&rotated, &[rows, heads * head_dim])
}

pub fn cuda_full_attention_layer(
    input: &CudaTrainTensor,
    weights: &CudaFullAttentionLayer<'_>,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dims().len() == 2,
        "cuda_full_attention_layer: expected rank-2 [rows, hidden] input, got {:?}",
        input.dims()
    );
    let rows = input.dims()[0];
    let q_dim = weights.heads_q * weights.head_dim;
    let q_out_dim = if weights.attn_output_gate {
        q_dim * 2
    } else {
        q_dim
    };
    ensure!(
        weights.q_weight.dims().len() == 2 && weights.q_weight.dims()[1] == q_out_dim,
        "cuda_full_attention_layer: q_weight must project to {q_out_dim}, got {:?}",
        weights.q_weight.dims()
    );

    let h_norm = cuda_rmsnorm(input, weights.input_norm_weight, weights.eps)?;
    let q_raw = cuda_matmul(&h_norm, weights.q_weight)?;
    let k = cuda_matmul(&h_norm, weights.k_weight)?;
    let v = cuda_matmul(&h_norm, weights.v_weight)?;

    let (q, gate) = if weights.attn_output_gate {
        let q_raw_3d = cuda_reshape(&q_raw, &[rows, weights.heads_q, weights.head_dim * 2])?;
        let q_3d = cuda_narrow_last_dim(&q_raw_3d, 0, weights.head_dim)?;
        let gate_3d = cuda_narrow_last_dim(&q_raw_3d, weights.head_dim, weights.head_dim)?;
        (
            cuda_reshape(&q_3d, &[rows, q_dim])?,
            Some(cuda_reshape(&gate_3d, &[rows, q_dim])?),
        )
    } else {
        (q_raw, None)
    };

    let q = match weights.q_norm_weight {
        Some(weight) => {
            cuda_per_head_rmsnorm_flat(&q, weight, weights.heads_q, weights.head_dim, weights.eps)?
        }
        None => q,
    };
    let k = match weights.k_norm_weight {
        Some(weight) => cuda_per_head_rmsnorm_flat(
            &k,
            weight,
            weights.heads_kv,
            weights.head_dim,
            weights.eps,
        )?,
        None => k,
    };

    let (q, k) = match &weights.rope {
        Some(rope) => (
            cuda_apply_rope_to_flat(
                &q,
                rope.cos,
                rope.sin,
                weights.heads_q,
                weights.head_dim,
                rope.rotary_dim,
            )?,
            cuda_apply_rope_to_flat(
                &k,
                rope.cos,
                rope.sin,
                weights.heads_kv,
                weights.head_dim,
                rope.rotary_dim,
            )?,
        ),
        None => (q, k),
    };

    let q_3d = cuda_reshape(&q, &[rows, weights.heads_q, weights.head_dim])?;
    let k_3d = cuda_reshape(&k, &[rows, weights.heads_kv, weights.head_dim])?;
    let v_3d = cuda_reshape(&v, &[rows, weights.heads_kv, weights.head_dim])?;
    let scale = 1.0 / (weights.head_dim as f32).sqrt();
    let attn = cuda_sdpa_prefill_causal(&q_3d, &k_3d, &v_3d, scale)?;
    let attn_flat = cuda_reshape(&attn, &[rows, q_dim])?;
    let attn_gated = match gate {
        Some(gate) => {
            let sig = cuda_sigmoid(&gate)?;
            cuda_mul(&attn_flat, &sig)?
        }
        None => attn_flat,
    };
    let o_out = cuda_matmul(&attn_gated, weights.o_weight)?;
    let after_attn = cuda_add(input, &o_out)?;
    let h_norm2 = cuda_rmsnorm(&after_attn, weights.post_norm_weight, weights.eps)?;
    let mlp = cuda_swiglu_mlp(
        &h_norm2,
        weights.gate_weight,
        weights.up_weight,
        weights.down_weight,
    )?;
    cuda_add(&after_attn, &mlp)
}

pub fn cuda_batched_matmul(lhs: &CudaTrainTensor, rhs: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        lhs.dims().len() >= 3 && rhs.dims().len() >= 3,
        "cuda_batched_matmul: expected rank >= 3 inputs, got lhs={:?} rhs={:?}",
        lhs.dims(),
        rhs.dims()
    );
    ensure!(
        lhs.dims().len() == rhs.dims().len(),
        "cuda_batched_matmul: rank mismatch lhs={:?} rhs={:?}",
        lhs.dims(),
        rhs.dims()
    );
    let rank = lhs.dims().len();
    ensure!(
        lhs.dims()[..rank - 2] == rhs.dims()[..rank - 2],
        "cuda_batched_matmul: batch dim mismatch lhs={:?} rhs={:?}",
        lhs.dims(),
        rhs.dims()
    );
    ensure!(
        lhs.dims()[rank - 1] == rhs.dims()[rank - 2],
        "cuda_batched_matmul: inner-dim mismatch lhs={:?} rhs={:?}",
        lhs.dims(),
        rhs.dims()
    );
    ensure!(
        lhs.dtype() == rhs.dtype(),
        "cuda_batched_matmul: dtype mismatch lhs={:?} rhs={:?}",
        lhs.dtype(),
        rhs.dtype()
    );
    let out = lhs
        .as_tensor()
        .broadcast_matmul(rhs.as_tensor())
        .context("cuda_batched_matmul: candle CUDA broadcast_matmul")?;
    let needs_grad = lhs.requires_grad()
        || lhs.grad_fn().is_some()
        || lhs.param_id().is_some()
        || rhs.requires_grad()
        || rhs.grad_fn().is_some()
        || rhs.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(BatchedMatmulBackward {
            inputs: vec![lhs.clone(), rhs.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_softmax_last_dim(input: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_softmax_last_dim: expected F32 input, got {:?}",
        input.dtype()
    );
    ensure!(
        !input.dims().is_empty(),
        "cuda_softmax_last_dim: expected non-empty shape"
    );
    let max_val = input
        .as_tensor()
        .max_keepdim(D::Minus1)
        .context("cuda_softmax_last_dim: max_keepdim")?;
    let shifted = input
        .as_tensor()
        .broadcast_sub(&max_val)
        .context("cuda_softmax_last_dim: shift")?;
    let exp_shifted = shifted.exp().context("cuda_softmax_last_dim: exp")?;
    let sum_exp = exp_shifted
        .sum_keepdim(D::Minus1)
        .context("cuda_softmax_last_dim: sum_keepdim")?;
    let out = exp_shifted
        .broadcast_div(&sum_exp)
        .context("cuda_softmax_last_dim: normalize")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let output_for_backward = CudaTrainTensor::new(out.clone())?;
    let grad_fn = needs_grad.then(|| {
        Arc::new(SoftmaxLastDimBackward {
            output: output_for_backward,
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_sigmoid(input: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_sigmoid: expected F32 input, got {:?}",
        input.dtype()
    );
    let neg = input.as_tensor().neg().context("cuda_sigmoid: neg")?;
    let exp_neg = neg.exp().context("cuda_sigmoid: exp")?;
    let one_plus = (exp_neg + 1.0).context("cuda_sigmoid: add one")?;
    let out = one_plus.recip().context("cuda_sigmoid: reciprocal")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let output_for_backward = CudaTrainTensor::new(out.clone())?;
    let grad_fn = needs_grad.then(|| {
        Arc::new(SigmoidBackward {
            output: output_for_backward,
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_exp(input: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_exp: expected F32 input, got {:?}",
        input.dtype()
    );
    let out = input.as_tensor().exp().context("cuda_exp: candle CUDA exp")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(ExpBackward {
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_softplus(input: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_softplus: expected F32 input, got {:?}",
        input.dtype()
    );
    let out = (input
        .as_tensor()
        .exp()
        .context("cuda_softplus: exp input")?
        + 1.0)
        .context("cuda_softplus: add one")?
        .log()
        .context("cuda_softplus: log")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(SoftplusBackward {
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_silu(input: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_silu: expected F32 input, got {:?}",
        input.dtype()
    );
    let neg = input.as_tensor().neg().context("cuda_silu: neg")?;
    let exp_neg = neg.exp().context("cuda_silu: exp")?;
    let one_plus = (exp_neg + 1.0).context("cuda_silu: add one")?;
    let sigmoid = one_plus.recip().context("cuda_silu: reciprocal")?;
    let out = (input.as_tensor() * &sigmoid).context("cuda_silu: x * sigmoid")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let sigmoid_for_backward = CudaTrainTensor::new(sigmoid)?;
    let grad_fn = needs_grad.then(|| {
        Arc::new(SiluBackward {
            sigmoid: sigmoid_for_backward,
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_repeat_kv_heads(input: &CudaTrainTensor, groups: usize) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_repeat_kv_heads: expected F32 input, got {:?}",
        input.dtype()
    );
    ensure!(
        input.dims().len() == 3,
        "cuda_repeat_kv_heads: expected rank-3 [heads_kv, rows, head_dim], got {:?}",
        input.dims()
    );
    ensure!(groups >= 1, "cuda_repeat_kv_heads: groups must be >= 1");
    if groups == 1 {
        return Ok(input.clone());
    }

    let heads_kv = input.dims()[0];
    let mut repeated = Vec::with_capacity(heads_kv * groups);
    for head in 0..heads_kv {
        let slice = input
            .as_tensor()
            .narrow(0, head, 1)
            .with_context(|| format!("cuda_repeat_kv_heads: narrow head {head}"))?
            .contiguous()
            .with_context(|| format!("cuda_repeat_kv_heads: contiguous head {head}"))?;
        for _ in 0..groups {
            repeated.push(slice.clone());
        }
    }
    let refs: Vec<&Tensor> = repeated.iter().collect();
    let out = Tensor::cat(&refs, 0).context("cuda_repeat_kv_heads: cat repeated heads")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(RepeatKvHeadsBackward {
            heads_kv,
            groups,
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_narrow_last_dim(
    input: &CudaTrainTensor,
    start: usize,
    len: usize,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_narrow_last_dim: expected F32 input, got {:?}",
        input.dtype()
    );
    ensure!(
        !input.dims().is_empty(),
        "cuda_narrow_last_dim: expected non-empty shape"
    );
    let last_dim = *input.dims().last().expect("checked non-empty shape");
    ensure!(
        start <= last_dim && start + len <= last_dim,
        "cuda_narrow_last_dim: slice [{start}, {}) out of bounds for last_dim={last_dim}",
        start + len
    );
    let out = input
        .as_tensor()
        .narrow(D::Minus1, start, len)
        .context("cuda_narrow_last_dim: candle CUDA narrow")?
        .contiguous()
        .context("cuda_narrow_last_dim: contiguous output")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(NarrowLastDimBackward {
            start,
            len,
            input_dims: input.dims().to_vec(),
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_narrow_rows(
    input: &CudaTrainTensor,
    start: usize,
    len: usize,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_narrow_rows: expected F32 input, got {:?}",
        input.dtype()
    );
    ensure!(
        input.dims().len() == 2,
        "cuda_narrow_rows: expected rank-2 [rows, dim], got {:?}",
        input.dims()
    );
    let rows = input.dims()[0];
    ensure!(
        start <= rows && start + len <= rows,
        "cuda_narrow_rows: slice [{start}, {}) out of bounds for rows={rows}",
        start + len
    );
    let out = input
        .as_tensor()
        .narrow(0, start, len)
        .context("cuda_narrow_rows: candle CUDA narrow")?
        .contiguous()
        .context("cuda_narrow_rows: contiguous output")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(NarrowRowsBackward {
            start,
            len,
            input_dims: input.dims().to_vec(),
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_cat_rows(parts: &[CudaTrainTensor]) -> Result<CudaTrainTensor> {
    ensure!(
        !parts.is_empty(),
        "cuda_cat_rows: expected at least one tensor"
    );
    let dim = parts[0].dims().get(1).copied().context(
        "cuda_cat_rows: expected rank-2 [rows, dim] tensors, got empty first shape",
    )?;
    let dtype = parts[0].dtype();
    for (idx, part) in parts.iter().enumerate() {
        ensure!(
            part.dims().len() == 2,
            "cuda_cat_rows: part {idx} expected rank-2 [rows, dim], got {:?}",
            part.dims()
        );
        ensure!(
            part.dims()[1] == dim,
            "cuda_cat_rows: part {idx} dim {} does not match first dim {dim}",
            part.dims()[1]
        );
        ensure!(
            part.dtype() == dtype,
            "cuda_cat_rows: part {idx} dtype {:?} does not match first dtype {:?}",
            part.dtype(),
            dtype
        );
    }
    let refs: Vec<&Tensor> = parts.iter().map(|part| part.as_tensor()).collect();
    let out = Tensor::cat(&refs, 0).context("cuda_cat_rows: cat rows")?;
    let needs_grad = parts
        .iter()
        .any(|part| part.requires_grad() || part.grad_fn().is_some() || part.param_id().is_some());
    let row_counts: Vec<usize> = parts.iter().map(|part| part.dims()[0]).collect();
    let grad_fn = needs_grad.then(|| {
        Arc::new(CatRowsBackward {
            row_counts,
            inputs: parts.to_vec(),
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_causal_depthwise_conv1d_prefill_zero_state(
    input: &CudaTrainTensor,
    weight: &CudaTrainTensor,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32 && weight.dtype() == DType::F32,
        "cuda_causal_depthwise_conv1d_prefill_zero_state: expected F32 input/weight, got {:?}/{:?}",
        input.dtype(),
        weight.dtype()
    );
    ensure!(
        input.dims().len() == 2,
        "cuda_causal_depthwise_conv1d_prefill_zero_state: expected input [rows, channels], got {:?}",
        input.dims()
    );
    ensure!(
        weight.dims().len() == 2 || weight.dims().len() == 3,
        "cuda_causal_depthwise_conv1d_prefill_zero_state: expected weight [channels,k] or [channels,1,k], got {:?}",
        weight.dims()
    );
    let rows = input.dims()[0];
    let channels = input.dims()[1];
    let kernel = *weight
        .dims()
        .last()
        .context("cuda_causal_depthwise_conv1d_prefill_zero_state: empty weight shape")?;
    ensure!(
        kernel > 0,
        "cuda_causal_depthwise_conv1d_prefill_zero_state: kernel must be non-zero"
    );
    ensure!(
        weight.dims()[0] == channels,
        "cuda_causal_depthwise_conv1d_prefill_zero_state: weight channels {} != input channels {}",
        weight.dims()[0],
        channels
    );
    if weight.dims().len() == 3 {
        ensure!(
            weight.dims()[1] == 1,
            "cuda_causal_depthwise_conv1d_prefill_zero_state: rank-3 weight middle dim must be 1, got {}",
            weight.dims()[1]
        );
    }

    let weight_2d = cuda_reshape(weight, &[channels, kernel])?;
    let prefix = CudaTrainTensor::new(Tensor::zeros(
        (kernel.saturating_sub(1), channels),
        DType::F32,
        input.as_tensor().device(),
    )?)?;
    let padded = cuda_cat_rows(&[prefix, input.clone()])?;
    let mut output = CudaTrainTensor::new(Tensor::zeros(
        (rows, channels),
        DType::F32,
        input.as_tensor().device(),
    )?)?;
    for j in 0..kernel {
        let x_slice = cuda_narrow_rows(&padded, j, rows)?;
        let w_col_2d = cuda_narrow_last_dim(&weight_2d, j, 1)?;
        let w_col = cuda_reshape(&w_col_2d, &[channels])?;
        let term = cuda_mul_last_dim_weight(&x_slice, &w_col)?;
        output = cuda_add(&output, &term)?;
    }
    Ok(output)
}

pub fn cuda_causal_depthwise_conv1d_prefill_with_state(
    input: &CudaTrainTensor,
    weight: &CudaTrainTensor,
    conv_state: &CudaTrainTensor,
) -> Result<(CudaTrainTensor, CudaTrainTensor)> {
    ensure!(
        input.dtype() == DType::F32 && weight.dtype() == DType::F32 && conv_state.dtype() == DType::F32,
        "cuda_causal_depthwise_conv1d_prefill_with_state: expected F32 input/weight/state, got {:?}/{:?}/{:?}",
        input.dtype(),
        weight.dtype(),
        conv_state.dtype()
    );
    ensure!(
        input.dims().len() == 2 && conv_state.dims().len() == 2,
        "cuda_causal_depthwise_conv1d_prefill_with_state: expected input/state rank-2, got {:?}/{:?}",
        input.dims(),
        conv_state.dims()
    );
    ensure!(
        weight.dims().len() == 2 || weight.dims().len() == 3,
        "cuda_causal_depthwise_conv1d_prefill_with_state: expected weight [channels,k] or [channels,1,k], got {:?}",
        weight.dims()
    );
    let rows = input.dims()[0];
    let channels = input.dims()[1];
    let kernel = *weight
        .dims()
        .last()
        .context("cuda_causal_depthwise_conv1d_prefill_with_state: empty weight shape")?;
    ensure!(
        kernel > 1,
        "cuda_causal_depthwise_conv1d_prefill_with_state: kernel must be > 1"
    );
    ensure!(
        weight.dims()[0] == channels,
        "cuda_causal_depthwise_conv1d_prefill_with_state: weight channels {} != input channels {}",
        weight.dims()[0],
        channels
    );
    if weight.dims().len() == 3 {
        ensure!(
            weight.dims()[1] == 1,
            "cuda_causal_depthwise_conv1d_prefill_with_state: rank-3 weight middle dim must be 1, got {}",
            weight.dims()[1]
        );
    }
    let state_rows = kernel - 1;
    ensure!(
        conv_state.dims() == [state_rows, channels],
        "cuda_causal_depthwise_conv1d_prefill_with_state: state shape {:?} != [{},{}]",
        conv_state.dims(),
        state_rows,
        channels
    );

    let weight_2d = cuda_reshape(weight, &[channels, kernel])?;
    let padded = cuda_cat_rows(&[conv_state.clone(), input.clone()])?;
    let mut output = CudaTrainTensor::new(Tensor::zeros(
        (rows, channels),
        DType::F32,
        input.as_tensor().device(),
    )?)?;
    for j in 0..kernel {
        let x_slice = cuda_narrow_rows(&padded, j, rows)?;
        let w_col_2d = cuda_narrow_last_dim(&weight_2d, j, 1)?;
        let w_col = cuda_reshape(&w_col_2d, &[channels])?;
        let term = cuda_mul_last_dim_weight(&x_slice, &w_col)?;
        output = cuda_add(&output, &term)?;
    }

    let next_state = if rows >= state_rows {
        cuda_narrow_rows(input, rows - state_rows, state_rows)?
    } else {
        let old_part = cuda_narrow_rows(conv_state, rows, state_rows - rows)?;
        cuda_cat_rows(&[old_part, input.clone()])?
    };
    Ok((output, next_state))
}

pub fn cuda_causal_depthwise_conv1d_next_state_zero_state(
    input: &CudaTrainTensor,
    kernel_size: usize,
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_causal_depthwise_conv1d_next_state_zero_state: expected F32 input, got {:?}",
        input.dtype()
    );
    ensure!(
        input.dims().len() == 2,
        "cuda_causal_depthwise_conv1d_next_state_zero_state: expected input [rows, channels], got {:?}",
        input.dims()
    );
    ensure!(
        kernel_size > 1,
        "cuda_causal_depthwise_conv1d_next_state_zero_state: kernel must be > 1"
    );
    let state_rows = kernel_size.saturating_sub(1);
    let rows = input.dims()[0];
    let channels = input.dims()[1];
    if rows >= state_rows {
        return cuda_narrow_rows(input, rows - state_rows, state_rows);
    }
    let prefix = CudaTrainTensor::new(Tensor::zeros(
        (state_rows - rows, channels),
        DType::F32,
        input.as_tensor().device(),
    )?)?;
    cuda_cat_rows(&[prefix, input.clone()])
}

pub struct CudaGdnSingleTokenOutput {
    pub out: CudaTrainTensor,
    pub next_state: CudaTrainTensor,
}

pub fn cuda_gdn_single_token_recurrence(
    q: &CudaTrainTensor,
    k: &CudaTrainTensor,
    v: &CudaTrainTensor,
    beta: &CudaTrainTensor,
    g: &CudaTrainTensor,
    state: &CudaTrainTensor,
) -> Result<CudaGdnSingleTokenOutput> {
    ensure!(
        q.dtype() == DType::F32
            && k.dtype() == DType::F32
            && v.dtype() == DType::F32
            && beta.dtype() == DType::F32
            && g.dtype() == DType::F32
            && state.dtype() == DType::F32,
        "cuda_gdn_single_token_recurrence: expected F32 tensors"
    );
    ensure!(
        q.dims().len() == 2
            && k.dims().len() == 2
            && v.dims().len() == 2
            && beta.dims().len() == 2
            && g.dims().len() == 2
            && state.dims().len() == 2,
        "cuda_gdn_single_token_recurrence: expected q/k/v/beta/g/state ranks 2, got {:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
        q.dims(),
        k.dims(),
        v.dims(),
        beta.dims(),
        g.dims(),
        state.dims()
    );
    let dk = q.dims()[1];
    let dv = v.dims()[1];
    ensure!(
        q.dims()[0] == 1
            && k.dims() == [1, dk]
            && v.dims()[0] == 1
            && beta.dims() == [1, 1]
            && g.dims() == [1, 1]
            && state.dims() == [dk, dv],
        "cuda_gdn_single_token_recurrence: expected q/k=[1,dk], v=[1,dv], beta/g=[1,1], state=[dk,dv], got {:?}/{:?}/{:?}/{:?}/{:?}/{:?}",
        q.dims(),
        k.dims(),
        v.dims(),
        beta.dims(),
        g.dims(),
        state.dims()
    );

    let p = cuda_exp(g).context("cuda GDN single-token exp(g)")?;
    let ks_entry = cuda_matmul(k, state).context("cuda GDN single-token k*S")?;
    let q_s = cuda_matmul(q, state).context("cuda GDN single-token q*S")?;
    let p_ks = cuda_mul_last_dim_broadcast(&ks_entry, &p)?;
    let v_prime = cuda_sub(v, &p_ks)?;
    let w = cuda_mul_last_dim_broadcast(&v_prime, beta)?;
    let k_t = cuda_transpose2d(k)?;
    let qk = cuda_matmul(q, &k_t).context("cuda GDN single-token q*k")?;
    let p_qs = cuda_mul_last_dim_broadcast(&q_s, &p)?;
    let qk_w = cuda_matmul(&qk, &w).context("cuda GDN single-token qk*w")?;
    let out = cuda_add(&p_qs, &qk_w)?;

    let state_flat = cuda_reshape(state, &[1, dk * dv])?;
    let state_scaled_flat = cuda_mul_last_dim_broadcast(&state_flat, &p)?;
    let state_scaled = cuda_reshape(&state_scaled_flat, &[dk, dv])?;
    let delta_state = cuda_matmul(&k_t, &w).context("cuda GDN single-token k^T*w")?;
    let next_state = cuda_add(&state_scaled, &delta_state)?;

    Ok(CudaGdnSingleTokenOutput { out, next_state })
}

pub fn cuda_index_select_rows(
    input: &CudaTrainTensor,
    indices: &[usize],
) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_index_select_rows: expected F32 input, got {:?}",
        input.dtype()
    );
    ensure!(
        input.dims().len() == 2,
        "cuda_index_select_rows: expected rank-2 [rows, dim], got {:?}",
        input.dims()
    );
    ensure!(
        !indices.is_empty(),
        "cuda_index_select_rows: indices must not be empty"
    );
    let rows = input.dims()[0];
    for &idx in indices {
        ensure!(
            idx < rows,
            "cuda_index_select_rows: index {idx} out of bounds for rows={rows}"
        );
    }

    let mut selected = Vec::with_capacity(indices.len());
    for &idx in indices {
        selected.push(
            input
                .as_tensor()
                .narrow(0, idx, 1)
                .with_context(|| format!("cuda_index_select_rows: narrow row {idx}"))?
                .contiguous()
                .with_context(|| format!("cuda_index_select_rows: contiguous row {idx}"))?,
        );
    }
    let refs: Vec<&Tensor> = selected.iter().collect();
    let out = Tensor::cat(&refs, 0).context("cuda_index_select_rows: cat selected rows")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(IndexSelectRowsBackward {
            indices: indices.to_vec(),
            input_rows: input.dims()[0],
            dim: input.dims()[1],
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_embedding_lookup(
    table: &CudaTrainTensor,
    token_ids: &[usize],
) -> Result<CudaTrainTensor> {
    ensure!(
        table.dims().len() == 2,
        "cuda_embedding_lookup: expected rank-2 [vocab, hidden] table, got {:?}",
        table.dims()
    );
    cuda_index_select_rows(table, token_ids)
}

pub fn cuda_permute_rh_to_hr(input: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_permute_rh_to_hr: expected F32 input, got {:?}",
        input.dtype()
    );
    ensure!(
        input.dims().len() == 3,
        "cuda_permute_rh_to_hr: expected rank-3 [rows, heads, head_dim], got {:?}",
        input.dims()
    );
    let out = input
        .as_tensor()
        .transpose(0, 1)
        .context("cuda_permute_rh_to_hr: transpose rows/heads")?
        .contiguous()
        .context("cuda_permute_rh_to_hr: contiguous output")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(PermuteRhToHrBackward {
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_permute_hr_to_rh(input: &CudaTrainTensor) -> Result<CudaTrainTensor> {
    ensure!(
        input.dtype() == DType::F32,
        "cuda_permute_hr_to_rh: expected F32 input, got {:?}",
        input.dtype()
    );
    ensure!(
        input.dims().len() == 3,
        "cuda_permute_hr_to_rh: expected rank-3 [heads, rows, head_dim], got {:?}",
        input.dims()
    );
    let out = input
        .as_tensor()
        .transpose(0, 1)
        .context("cuda_permute_hr_to_rh: transpose heads/rows")?
        .contiguous()
        .context("cuda_permute_hr_to_rh: contiguous output")?;
    let needs_grad =
        input.requires_grad() || input.grad_fn().is_some() || input.param_id().is_some();
    let grad_fn = needs_grad.then(|| {
        Arc::new(PermuteHrToRhBackward {
            inputs: vec![input.clone()],
        }) as Arc<dyn CudaBackwardOp>
    });
    CudaTrainTensor::from_op(out, grad_fn)
}

pub fn cuda_sdpa_unmasked(
    q: &CudaTrainTensor,
    k: &CudaTrainTensor,
    v: &CudaTrainTensor,
    scale: f32,
) -> Result<CudaTrainTensor> {
    ensure!(
        q.dtype() == DType::F32 && k.dtype() == DType::F32 && v.dtype() == DType::F32,
        "cuda_sdpa_unmasked: expected F32 inputs, got q={:?} k={:?} v={:?}",
        q.dtype(),
        k.dtype(),
        v.dtype()
    );
    ensure!(
        q.dims().len() == 3 && k.dims().len() == 3 && v.dims().len() == 3,
        "cuda_sdpa_unmasked: expected rank-3 inputs, got q={:?} k={:?} v={:?}",
        q.dims(),
        k.dims(),
        v.dims()
    );
    let rows = q.dims()[0];
    let heads_q = q.dims()[1];
    let head_dim = q.dims()[2];
    let heads_kv = k.dims()[1];
    ensure!(
        k.dims() == [rows, heads_kv, head_dim],
        "cuda_sdpa_unmasked: k shape {:?} mismatch with q {:?}",
        k.dims(),
        q.dims()
    );
    ensure!(
        v.dims() == [rows, heads_kv, head_dim],
        "cuda_sdpa_unmasked: v shape {:?} mismatch with q {:?}",
        v.dims(),
        q.dims()
    );
    ensure!(
        heads_q % heads_kv == 0,
        "cuda_sdpa_unmasked: heads_q ({heads_q}) must be a multiple of heads_kv ({heads_kv})"
    );
    let groups = heads_q / heads_kv;

    let q_perm = cuda_permute_rh_to_hr(q)?;
    let k_perm = cuda_permute_rh_to_hr(k)?;
    let v_perm = cuda_permute_rh_to_hr(v)?;
    let k_bcast = cuda_repeat_kv_heads(&k_perm, groups)?;
    let v_bcast = cuda_repeat_kv_heads(&v_perm, groups)?;
    let k_t = cuda_transpose_last_two(&k_bcast)?;
    let scores = cuda_batched_matmul(&q_perm, &k_t)?;
    let scaled = cuda_scale(&scores, scale)?;
    let attn = cuda_softmax_last_dim(&scaled)?;
    let out_perm = cuda_batched_matmul(&attn, &v_bcast)?;
    cuda_permute_hr_to_rh(&out_perm)
}

pub fn cuda_sdpa_prefill_causal(
    q: &CudaTrainTensor,
    k: &CudaTrainTensor,
    v: &CudaTrainTensor,
    scale: f32,
) -> Result<CudaTrainTensor> {
    ensure!(
        q.dtype() == DType::F32 && k.dtype() == DType::F32 && v.dtype() == DType::F32,
        "cuda_sdpa_prefill_causal: expected F32 inputs, got q={:?} k={:?} v={:?}",
        q.dtype(),
        k.dtype(),
        v.dtype()
    );
    ensure!(
        q.dims().len() == 3 && k.dims().len() == 3 && v.dims().len() == 3,
        "cuda_sdpa_prefill_causal: expected rank-3 inputs, got q={:?} k={:?} v={:?}",
        q.dims(),
        k.dims(),
        v.dims()
    );
    ensure!(
        q.dims()[0] == k.dims()[0] && q.dims()[0] == v.dims()[0],
        "cuda_sdpa_prefill_causal: prefill requires q/k/v row counts to match, got q={:?} k={:?} v={:?}",
        q.dims(),
        k.dims(),
        v.dims()
    );
    let rows = q.dims()[0];
    let heads_q = q.dims()[1];
    let head_dim = q.dims()[2];
    let heads_kv = k.dims()[1];
    ensure!(
        k.dims() == [rows, heads_kv, head_dim],
        "cuda_sdpa_prefill_causal: k shape {:?} mismatch with q {:?}",
        k.dims(),
        q.dims()
    );
    ensure!(
        v.dims() == [rows, heads_kv, head_dim],
        "cuda_sdpa_prefill_causal: v shape {:?} mismatch with q {:?}",
        v.dims(),
        q.dims()
    );
    ensure!(
        heads_q % heads_kv == 0,
        "cuda_sdpa_prefill_causal: heads_q ({heads_q}) must be a multiple of heads_kv ({heads_kv})"
    );
    let groups = heads_q / heads_kv;

    let q_perm = cuda_permute_rh_to_hr(q)?;
    let k_perm = cuda_permute_rh_to_hr(k)?;
    let v_perm = cuda_permute_rh_to_hr(v)?;
    let k_bcast = cuda_repeat_kv_heads(&k_perm, groups)?;
    let v_bcast = cuda_repeat_kv_heads(&v_perm, groups)?;
    let k_t = cuda_transpose_last_two(&k_bcast)?;
    let scores = cuda_batched_matmul(&q_perm, &k_t)?;
    let scaled = cuda_scale(&scores, scale)?;
    let masked = cuda_causal_mask(&scaled, 0)?;
    let attn = cuda_softmax_last_dim(&masked)?;
    let out_perm = cuda_batched_matmul(&attn, &v_bcast)?;
    cuda_permute_hr_to_rh(&out_perm)
}

#[derive(Debug)]
struct AddBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for AddBackward {
    fn op_name(&self) -> &'static str {
        "cuda_add"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        Ok(self.inputs.iter().map(|_| Some(grad_out.clone())).collect())
    }
}

#[derive(Debug)]
struct AddLastDimBiasBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for AddLastDimBiasBackward {
    fn op_name(&self) -> &'static str {
        "cuda_add_last_dim_bias"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 2,
            "cuda_add_last_dim_bias backward expected two inputs, got {}",
            self.inputs.len()
        );
        let input_grad = grad_out.clone();
        let mut bias_grad = grad_out.as_tensor().clone();
        while bias_grad.dims().len() > 1 {
            bias_grad = bias_grad
                .sum(0)
                .context("cuda_add_last_dim_bias backward: reduce leading dim")?;
        }
        Ok(vec![
            Some(input_grad),
            Some(CudaTrainTensor::new(
                bias_grad
                    .contiguous()
                    .context("cuda_add_last_dim_bias backward: contiguous bias grad")?,
            )?),
        ])
    }
}

#[derive(Debug)]
struct SubBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for SubBackward {
    fn op_name(&self) -> &'static str {
        "cuda_sub"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 2,
            "cuda_sub backward expected two inputs, got {}",
            self.inputs.len()
        );
        let rhs_grad = grad_out
            .as_tensor()
            .affine(-1.0, 0.0)
            .context("cuda_sub backward: rhs neg grad")?;
        Ok(vec![
            Some(grad_out.clone()),
            Some(CudaTrainTensor::new(rhs_grad)?),
        ])
    }
}

#[derive(Debug)]
struct MulBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for MulBackward {
    fn op_name(&self) -> &'static str {
        "cuda_mul"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 2,
            "cuda_mul backward expected two inputs, got {}",
            self.inputs.len()
        );
        let lhs_grad = (grad_out.as_tensor() * self.inputs[1].as_tensor())
            .context("cuda_mul backward: lhs grad")?;
        let rhs_grad = (grad_out.as_tensor() * self.inputs[0].as_tensor())
            .context("cuda_mul backward: rhs grad")?;
        Ok(vec![
            Some(CudaTrainTensor::new(lhs_grad)?),
            Some(CudaTrainTensor::new(rhs_grad)?),
        ])
    }
}

#[derive(Debug)]
struct MulLastDimWeightBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for MulLastDimWeightBackward {
    fn op_name(&self) -> &'static str {
        "cuda_mul_last_dim_weight"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 2,
            "cuda_mul_last_dim_weight backward expected two inputs, got {}",
            self.inputs.len()
        );
        let input = &self.inputs[0];
        let weight = &self.inputs[1];
        let input_grad = grad_out
            .as_tensor()
            .broadcast_mul(weight.as_tensor())
            .context("cuda_mul_last_dim_weight backward: input grad")?;
        let mut weight_grad = (grad_out.as_tensor() * input.as_tensor())
            .context("cuda_mul_last_dim_weight backward: weighted grad")?;
        while weight_grad.dims().len() > 1 {
            weight_grad = weight_grad
                .sum(0)
                .context("cuda_mul_last_dim_weight backward: reduce leading dim")?;
        }
        Ok(vec![
            Some(CudaTrainTensor::new(input_grad)?),
            Some(CudaTrainTensor::new(
                weight_grad
                    .contiguous()
                    .context("cuda_mul_last_dim_weight backward: contiguous weight grad")?,
            )?),
        ])
    }
}

#[derive(Debug)]
struct MulLastDimBroadcastBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for MulLastDimBroadcastBackward {
    fn op_name(&self) -> &'static str {
        "cuda_mul_last_dim_broadcast"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 2,
            "cuda_mul_last_dim_broadcast backward expected two inputs, got {}",
            self.inputs.len()
        );
        let input = &self.inputs[0];
        let scalar = &self.inputs[1];
        let input_grad = grad_out
            .as_tensor()
            .broadcast_mul(scalar.as_tensor())
            .context("cuda_mul_last_dim_broadcast backward: input grad")?;
        let scalar_grad = (grad_out.as_tensor() * input.as_tensor())
            .context("cuda_mul_last_dim_broadcast backward: weighted grad")?
            .sum_keepdim(D::Minus1)
            .context("cuda_mul_last_dim_broadcast backward: reduce last dim")?;
        Ok(vec![
            Some(CudaTrainTensor::new(input_grad)?),
            Some(CudaTrainTensor::new(
                scalar_grad
                    .contiguous()
                    .context("cuda_mul_last_dim_broadcast backward: contiguous scalar grad")?,
            )?),
        ])
    }
}

#[derive(Debug)]
struct DivBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for DivBackward {
    fn op_name(&self) -> &'static str {
        "cuda_div"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 2,
            "cuda_div backward expected two inputs, got {}",
            self.inputs.len()
        );
        let lhs = &self.inputs[0];
        let rhs = &self.inputs[1];
        let lhs_grad = grad_out
            .as_tensor()
            .broadcast_div(rhs.as_tensor())
            .context("cuda_div backward: lhs grad")?;
        let rhs_sq = (rhs.as_tensor() * rhs.as_tensor()).context("cuda_div backward: rhs square")?;
        let neg_lhs = lhs
            .as_tensor()
            .affine(-1.0, 0.0)
            .context("cuda_div backward: neg lhs")?;
        let rhs_factor = neg_lhs
            .broadcast_div(&rhs_sq)
            .context("cuda_div backward: rhs factor")?;
        let rhs_grad =
            (grad_out.as_tensor() * &rhs_factor).context("cuda_div backward: rhs grad")?;
        Ok(vec![
            Some(CudaTrainTensor::new(lhs_grad)?),
            Some(CudaTrainTensor::new(rhs_grad)?),
        ])
    }
}

#[derive(Debug)]
struct ScaleBackward {
    inputs: Vec<CudaTrainTensor>,
    scale: f32,
}

impl CudaBackwardOp for ScaleBackward {
    fn op_name(&self) -> &'static str {
        "cuda_scale"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_scale backward expected one input, got {}",
            self.inputs.len()
        );
        let grad = grad_out
            .as_tensor()
            .affine(self.scale as f64, 0.0)
            .context("cuda_scale backward: scale grad")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct CausalMaskBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for CausalMaskBackward {
    fn op_name(&self) -> &'static str {
        "cuda_causal_mask"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        Ok(vec![Some(CudaTrainTensor::new(
            grad_out.as_tensor().clone(),
        )?)])
    }
}

#[derive(Debug)]
struct RmsNormBackward {
    inputs: Vec<CudaTrainTensor>,
    eps: f32,
}

impl CudaBackwardOp for RmsNormBackward {
    fn op_name(&self) -> &'static str {
        "cuda_rmsnorm"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 2,
            "cuda_rmsnorm backward expected two inputs, got {}",
            self.inputs.len()
        );
        let input = &self.inputs[0];
        let weight = &self.inputs[1];
        let hidden = *input
            .dims()
            .last()
            .context("cuda_rmsnorm backward: input must have at least one dimension")?;
        let weight_plus_one = (weight.as_tensor().ones_like()? + weight.as_tensor())
            .context("cuda_rmsnorm backward: weight plus one")?;
        let u = grad_out
            .as_tensor()
            .broadcast_mul(&weight_plus_one)
            .context("cuda_rmsnorm backward: grad * weight")?;
        let dot = (&u * input.as_tensor())
            .context("cuda_rmsnorm backward: u * input")?
            .sum_keepdim(D::Minus1)
            .context("cuda_rmsnorm backward: row dot")?;
        let variance = input
            .as_tensor()
            .sqr()
            .context("cuda_rmsnorm backward: square input")?
            .mean_keepdim(D::Minus1)
            .context("cuda_rmsnorm backward: row variance")?;
        let rms_inv = (variance + self.eps as f64)
            .context("cuda_rmsnorm backward: add eps")?
            .sqrt()
            .context("cuda_rmsnorm backward: sqrt variance")?
            .recip()
            .context("cuda_rmsnorm backward: reciprocal rms")?;
        let rms_inv_sq = rms_inv
            .sqr()
            .context("cuda_rmsnorm backward: inv rms square")?;
        let rms_inv_cubed = rms_inv_sq
            .broadcast_mul(&rms_inv)
            .context("cuda_rmsnorm backward: inv rms cubed")?;
        let correction_scale = rms_inv_cubed
            .affine(1.0f64 / hidden as f64, 0.0)
            .context("cuda_rmsnorm backward: correction scale")?;
        let correction = input
            .as_tensor()
            .broadcast_mul(
                &dot.broadcast_mul(&correction_scale)
                    .context("cuda_rmsnorm backward: dot correction")?,
            )
            .context("cuda_rmsnorm backward: correction")?;
        let grad_input = (u
            .broadcast_mul(&rms_inv)
            .context("cuda_rmsnorm backward: direct grad")?
            - correction)
            .context("cuda_rmsnorm backward: input grad")?;
        Ok(vec![Some(CudaTrainTensor::new(grad_input)?), None])
    }
}

#[derive(Debug)]
struct RopeBackward {
    cos: CudaTrainTensor,
    sin: CudaTrainTensor,
    rotary_dim: usize,
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for RopeBackward {
    fn op_name(&self) -> &'static str {
        "cuda_rope"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_rope backward expected one input, got {}",
            self.inputs.len()
        );
        let grad = cuda_rope_apply(grad_out, &self.cos, &self.sin, self.rotary_dim, true)
            .context("cuda_rope backward: inverse rotation")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct CastBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for CastBackward {
    fn op_name(&self) -> &'static str {
        "cuda_to_dtype"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_to_dtype backward expected one input, got {}",
            self.inputs.len()
        );
        let input = &self.inputs[0];
        let grad = grad_out
            .as_tensor()
            .to_dtype(input.dtype())
            .with_context(|| format!("cuda_to_dtype backward: restore {:?}", input.dtype()))?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct SumAllBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for SumAllBackward {
    fn op_name(&self) -> &'static str {
        "cuda_sum_all"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_sum_all backward expected one input, got {}",
            self.inputs.len()
        );
        let input = &self.inputs[0];
        let grad = grad_out
            .as_tensor()
            .broadcast_as(input.as_tensor().shape())
            .context("cuda_sum_all backward: broadcast scalar grad")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct MeanAllBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for MeanAllBackward {
    fn op_name(&self) -> &'static str {
        "cuda_mean_all"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_mean_all backward expected one input, got {}",
            self.inputs.len()
        );
        let input = &self.inputs[0];
        let scale = 1.0 / input.num_elements() as f64;
        let grad = grad_out
            .as_tensor()
            .broadcast_as(input.as_tensor().shape())
            .context("cuda_mean_all backward: broadcast scalar grad")?
            .affine(scale, 0.0)
            .context("cuda_mean_all backward: scale grad")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct PermuteRhToHrBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for PermuteRhToHrBackward {
    fn op_name(&self) -> &'static str {
        "cuda_permute_rh_to_hr"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        Ok(vec![Some(cuda_permute_hr_to_rh(grad_out)?)])
    }
}

#[derive(Debug)]
struct PermuteHrToRhBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for PermuteHrToRhBackward {
    fn op_name(&self) -> &'static str {
        "cuda_permute_hr_to_rh"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        Ok(vec![Some(cuda_permute_rh_to_hr(grad_out)?)])
    }
}

#[derive(Debug)]
struct IndexSelectRowsBackward {
    indices: Vec<usize>,
    input_rows: usize,
    dim: usize,
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for IndexSelectRowsBackward {
    fn op_name(&self) -> &'static str {
        "cuda_index_select_rows"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_index_select_rows backward expected one input, got {}",
            self.inputs.len()
        );
        ensure!(
            grad_out.dims() == [self.indices.len(), self.dim].as_slice(),
            "cuda_index_select_rows backward: grad shape {:?} incompatible with n_out={} dim={}",
            grad_out.dims(),
            self.indices.len(),
            self.dim
        );

        let mut rows = Vec::with_capacity(self.input_rows);
        for input_row in 0..self.input_rows {
            let mut accum = Tensor::zeros(
                vec![1usize, self.dim],
                grad_out.dtype(),
                grad_out.as_tensor().device(),
            )
            .with_context(|| format!("cuda_index_select_rows backward: zero row {input_row}"))?;
            for (out_row, &idx) in self.indices.iter().enumerate() {
                if idx != input_row {
                    continue;
                }
                let grad_row = grad_out
                    .as_tensor()
                    .narrow(0, out_row, 1)
                    .with_context(|| {
                        format!("cuda_index_select_rows backward: narrow grad row {out_row}")
                    })?
                    .contiguous()
                    .with_context(|| {
                        format!("cuda_index_select_rows backward: contiguous grad row {out_row}")
                    })?;
                accum = (&accum + &grad_row)
                    .context("cuda_index_select_rows backward: scatter-add row")?;
            }
            rows.push(accum);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let grad = Tensor::cat(&refs, 0).context("cuda_index_select_rows backward: cat rows")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct NarrowLastDimBackward {
    start: usize,
    len: usize,
    input_dims: Vec<usize>,
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for NarrowLastDimBackward {
    fn op_name(&self) -> &'static str {
        "cuda_narrow_last_dim"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_narrow_last_dim backward expected one input, got {}",
            self.inputs.len()
        );
        let last_dim = *self
            .input_dims
            .last()
            .context("cuda_narrow_last_dim backward: empty input dims")?;
        ensure!(
            self.start + self.len <= last_dim,
            "cuda_narrow_last_dim backward: invalid saved slice"
        );

        let mut chunks = Vec::new();
        if self.start > 0 {
            let mut shape = self.input_dims.clone();
            *shape.last_mut().expect("checked non-empty shape") = self.start;
            chunks.push(
                Tensor::zeros(shape, grad_out.dtype(), grad_out.as_tensor().device())
                    .context("cuda_narrow_last_dim backward: prefix zeros")?,
            );
        }
        chunks.push(grad_out.as_tensor().clone());
        let suffix = last_dim - self.start - self.len;
        if suffix > 0 {
            let mut shape = self.input_dims.clone();
            *shape.last_mut().expect("checked non-empty shape") = suffix;
            chunks.push(
                Tensor::zeros(shape, grad_out.dtype(), grad_out.as_tensor().device())
                    .context("cuda_narrow_last_dim backward: suffix zeros")?,
            );
        }
        let refs: Vec<&Tensor> = chunks.iter().collect();
        let grad = Tensor::cat(&refs, D::Minus1)
            .context("cuda_narrow_last_dim backward: cat padded grad")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct NarrowRowsBackward {
    start: usize,
    len: usize,
    input_dims: Vec<usize>,
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for NarrowRowsBackward {
    fn op_name(&self) -> &'static str {
        "cuda_narrow_rows"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_narrow_rows backward expected one input, got {}",
            self.inputs.len()
        );
        ensure!(
            self.input_dims.len() == 2,
            "cuda_narrow_rows backward expected rank-2 saved input dims, got {:?}",
            self.input_dims
        );
        let rows = self.input_dims[0];
        let dim = self.input_dims[1];
        ensure!(
            self.start + self.len <= rows,
            "cuda_narrow_rows backward: invalid saved slice"
        );
        ensure!(
            grad_out.dims() == [self.len, dim],
            "cuda_narrow_rows backward: grad shape {:?} incompatible with [{},{}]",
            grad_out.dims(),
            self.len,
            dim
        );

        let mut chunks = Vec::new();
        if self.start > 0 {
            chunks.push(
                Tensor::zeros((self.start, dim), grad_out.dtype(), grad_out.as_tensor().device())
                    .context("cuda_narrow_rows backward: prefix zeros")?,
            );
        }
        chunks.push(grad_out.as_tensor().clone());
        let suffix = rows - self.start - self.len;
        if suffix > 0 {
            chunks.push(
                Tensor::zeros((suffix, dim), grad_out.dtype(), grad_out.as_tensor().device())
                    .context("cuda_narrow_rows backward: suffix zeros")?,
            );
        }
        let refs: Vec<&Tensor> = chunks.iter().collect();
        let grad =
            Tensor::cat(&refs, 0).context("cuda_narrow_rows backward: cat padded grad")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct CatRowsBackward {
    row_counts: Vec<usize>,
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for CatRowsBackward {
    fn op_name(&self) -> &'static str {
        "cuda_cat_rows"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == self.row_counts.len(),
            "cuda_cat_rows backward input/count mismatch: {} vs {}",
            self.inputs.len(),
            self.row_counts.len()
        );
        ensure!(
            grad_out.dims().len() == 2,
            "cuda_cat_rows backward expected rank-2 grad, got {:?}",
            grad_out.dims()
        );
        let total_rows: usize = self.row_counts.iter().sum();
        ensure!(
            grad_out.dims()[0] == total_rows,
            "cuda_cat_rows backward grad rows {} != saved total {}",
            grad_out.dims()[0],
            total_rows
        );
        let mut offset = 0usize;
        let mut grads = Vec::with_capacity(self.row_counts.len());
        for &rows in &self.row_counts {
            let end = offset + rows;
            let grad = grad_out
                .as_tensor()
                .narrow(0, offset, rows)
                .with_context(|| format!("cuda_cat_rows backward: narrow rows {offset}..{end}"))?
                .contiguous()
                .context("cuda_cat_rows backward: contiguous grad slice")?;
            grads.push(Some(CudaTrainTensor::new(grad)?));
            offset = end;
        }
        Ok(grads)
    }
}

#[derive(Debug)]
struct RepeatKvHeadsBackward {
    heads_kv: usize,
    groups: usize,
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for RepeatKvHeadsBackward {
    fn op_name(&self) -> &'static str {
        "cuda_repeat_kv_heads"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_repeat_kv_heads backward expected one input, got {}",
            self.inputs.len()
        );
        ensure!(
            grad_out.dims().len() == 3 && grad_out.dims()[0] == self.heads_kv * self.groups,
            "cuda_repeat_kv_heads backward: grad shape {:?} incompatible with heads={} groups={}",
            grad_out.dims(),
            self.heads_kv,
            self.groups
        );

        let mut head_grads = Vec::with_capacity(self.heads_kv);
        for head in 0..self.heads_kv {
            let mut accum: Option<Tensor> = None;
            for group in 0..self.groups {
                let slot = head * self.groups + group;
                let slice = grad_out
                    .as_tensor()
                    .narrow(0, slot, 1)
                    .with_context(|| {
                        format!("cuda_repeat_kv_heads backward: narrow repeated head {slot}")
                    })?
                    .contiguous()
                    .with_context(|| {
                        format!("cuda_repeat_kv_heads backward: contiguous repeated head {slot}")
                    })?;
                accum = Some(match accum {
                    Some(existing) => {
                        (&existing + &slice).context("cuda_repeat_kv_heads backward: sum group")?
                    }
                    None => slice,
                });
            }
            head_grads.push(accum.context("cuda_repeat_kv_heads backward: empty group")?);
        }
        let refs: Vec<&Tensor> = head_grads.iter().collect();
        let grad = Tensor::cat(&refs, 0).context("cuda_repeat_kv_heads backward: cat heads")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct SigmoidBackward {
    output: CudaTrainTensor,
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for SigmoidBackward {
    fn op_name(&self) -> &'static str {
        "cuda_sigmoid"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_sigmoid backward expected one input, got {}",
            self.inputs.len()
        );
        let one_minus = self
            .output
            .as_tensor()
            .affine(-1.0, 1.0)
            .context("cuda_sigmoid backward: one minus output")?;
        let deriv =
            (self.output.as_tensor() * &one_minus).context("cuda_sigmoid backward: derivative")?;
        let grad = (grad_out.as_tensor() * &deriv).context("cuda_sigmoid backward: input grad")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct ExpBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for ExpBackward {
    fn op_name(&self) -> &'static str {
        "cuda_exp"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_exp backward expected one input, got {}",
            self.inputs.len()
        );
        let exp = self.inputs[0]
            .as_tensor()
            .exp()
            .context("cuda_exp backward: exp input")?;
        let grad = (grad_out.as_tensor() * &exp).context("cuda_exp backward: input grad")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct SoftplusBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for SoftplusBackward {
    fn op_name(&self) -> &'static str {
        "cuda_softplus"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_softplus backward expected one input, got {}",
            self.inputs.len()
        );
        let exp_neg = self.inputs[0]
            .as_tensor()
            .neg()
            .context("cuda_softplus backward: negate input")?
            .exp()
            .context("cuda_softplus backward: exp -input")?;
        let sigmoid = (exp_neg + 1.0)
            .context("cuda_softplus backward: add one")?
            .recip()
            .context("cuda_softplus backward: reciprocal")?;
        let grad =
            (grad_out.as_tensor() * &sigmoid).context("cuda_softplus backward: input grad")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct SiluBackward {
    sigmoid: CudaTrainTensor,
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for SiluBackward {
    fn op_name(&self) -> &'static str {
        "cuda_silu"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_silu backward expected one input, got {}",
            self.inputs.len()
        );
        let input = &self.inputs[0];
        let one_minus_sigmoid = self
            .sigmoid
            .as_tensor()
            .affine(-1.0, 1.0)
            .context("cuda_silu backward: one minus sigmoid")?;
        let x_term = (input.as_tensor() * &one_minus_sigmoid)
            .context("cuda_silu backward: x * (1 - sigmoid)")?;
        let bracket = x_term
            .affine(1.0, 1.0)
            .context("cuda_silu backward: derivative bracket")?;
        let deriv = (self.sigmoid.as_tensor() * &bracket)
            .context("cuda_silu backward: derivative")?;
        let grad = (grad_out.as_tensor() * &deriv).context("cuda_silu backward: input grad")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct SoftmaxLastDimBackward {
    output: CudaTrainTensor,
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for SoftmaxLastDimBackward {
    fn op_name(&self) -> &'static str {
        "cuda_softmax_last_dim"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_softmax_last_dim backward expected one input, got {}",
            self.inputs.len()
        );
        let weighted = (self.output.as_tensor() * grad_out.as_tensor())
            .context("cuda_softmax_last_dim backward: y * grad_out")?;
        let row_dot = weighted
            .sum_keepdim(D::Minus1)
            .context("cuda_softmax_last_dim backward: row dot")?;
        let centered = grad_out
            .as_tensor()
            .broadcast_sub(&row_dot)
            .context("cuda_softmax_last_dim backward: center grad")?;
        let grad = (self.output.as_tensor() * &centered)
            .context("cuda_softmax_last_dim backward: input grad")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct BatchedMatmulBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for BatchedMatmulBackward {
    fn op_name(&self) -> &'static str {
        "cuda_batched_matmul"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 2,
            "cuda_batched_matmul backward expected two inputs, got {}",
            self.inputs.len()
        );
        let lhs = &self.inputs[0];
        let rhs = &self.inputs[1];
        let rank = lhs.dims().len();
        let grad = grad_out
            .as_tensor()
            .contiguous()
            .context("cuda_batched_matmul backward: grad_out contiguous")?;
        let rhs_t = rhs
            .as_tensor()
            .transpose(rank - 2, rank - 1)
            .context("cuda_batched_matmul backward: rhs transpose")?
            .contiguous()
            .context("cuda_batched_matmul backward: rhs_t contiguous")?;
        let lhs_t = lhs
            .as_tensor()
            .transpose(rank - 2, rank - 1)
            .context("cuda_batched_matmul backward: lhs transpose")?
            .contiguous()
            .context("cuda_batched_matmul backward: lhs_t contiguous")?;
        let lhs_grad = grad
            .broadcast_matmul(&rhs_t)
            .context("cuda_batched_matmul backward: lhs grad")?;
        let rhs_grad = lhs_t
            .broadcast_matmul(&grad)
            .context("cuda_batched_matmul backward: rhs grad")?;
        Ok(vec![
            Some(CudaTrainTensor::new(lhs_grad)?),
            Some(CudaTrainTensor::new(rhs_grad)?),
        ])
    }
}

#[derive(Debug)]
struct ReshapeBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for ReshapeBackward {
    fn op_name(&self) -> &'static str {
        "cuda_reshape"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_reshape backward expected one input, got {}",
            self.inputs.len()
        );
        let input = &self.inputs[0];
        let grad = grad_out
            .as_tensor()
            .reshape(input.dims())
            .context("cuda_reshape backward: restore input shape")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct Transpose2dBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for Transpose2dBackward {
    fn op_name(&self) -> &'static str {
        "cuda_transpose2d"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 1,
            "cuda_transpose2d backward expected one input, got {}",
            self.inputs.len()
        );
        let grad = grad_out
            .as_tensor()
            .t()
            .context("cuda_transpose2d backward: transpose grad")?
            .contiguous()
            .context("cuda_transpose2d backward: contiguous grad")?;
        Ok(vec![Some(CudaTrainTensor::new(grad)?)])
    }
}

#[derive(Debug)]
struct TransposeLastTwoBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for TransposeLastTwoBackward {
    fn op_name(&self) -> &'static str {
        "cuda_transpose_last_two"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        Ok(vec![Some(cuda_transpose_last_two(grad_out)?)])
    }
}

#[derive(Debug)]
struct MatmulBackward {
    inputs: Vec<CudaTrainTensor>,
}

impl CudaBackwardOp for MatmulBackward {
    fn op_name(&self) -> &'static str {
        "cuda_matmul"
    }

    fn input_refs(&self) -> &[CudaTrainTensor] {
        &self.inputs
    }

    fn backward(&self, grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
        ensure!(
            self.inputs.len() == 2,
            "cuda_matmul backward expected two inputs, got {}",
            self.inputs.len()
        );
        let lhs = &self.inputs[0];
        let rhs = &self.inputs[1];
        let grad = grad_out
            .as_tensor()
            .contiguous()
            .context("cuda_matmul backward: grad_out contiguous")?;
        let rhs_t = rhs
            .as_tensor()
            .t()
            .context("cuda_matmul backward: rhs transpose")?
            .contiguous()
            .context("cuda_matmul backward: rhs_t contiguous")?;
        let lhs_t = lhs
            .as_tensor()
            .t()
            .context("cuda_matmul backward: lhs transpose")?
            .contiguous()
            .context("cuda_matmul backward: lhs_t contiguous")?;
        let lhs_grad = grad
            .matmul(&rhs_t)
            .context("cuda_matmul backward: lhs grad")?;
        let rhs_grad = lhs_t
            .matmul(&grad)
            .context("cuda_matmul backward: rhs grad")?;
        Ok(vec![
            Some(CudaTrainTensor::new(lhs_grad)?),
            Some(CudaTrainTensor::new(rhs_grad)?),
        ])
    }
}

fn collect_topo(
    t: &CudaTrainTensor,
    visited: &mut HashSet<u64>,
    order: &mut Vec<CudaTrainTensor>,
    leaves: &mut Vec<CudaTrainTensor>,
) {
    if !visited.insert(t.op_id()) {
        return;
    }
    if let Some(gf) = t.grad_fn() {
        for input in gf.input_refs() {
            collect_topo(input, visited, order, leaves);
        }
        order.push(t.clone());
    } else {
        leaves.push(t.clone());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cuda_model_config() -> ModelConfig {
        ModelConfig {
            hidden_size: 2,
            num_layers: 1,
            num_attention_heads: 1,
            num_kv_heads: 1,
            head_dim: 2,
            intermediate_size: 2,
            vocab_size: 4,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            dtype: kiln_core::config::DType::FP32,
            num_full_attention_layers: 1,
            full_attention_interval: 1,
            attn_output_gate: false,
            linear_num_key_heads: 1,
            linear_key_head_dim: 2,
            linear_num_value_heads: 1,
            linear_value_head_dim: 2,
            linear_conv_kernel_dim: 4,
            partial_rotary_factor: 1.0,
        }
    }

    #[test]
    fn cuda_model_weights_from_gpu_weights_imports_full_attention_layer() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda model weight import smoke: {err}");
                return Ok(());
            }
        };
        let config = tiny_cuda_model_config();
        let embed_tokens = Tensor::from_vec(
            vec![0.1f32, -0.2, 0.3, 0.4, 0.5, -0.6, 0.7, 0.8],
            (4usize, 2usize),
            &device,
        )?;
        let embed_tokens_t = embed_tokens.t()?.contiguous()?;
        let q_proj = Tensor::from_vec(vec![0.2f32, -0.3, 0.4, 0.1], (2usize, 2usize), &device)?;
        let k_proj = Tensor::from_vec(vec![0.1f32, 0.6, 0.8, -0.2], (2usize, 2usize), &device)?;
        let v_proj = Tensor::from_vec(vec![0.7f32, -0.2, -0.5, 0.6], (2usize, 2usize), &device)?;
        let o_proj = Tensor::from_vec(vec![0.3f32, -0.4, 0.8, 0.2], (2usize, 2usize), &device)?;
        let gate_proj =
            Tensor::from_vec(vec![0.25f32, -0.15, 0.35, 0.05], (2usize, 2usize), &device)?;
        let up_proj = Tensor::from_vec(vec![0.45f32, 0.2, -0.1, 0.55], (2usize, 2usize), &device)?;
        let down_proj =
            Tensor::from_vec(vec![0.6f32, -0.25, 0.15, 0.5], (2usize, 2usize), &device)?;
        let weights = GpuWeights {
            embed_tokens,
            embed_tokens_t,
            layers: vec![GpuLayerWeights {
                input_layernorm: Tensor::zeros((2usize,), DType::F32, &device)?,
                post_attention_layernorm: Tensor::zeros((2usize,), DType::F32, &device)?,
                attention: GpuAttentionWeights::Full(GpuFullAttentionWeights {
                    q_proj_t: q_proj.t()?.contiguous()?,
                    k_proj_t: k_proj.t()?.contiguous()?,
                    v_proj_t: v_proj.t()?.contiguous()?,
                    o_proj_t: o_proj.t()?.contiguous()?,
                    q_proj,
                    k_proj,
                    v_proj,
                    o_proj,
                    q_norm: Tensor::zeros((2usize,), DType::F32, &device)?,
                    k_norm: Tensor::zeros((2usize,), DType::F32, &device)?,
                    qkv_proj_t: None,
                    q_proj_marlin: None,
                }),
                mlp: crate::forward::GpuFfnWeights {
                    gate_proj_t: gate_proj.t()?.contiguous()?,
                    up_proj_t: up_proj.t()?.contiguous()?,
                    down_proj_t: down_proj.t()?.contiguous()?,
                    gate_proj,
                    up_proj,
                    down_proj,
                    gate_proj_marlin: None,
                    up_proj_marlin: None,
                    down_proj_marlin: None,
                },
            }],
            final_norm: Tensor::zeros((2usize,), DType::F32, &device)?,
            rotary_inv_freq: Tensor::from_vec(vec![1.0f32], (1usize,), &device)?,
            mtp: None,
        };

        let imported = CudaModelWeights::from_gpu_weights(&weights, &config)?;
        assert_eq!(imported.vocab, 4);
        assert_eq!(imported.hidden, 2);
        assert_eq!(imported.rotary_dim, 2);
        assert_eq!(imported.token_embedding.dims(), &[4, 2]);
        assert_eq!(imported.lm_head_weight.dims(), &[2, 4]);
        let CudaLayerWeights::FullAttention(layer) = &imported.layers[0] else {
            panic!("expected imported FullAttention layer");
        };
        assert_eq!(cuda_count_gdn_layers(&imported), 0);
        assert_eq!(layer.q_weight.dims(), &[2, 2]);
        assert_eq!(layer.heads_q, 1);
        assert!(!layer.q_weight.requires_grad());

        let input = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.1f32, 0.2, -0.3, 0.4],
            (2usize, 2usize),
            &device,
        )?)?;
        let output = cuda_full_attention_layer(&input, &layer.as_borrowed(None))?;
        assert_eq!(output.dims(), &[2, 2]);
        Ok(())
    }

    #[test]
    fn cuda_linear_attention_state_zeros_allocates_layer_state() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda GDN state smoke: {err}");
                return Ok(());
            }
        };

        let state = CudaLinearAttentionState::zeros(&device, 2, 3, 4, 5, 6, 7, 4)?;
        assert_eq!(state.layers.len(), 2);
        for layer in &state.layers {
            assert_eq!(layer.recurrent_n_elements, 3 * 4 * 5 * 6);
            assert_eq!(layer.recurrent_state.dims(), &[3, 4, 5, 6]);
            assert_eq!(layer.conv_n_elements, 3 * 7 * 3);
            assert_eq!(layer.conv_state.dims(), &[3, 7, 3]);
            assert!(layer.recurrent_state.to_vec_f32()?.iter().all(|v| *v == 0.0));
            assert!(layer.conv_state.to_vec_f32()?.iter().all(|v| *v == 0.0));
        }
        Ok(())
    }

    #[test]
    fn cuda_train_tensor_rejects_cpu_tensor() -> Result<()> {
        let cpu = Tensor::zeros((2usize,), DType::F32, &Device::Cpu)?;
        let err = CudaTrainTensor::new(cpu).unwrap_err();
        assert!(err.to_string().contains("requires a CUDA tensor"));
        Ok(())
    }

    #[test]
    fn cuda_train_op_ids_are_monotonic() {
        let first = next_cuda_train_op_id();
        let second = next_cuda_train_op_id();
        assert!(second > first);
    }

    #[test]
    fn cuda_train_tensor_sgd_and_adamw_update_in_place() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_train_tensor smoke: {err}");
                return Ok(());
            }
        };

        let param = CudaTrainTensor::new(Tensor::new(vec![1.0f32, -2.0], &device)?)?;
        let grad = CudaTrainTensor::new(Tensor::new(vec![0.25f32, -0.5], &device)?)?;
        param.sgd_step_inplace(&grad, 0.5)?;
        let actual = param.to_vec_f32()?;
        assert!((actual[0] - 0.875).abs() < 1e-6);
        assert!((actual[1] + 1.75).abs() < 1e-6);

        let m = CudaTrainTensor::zeros_like(param.as_tensor())?;
        let v = CudaTrainTensor::zeros_like(param.as_tensor())?;
        param.adamw_step_inplace(&grad, &m, &v, 0.01, 0.9, 0.999, 1e-8, 0.0, 1)?;
        let after_adam = param.to_vec_f32()?;
        assert!(
            after_adam[0] < actual[0],
            "positive grad should reduce first param"
        );
        assert!(
            after_adam[1] > actual[1],
            "negative grad should increase second param"
        );
        assert!(m.to_vec_f32()?.iter().any(|x| x.abs() > 0.0));
        assert!(v.to_vec_f32()?.iter().any(|x| x.abs() > 0.0));
        Ok(())
    }

    #[test]
    fn cuda_train_tensor_tracks_parameter_metadata_and_detach() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_train_tensor metadata smoke: {err}");
                return Ok(());
            }
        };

        let tensor = Tensor::new(vec![1.0f32], &device)?;
        let param_id = tensor.id();
        let param = CudaTrainTensor::parameter(tensor, param_id)?;
        assert!(param.requires_grad());
        assert_eq!(param.param_id(), Some(param_id));

        let detached = param.detach();
        assert!(!detached.requires_grad());
        assert_eq!(detached.param_id(), None);
        assert!(detached.op_id() > param.op_id());
        assert_eq!(detached.to_vec_f32()?, vec![1.0]);
        Ok(())
    }

    #[test]
    fn cuda_add_last_dim_bias_backward_reduces_leading_dims() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda add-bias smoke: {err}");
                return Ok(());
            }
        };

        let input_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2usize, 3usize),
            &device,
        )?;
        let input_id = input_tensor.id();
        let input = CudaTrainTensor::parameter(input_tensor, input_id)?;
        let bias_tensor = Tensor::from_vec(vec![0.5f32, -0.25, 1.5], (3usize,), &device)?;
        let bias_id = bias_tensor.id();
        let bias = CudaTrainTensor::parameter(bias_tensor, bias_id)?;

        let out = cuda_add_last_dim_bias(&input, &bias)?;
        assert_eq!(out.to_vec_f32()?, vec![1.5, 1.75, 4.5, 4.5, 4.75, 7.5]);
        let loss = cuda_sum_all(&out)?;
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(input_id).context("missing input grad")?.to_vec_f32()?,
            vec![1.0; 6]
        );
        assert_eq!(
            grads.get(bias_id).context("missing bias grad")?.to_vec_f32()?,
            vec![2.0; 3]
        );
        Ok(())
    }

    #[test]
    fn cuda_mul_last_dim_weight_backward_reduces_leading_dims() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda mul-weight smoke: {err}");
                return Ok(());
            }
        };

        let input_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5],
            (2usize, 3usize),
            &device,
        )?;
        let input_id = input_tensor.id();
        let input = CudaTrainTensor::parameter(input_tensor, input_id)?;
        let weight_tensor = Tensor::from_vec(vec![0.5f32, -2.0, 3.0], (3usize,), &device)?;
        let weight_id = weight_tensor.id();
        let weight = CudaTrainTensor::parameter(weight_tensor, weight_id)?;

        let out = cuda_mul_last_dim_weight(&input, &weight)?;
        assert_eq!(out.to_vec_f32()?, vec![0.5, -4.0, 9.0, 2.0, 2.0, 1.5]);
        let loss = cuda_sum_all(&out)?;
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(input_id).context("missing input grad")?.to_vec_f32()?,
            vec![0.5, -2.0, 3.0, 0.5, -2.0, 3.0]
        );
        assert_eq!(
            grads
                .get(weight_id)
                .context("missing weight grad")?
                .to_vec_f32()?,
            vec![5.0, 1.0, 3.5]
        );
        Ok(())
    }

    #[test]
    fn cuda_train_arena_tracks_allocations_and_clear() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_train_arena smoke: {err}");
                return Ok(());
            }
        };

        let mut arena = CudaTrainArena::new(&device)?;
        let a = arena.zeros(&[2, 3], DType::F32)?;
        assert_eq!(a.dims(), &[2, 3]);
        assert_eq!(arena.allocation_count(), 1);
        assert_eq!(arena.allocated_bytes(), 24);

        let b = arena.zeros(&[4], DType::BF16)?;
        assert_eq!(b.dims(), &[4]);
        assert_eq!(arena.allocation_count(), 2);
        assert_eq!(arena.allocated_bytes(), 32);

        arena.clear();
        assert_eq!(arena.allocation_count(), 0);
        assert_eq!(arena.allocated_bytes(), 0);
        Ok(())
    }

    #[derive(Debug)]
    struct FixedGradOp {
        inputs: Vec<CudaTrainTensor>,
        grads: Vec<CudaTrainTensor>,
    }

    impl CudaBackwardOp for FixedGradOp {
        fn op_name(&self) -> &'static str {
            "fixed_grad"
        }

        fn input_refs(&self) -> &[CudaTrainTensor] {
            &self.inputs
        }

        fn backward(&self, _grad_out: &CudaTrainTensor) -> Result<Vec<Option<CudaTrainTensor>>> {
            Ok(self.grads.iter().cloned().map(Some).collect())
        }
    }

    #[test]
    fn cuda_backward_collects_parameter_gradients() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::new(vec![2.0f32, -3.0], &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let grad = CudaTrainTensor::new(Tensor::new(vec![0.5f32, -1.25], &device)?)?;
        let loss = CudaTrainTensor::from_op(
            Tensor::new(vec![1.0f32], &device)?,
            Some(Arc::new(FixedGradOp {
                inputs: vec![param.clone()],
                grads: vec![grad],
            })),
        )?;

        let grads = cuda_backward(&loss)?;
        assert_eq!(grads.len(), 1);
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![0.5, -1.25]
        );
        Ok(())
    }

    #[test]
    fn cuda_backward_accumulates_duplicate_parameter_grads() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_backward accumulation smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::new(vec![2.0f32], &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let grad_a = CudaTrainTensor::new(Tensor::new(vec![0.5f32], &device)?)?;
        let grad_b = CudaTrainTensor::new(Tensor::new(vec![1.25f32], &device)?)?;
        let loss = CudaTrainTensor::from_op(
            Tensor::new(vec![1.0f32], &device)?,
            Some(Arc::new(FixedGradOp {
                inputs: vec![param.clone(), param],
                grads: vec![grad_a, grad_b],
            })),
        )?;

        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![1.75]
        );
        Ok(())
    }

    #[test]
    fn cuda_add_backward_routes_to_both_parameters() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_add backward smoke: {err}");
                return Ok(());
            }
        };

        let lhs_tensor = Tensor::new(vec![2.0f32], &device)?;
        let rhs_tensor = Tensor::new(vec![3.0f32], &device)?;
        let lhs_id = lhs_tensor.id();
        let rhs_id = rhs_tensor.id();
        let lhs = CudaTrainTensor::parameter(lhs_tensor, lhs_id)?;
        let rhs = CudaTrainTensor::parameter(rhs_tensor, rhs_id)?;
        let loss = cuda_add(&lhs, &rhs)?;

        assert_eq!(loss.to_vec_f32()?, vec![5.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(grads.get(lhs_id).expect("lhs grad").to_vec_f32()?, vec![1.0]);
        assert_eq!(grads.get(rhs_id).expect("rhs grad").to_vec_f32()?, vec![1.0]);
        Ok(())
    }

    #[test]
    fn cuda_add_backward_accumulates_shared_parameter() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_add shared-param smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::new(vec![2.0f32], &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let loss = cuda_add(&param, &param)?;

        assert_eq!(loss.to_vec_f32()?, vec![4.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![2.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_sub_backward_negates_rhs_grad() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_sub backward smoke: {err}");
                return Ok(());
            }
        };

        let lhs_tensor = Tensor::new(vec![5.0f32, 7.0], &device)?;
        let rhs_tensor = Tensor::new(vec![2.0f32, 11.0], &device)?;
        let lhs_id = lhs_tensor.id();
        let rhs_id = rhs_tensor.id();
        let lhs = CudaTrainTensor::parameter(lhs_tensor, lhs_id)?;
        let rhs = CudaTrainTensor::parameter(rhs_tensor, rhs_id)?;
        let diff = cuda_sub(&lhs, &rhs)?;
        let loss = cuda_sum_all(&diff)?;

        assert_eq!(diff.to_vec_f32()?, vec![3.0, -4.0]);
        assert_eq!(loss.to_vec_f32()?, vec![-1.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(lhs_id).expect("lhs grad").to_vec_f32()?,
            vec![1.0, 1.0]
        );
        assert_eq!(
            grads.get(rhs_id).expect("rhs grad").to_vec_f32()?,
            vec![-1.0, -1.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_mul_backward_uses_saved_inputs() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_mul backward smoke: {err}");
                return Ok(());
            }
        };

        let lhs_tensor = Tensor::new(vec![2.0f32], &device)?;
        let rhs_tensor = Tensor::new(vec![3.0f32], &device)?;
        let lhs_id = lhs_tensor.id();
        let rhs_id = rhs_tensor.id();
        let lhs = CudaTrainTensor::parameter(lhs_tensor, lhs_id)?;
        let rhs = CudaTrainTensor::parameter(rhs_tensor, rhs_id)?;
        let loss = cuda_mul(&lhs, &rhs)?;

        assert_eq!(loss.to_vec_f32()?, vec![6.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(grads.get(lhs_id).expect("lhs grad").to_vec_f32()?, vec![3.0]);
        assert_eq!(grads.get(rhs_id).expect("rhs grad").to_vec_f32()?, vec![2.0]);
        Ok(())
    }

    #[test]
    fn cuda_mul_backward_accumulates_shared_parameter() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_mul shared-param smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::new(vec![4.0f32], &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let loss = cuda_mul(&param, &param)?;

        assert_eq!(loss.to_vec_f32()?, vec![16.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![8.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_mul_last_dim_broadcast_backward_reduces_last_dim() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!(
                    "CUDA unavailable, skipping cuda_mul_last_dim_broadcast smoke: {err}"
                );
                return Ok(());
            }
        };

        let input_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2usize, 3usize),
            &device,
        )?;
        let scalar_tensor = Tensor::from_vec(vec![10.0f32, 20.0], (2usize, 1usize), &device)?;
        let input_id = input_tensor.id();
        let scalar_id = scalar_tensor.id();
        let input = CudaTrainTensor::parameter(input_tensor, input_id)?;
        let scalar = CudaTrainTensor::parameter(scalar_tensor, scalar_id)?;
        let product = cuda_mul_last_dim_broadcast(&input, &scalar)?;
        let loss = cuda_sum_all(&product)?;

        assert_eq!(product.dims(), &[2, 3]);
        assert_eq!(
            product.to_vec_f32()?,
            vec![10.0, 20.0, 30.0, 80.0, 100.0, 120.0]
        );
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(input_id).expect("input grad").to_vec_f32()?,
            vec![10.0, 10.0, 10.0, 20.0, 20.0, 20.0]
        );
        assert_eq!(
            grads.get(scalar_id).expect("scalar grad").to_vec_f32()?,
            vec![6.0, 15.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_div_backward_uses_quotient_rule() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_div backward smoke: {err}");
                return Ok(());
            }
        };

        let lhs_tensor = Tensor::new(vec![6.0f32, 8.0], &device)?;
        let rhs_tensor = Tensor::new(vec![2.0f32, 4.0], &device)?;
        let lhs_id = lhs_tensor.id();
        let rhs_id = rhs_tensor.id();
        let lhs = CudaTrainTensor::parameter(lhs_tensor, lhs_id)?;
        let rhs = CudaTrainTensor::parameter(rhs_tensor, rhs_id)?;
        let quotient = cuda_div(&lhs, &rhs)?;
        let loss = cuda_sum_all(&quotient)?;

        assert_eq!(quotient.to_vec_f32()?, vec![3.0, 2.0]);
        assert_eq!(loss.to_vec_f32()?, vec![5.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(lhs_id).expect("lhs grad").to_vec_f32()?,
            vec![0.5, 0.25]
        );
        assert_eq!(
            grads.get(rhs_id).expect("rhs grad").to_vec_f32()?,
            vec![-1.5, -0.5]
        );
        Ok(())
    }

    #[test]
    fn cuda_scale_backward_scales_grad() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_scale backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::new(vec![1.0f32, -3.0], &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let scaled = cuda_scale(&param, 2.5)?;
        let loss = cuda_sum_all(&scaled)?;

        assert_eq!(scaled.to_vec_f32()?, vec![2.5, -7.5]);
        assert_eq!(loss.to_vec_f32()?, vec![-5.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![2.5, 2.5]
        );
        Ok(())
    }

    #[test]
    fn cuda_causal_mask_applies_future_token_bias() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_causal_mask smoke: {err}");
                return Ok(());
            }
        };

        let scores_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            (1usize, 3usize, 3usize),
            &device,
        )?;
        let scores_id = scores_tensor.id();
        let scores = CudaTrainTensor::parameter(scores_tensor, scores_id)?;
        let masked = cuda_causal_mask(&scores, 0)?;
        let loss = cuda_sum_all(&masked)?;

        let values = masked.to_vec_f32()?;
        assert_eq!(values[0], 1.0);
        assert!(values[1] < -1.0e20);
        assert!(values[2] < -1.0e20);
        assert_eq!(values[3], 4.0);
        assert_eq!(values[4], 5.0);
        assert!(values[5] < -1.0e20);
        assert_eq!(values[6], 7.0);
        assert_eq!(values[7], 8.0);
        assert_eq!(values[8], 9.0);

        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(scores_id).expect("scores grad").to_vec_f32()?,
            vec![1.0; 9]
        );
        Ok(())
    }

    #[test]
    fn cuda_rmsnorm_backward_matches_analytic_dx() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_rmsnorm backward smoke: {err}");
                return Ok(());
            }
        };

        let input_tensor = Tensor::from_vec(
            vec![3.0f32, 4.0, 1.0, -2.0],
            (2usize, 2usize),
            &device,
        )?;
        let weight_tensor = Tensor::from_vec(vec![0.0f32, 0.5], (2usize,), &device)?;
        let input_id = input_tensor.id();
        let weight_id = weight_tensor.id();
        let input = CudaTrainTensor::parameter(input_tensor, input_id)?;
        let weight = CudaTrainTensor::parameter(weight_tensor, weight_id)?;
        let normed = cuda_rmsnorm(&input, &weight, 0.0)?;
        let loss = cuda_sum_all(&normed)?;

        let expected_out = vec![
            0.84852815f32,
            1.6970563,
            0.6324555,
            -1.8973665,
        ];
        let actual_out = normed.to_vec_f32()?;
        for (idx, (actual, expected)) in actual_out.iter().zip(expected_out.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "rmsnorm output mismatch at {idx}: actual={actual} expected={expected}"
            );
        }

        let grads = cuda_backward(&loss)?;
        let expected_grad = vec![-0.02262742f32, 0.01697056, 0.8854377, 0.44271886];
        let actual_grad = grads.get(input_id).expect("input grad").to_vec_f32()?;
        for (idx, (actual, expected)) in actual_grad.iter().zip(expected_grad.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "rmsnorm grad mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
        assert!(
            grads.get(weight_id).is_none(),
            "cuda_rmsnorm should keep frozen base weight gradient empty"
        );
        Ok(())
    }

    #[test]
    fn cuda_rope_backward_applies_inverse_rotation() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_rope backward smoke: {err}");
                return Ok(());
            }
        };

        let input_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 9.0, 10.0, 3.0, 4.0, 11.0, 12.0],
            (2usize, 1usize, 4usize),
            &device,
        )?;
        let input_id = input_tensor.id();
        let input = CudaTrainTensor::parameter(input_tensor, input_id)?;
        let cos = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 0.0],
            (2usize, 1usize),
            &device,
        )?)?;
        let sin = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 1.0],
            (2usize, 1usize),
            &device,
        )?)?;
        let rotated = cuda_rope(&input, &cos, &sin, 2)?;
        let loss = cuda_sum_all(&rotated)?;

        assert_eq!(rotated.dims(), &[2, 1, 4]);
        assert_eq!(
            rotated.to_vec_f32()?,
            vec![1.0, 2.0, 9.0, 10.0, -4.0, 3.0, 11.0, 12.0]
        );
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(input_id).expect("input grad").to_vec_f32()?,
            vec![1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_to_dtype_backward_passthrough_restores_input_dtype() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_to_dtype backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::new(vec![1.0f32, 2.0], &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let bf16 = cuda_to_dtype(&param, DType::BF16)?;
        let f32_again = cuda_to_dtype(&bf16, DType::F32)?;
        let loss = cuda_sum_all(&f32_again)?;

        assert_eq!(bf16.dtype(), DType::BF16);
        assert_eq!(f32_again.dtype(), DType::F32);
        assert_eq!(loss.to_vec_f32()?, vec![3.0]);
        let grads = cuda_backward(&loss)?;
        let grad = grads.get(param_id).expect("param grad");
        assert_eq!(grad.dtype(), DType::F32);
        assert_eq!(grad.to_vec_f32()?, vec![1.0, 1.0]);
        Ok(())
    }

    #[test]
    fn cuda_sum_all_backward_broadcasts_scalar_grad() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_sum_all backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::new(vec![1.0f32, 2.0, 3.0], &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let loss = cuda_sum_all(&param)?;

        assert_eq!(loss.to_vec_f32()?, vec![6.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![1.0, 1.0, 1.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_mean_all_backward_scales_scalar_grad() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_mean_all backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::new(vec![2.0f32, 4.0, 6.0, 8.0], &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let loss = cuda_mean_all(&param)?;

        assert_eq!(loss.to_vec_f32()?, vec![5.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![0.25, 0.25, 0.25, 0.25]
        );
        Ok(())
    }

    #[test]
    fn cuda_reshape_backward_restores_input_shape() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_reshape backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2usize, 3usize),
            &device,
        )?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let reshaped = cuda_reshape(&param, &[3, 2])?;
        let weights = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0],
            (3usize, 2usize),
            &device,
        )?)?;
        let weighted = cuda_mul(&reshaped, &weights)?;
        let loss = cuda_sum_all(&weighted)?;

        assert_eq!(reshaped.dims(), &[3, 2]);
        assert_eq!(loss.to_vec_f32()?, vec![302.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_transpose2d_backward_transposes_grad() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_transpose2d backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2usize, 3usize),
            &device,
        )?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let transposed = cuda_transpose2d(&param)?;
        let weights = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0],
            (3usize, 2usize),
            &device,
        )?)?;
        let weighted = cuda_mul(&transposed, &weights)?;
        let loss = cuda_sum_all(&weighted)?;

        assert_eq!(transposed.dims(), &[3, 2]);
        assert_eq!(transposed.to_vec_f32()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(loss.to_vec_f32()?, vec![334.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_transpose_last_two_backward_transposes_batched_grad() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_transpose_last_two backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 1.0, 2.0, 3.0, -2.0, 4.0],
            (2usize, 2usize, 3usize),
            &device,
        )?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let transposed = cuda_transpose_last_two(&param)?;
        let weights = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0],
            (2usize, 3usize, 2usize),
            &device,
        )?)?;
        let weighted = cuda_mul(&transposed, &weights)?;
        let loss = cuda_sum_all(&weighted)?;

        assert_eq!(transposed.dims(), &[2, 3, 2]);
        assert_eq!(
            transposed.to_vec_f32()?,
            vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, -1.0, 3.0, 1.0, -2.0, 2.0, 4.0]
        );
        assert_eq!(loss.to_vec_f32()?, vec![358.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_sum_of_square_graph_matches_expected_grad() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_sum square graph smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::new(vec![1.0f32, 2.0, 3.0], &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let squared = cuda_mul(&param, &param)?;
        let loss = cuda_sum_all(&squared)?;

        assert_eq!(loss.to_vec_f32()?, vec![14.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![2.0, 4.0, 6.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_matmul_sum_backward_matches_expected_grads() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_matmul backward smoke: {err}");
                return Ok(());
            }
        };

        let lhs_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2usize, 3usize),
            &device,
        )?;
        let rhs_tensor = Tensor::from_vec(
            vec![5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0],
            (3usize, 2usize),
            &device,
        )?;
        let lhs_id = lhs_tensor.id();
        let rhs_id = rhs_tensor.id();
        let lhs = CudaTrainTensor::parameter(lhs_tensor, lhs_id)?;
        let rhs = CudaTrainTensor::parameter(rhs_tensor, rhs_id)?;
        let product = cuda_matmul(&lhs, &rhs)?;
        let loss = cuda_sum_all(&product)?;

        assert_eq!(product.to_vec_f32()?, vec![46.0, 52.0, 109.0, 124.0]);
        assert_eq!(loss.to_vec_f32()?, vec![331.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(lhs_id).expect("lhs grad").to_vec_f32()?,
            vec![11.0, 15.0, 19.0, 11.0, 15.0, 19.0]
        );
        assert_eq!(
            grads.get(rhs_id).expect("rhs grad").to_vec_f32()?,
            vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_batched_matmul_sum_backward_matches_expected_grads() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_batched_matmul backward smoke: {err}");
                return Ok(());
            }
        };

        let lhs_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 1.0, 2.0, 3.0, -2.0, 4.0],
            (2usize, 2usize, 3usize),
            &device,
        )?;
        let rhs_tensor = Tensor::from_vec(
            vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0],
            (2usize, 3usize, 2usize),
            &device,
        )?;
        let lhs_id = lhs_tensor.id();
        let rhs_id = rhs_tensor.id();
        let lhs = CudaTrainTensor::parameter(lhs_tensor, lhs_id)?;
        let rhs = CudaTrainTensor::parameter(rhs_tensor, rhs_id)?;
        let product = cuda_batched_matmul(&lhs, &rhs)?;
        let loss = cuda_sum_all(&product)?;

        assert_eq!(product.dims(), &[2, 2, 2]);
        assert_eq!(
            product.to_vec_f32()?,
            vec![14.0, 140.0, 32.0, 320.0, 13.0, 7.0, 26.0, 11.0]
        );
        assert_eq!(loss.to_vec_f32()?, vec![563.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(lhs_id).expect("lhs grad").to_vec_f32()?,
            vec![11.0, 22.0, 33.0, 11.0, 22.0, 33.0, 5.0, 7.0, 9.0, 5.0, 7.0, 9.0]
        );
        assert_eq!(
            grads.get(rhs_id).expect("rhs grad").to_vec_f32()?,
            vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0, 2.0, 2.0, -1.0, -1.0, 6.0, 6.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_softmax_last_dim_backward_matches_cpu_reference() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_softmax_last_dim backward smoke: {err}");
                return Ok(());
            }
        };

        let data = vec![0.0f32, 1.0, 2.0, -1.0, 0.5, 3.0];
        let param_tensor = Tensor::from_vec(data.clone(), (2usize, 3usize), &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let probs = cuda_softmax_last_dim(&param)?;
        let squared = cuda_mul(&probs, &probs)?;
        let loss = cuda_sum_all(&squared)?;

        let mut expected_probs = Vec::new();
        let mut expected_grad = Vec::new();
        for row in data.chunks(3) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = row.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            let probs: Vec<f32> = exp.iter().map(|v| v / sum).collect();
            let dot: f32 = probs.iter().map(|p| p * (2.0 * p)).sum();
            expected_probs.extend(probs.iter().copied());
            expected_grad.extend(probs.iter().map(|p| p * (2.0 * p - dot)));
        }

        let actual_probs = probs.to_vec_f32()?;
        for (idx, (actual, expected)) in actual_probs.iter().zip(expected_probs.iter()).enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-5,
                "softmax output mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
        let expected_loss: f32 = expected_probs.iter().map(|p| p * p).sum();
        assert!((loss.to_vec_f32()?[0] - expected_loss).abs() < 1e-5);

        let grads = cuda_backward(&loss)?;
        let actual_grad = grads.get(param_id).expect("param grad").to_vec_f32()?;
        for (idx, (actual, expected)) in actual_grad.iter().zip(expected_grad.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "softmax grad mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_sigmoid_backward_matches_cpu_reference() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_sigmoid backward smoke: {err}");
                return Ok(());
            }
        };

        let data = vec![-2.0f32, 0.0, 2.0];
        let param_tensor = Tensor::new(data.clone(), &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let sigmoid = cuda_sigmoid(&param)?;
        let loss = cuda_sum_all(&sigmoid)?;

        let mut expected_out = Vec::new();
        let mut expected_grad = Vec::new();
        for x in data {
            let y = 1.0 / (1.0 + (-x).exp());
            expected_out.push(y);
            expected_grad.push(y * (1.0 - y));
        }

        let actual_out = sigmoid.to_vec_f32()?;
        for (idx, (actual, expected)) in actual_out.iter().zip(expected_out.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "sigmoid output mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
        let expected_loss: f32 = expected_out.iter().sum();
        assert!((loss.to_vec_f32()?[0] - expected_loss).abs() < 1e-5);

        let grads = cuda_backward(&loss)?;
        let actual_grad = grads.get(param_id).expect("param grad").to_vec_f32()?;
        for (idx, (actual, expected)) in actual_grad.iter().zip(expected_grad.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "sigmoid grad mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_exp_and_softplus_backward_match_cpu_reference() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda exp/softplus smoke: {err}");
                return Ok(());
            }
        };

        let data = vec![-1.0f32, 0.0, 1.5];
        let exp_tensor = Tensor::new(data.clone(), &device)?;
        let exp_id = exp_tensor.id();
        let exp_param = CudaTrainTensor::parameter(exp_tensor, exp_id)?;
        let exp_out = cuda_exp(&exp_param)?;
        let exp_loss = cuda_sum_all(&exp_out)?;
        let exp_grads = cuda_backward(&exp_loss)?;

        let expected_exp: Vec<f32> = data.iter().map(|x| x.exp()).collect();
        for (idx, (actual, expected)) in exp_out
            .to_vec_f32()?
            .iter()
            .zip(expected_exp.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-5,
                "exp output mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
        for (idx, (actual, expected)) in exp_grads
            .get(exp_id)
            .expect("exp grad")
            .to_vec_f32()?
            .iter()
            .zip(expected_exp.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-5,
                "exp grad mismatch at {idx}: actual={actual} expected={expected}"
            );
        }

        let softplus_tensor = Tensor::new(data.clone(), &device)?;
        let softplus_id = softplus_tensor.id();
        let softplus_param = CudaTrainTensor::parameter(softplus_tensor, softplus_id)?;
        let softplus_out = cuda_softplus(&softplus_param)?;
        let softplus_loss = cuda_sum_all(&softplus_out)?;
        let softplus_grads = cuda_backward(&softplus_loss)?;

        let expected_softplus: Vec<f32> = data.iter().map(|x| (1.0 + x.exp()).ln()).collect();
        let expected_softplus_grad: Vec<f32> =
            data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        for (idx, (actual, expected)) in softplus_out
            .to_vec_f32()?
            .iter()
            .zip(expected_softplus.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-5,
                "softplus output mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
        for (idx, (actual, expected)) in softplus_grads
            .get(softplus_id)
            .expect("softplus grad")
            .to_vec_f32()?
            .iter()
            .zip(expected_softplus_grad.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-5,
                "softplus grad mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_silu_backward_matches_cpu_reference() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_silu backward smoke: {err}");
                return Ok(());
            }
        };

        let data = vec![-1.0f32, 0.0, 2.0];
        let param_tensor = Tensor::new(data.clone(), &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let silu = cuda_silu(&param)?;
        let loss = cuda_sum_all(&silu)?;

        let mut expected_out = Vec::new();
        let mut expected_grad = Vec::new();
        for x in data {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            expected_out.push(x * sigmoid);
            expected_grad.push(sigmoid * (1.0 + x * (1.0 - sigmoid)));
        }

        let actual_out = silu.to_vec_f32()?;
        for (idx, (actual, expected)) in actual_out.iter().zip(expected_out.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "silu output mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
        let expected_loss: f32 = expected_out.iter().sum();
        assert!((loss.to_vec_f32()?[0] - expected_loss).abs() < 1e-5);

        let grads = cuda_backward(&loss)?;
        let actual_grad = grads.get(param_id).expect("param grad").to_vec_f32()?;
        for (idx, (actual, expected)) in actual_grad.iter().zip(expected_grad.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-5,
                "silu grad mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_swiglu_mlp_backward_reaches_input_and_weights() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_swiglu_mlp backward smoke: {err}");
                return Ok(());
            }
        };

        let input_tensor = Tensor::from_vec(
            vec![0.5f32, -1.0, 1.5, 0.25],
            (2usize, 2usize),
            &device,
        )?;
        let gate_tensor = Tensor::from_vec(
            vec![0.2f32, -0.3, 0.4, 0.1],
            (2usize, 2usize),
            &device,
        )?;
        let up_tensor = Tensor::from_vec(
            vec![0.7f32, -0.2, -0.5, 0.6],
            (2usize, 2usize),
            &device,
        )?;
        let down_tensor = Tensor::from_vec(
            vec![0.3f32, -0.4, 0.8, 0.2],
            (2usize, 2usize),
            &device,
        )?;
        let input_id = input_tensor.id();
        let gate_id = gate_tensor.id();
        let up_id = up_tensor.id();
        let down_id = down_tensor.id();
        let input = CudaTrainTensor::parameter(input_tensor, input_id)?;
        let gate = CudaTrainTensor::parameter(gate_tensor, gate_id)?;
        let up = CudaTrainTensor::parameter(up_tensor, up_id)?;
        let down = CudaTrainTensor::parameter(down_tensor, down_id)?;

        let out = cuda_swiglu_mlp(&input, &gate, &up, &down)?;
        let loss = cuda_sum_all(&out)?;

        assert_eq!(out.dims(), &[2, 2]);
        let out_values = out.to_vec_f32()?;
        assert!(
            out_values.iter().all(|value| value.is_finite()),
            "SwiGLU output should stay finite: {out_values:?}"
        );
        let grads = cuda_backward(&loss)?;
        for (name, id) in [
            ("input", input_id),
            ("gate", gate_id),
            ("up", up_id),
            ("down", down_id),
        ] {
            let values = grads
                .get(id)
                .with_context(|| format!("missing {name} grad"))?
                .to_vec_f32()?;
            assert!(
                values.iter().all(|value| value.is_finite())
                    && values.iter().any(|value| value.abs() > 1e-6),
                "{name} grad should be finite and non-zero: {values:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_full_attention_layer_backward_reaches_core_weights() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_full_attention_layer backward smoke: {err}");
                return Ok(());
            }
        };

        let input_tensor = Tensor::from_vec(
            vec![0.5f32, -1.0, 1.5, 0.25],
            (2usize, 2usize),
            &device,
        )?;
        let input_norm_tensor = Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?;
        let q_tensor = Tensor::from_vec(
            vec![0.2f32, -0.3, 0.05, 0.4, 0.4, 0.1, -0.2, 0.3],
            (2usize, 4usize),
            &device,
        )?;
        let k_tensor = Tensor::from_vec(
            vec![0.1f32, 0.6, 0.8, -0.2],
            (2usize, 2usize),
            &device,
        )?;
        let v_tensor = Tensor::from_vec(
            vec![0.7f32, -0.2, -0.5, 0.6],
            (2usize, 2usize),
            &device,
        )?;
        let q_norm_tensor = Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?;
        let k_norm_tensor = Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?;
        let o_tensor = Tensor::from_vec(
            vec![0.3f32, -0.4, 0.8, 0.2],
            (2usize, 2usize),
            &device,
        )?;
        let post_norm_tensor = Tensor::from_vec(vec![0.0f32, 0.0], (2usize,), &device)?;
        let gate_tensor = Tensor::from_vec(
            vec![0.25f32, -0.15, 0.35, 0.05],
            (2usize, 2usize),
            &device,
        )?;
        let up_tensor = Tensor::from_vec(
            vec![0.45f32, 0.2, -0.1, 0.55],
            (2usize, 2usize),
            &device,
        )?;
        let down_tensor = Tensor::from_vec(
            vec![0.6f32, -0.25, 0.15, 0.5],
            (2usize, 2usize),
            &device,
        )?;
        let cos = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 0.0],
            (2usize, 1usize),
            &device,
        )?)?;
        let sin = CudaTrainTensor::new(Tensor::from_vec(
            vec![0.0f32, 1.0],
            (2usize, 1usize),
            &device,
        )?)?;

        let input_id = input_tensor.id();
        let input_norm_id = input_norm_tensor.id();
        let q_id = q_tensor.id();
        let k_id = k_tensor.id();
        let v_id = v_tensor.id();
        let q_norm_id = q_norm_tensor.id();
        let k_norm_id = k_norm_tensor.id();
        let o_id = o_tensor.id();
        let post_norm_id = post_norm_tensor.id();
        let gate_id = gate_tensor.id();
        let up_id = up_tensor.id();
        let down_id = down_tensor.id();
        let input = CudaTrainTensor::parameter(input_tensor, input_id)?;
        let input_norm = CudaTrainTensor::parameter(input_norm_tensor, input_norm_id)?;
        let q_weight = CudaTrainTensor::parameter(q_tensor, q_id)?;
        let k_weight = CudaTrainTensor::parameter(k_tensor, k_id)?;
        let v_weight = CudaTrainTensor::parameter(v_tensor, v_id)?;
        let q_norm = CudaTrainTensor::parameter(q_norm_tensor, q_norm_id)?;
        let k_norm = CudaTrainTensor::parameter(k_norm_tensor, k_norm_id)?;
        let o_weight = CudaTrainTensor::parameter(o_tensor, o_id)?;
        let post_norm = CudaTrainTensor::parameter(post_norm_tensor, post_norm_id)?;
        let gate_weight = CudaTrainTensor::parameter(gate_tensor, gate_id)?;
        let up_weight = CudaTrainTensor::parameter(up_tensor, up_id)?;
        let down_weight = CudaTrainTensor::parameter(down_tensor, down_id)?;
        let weights = CudaFullAttentionLayer {
            input_norm_weight: &input_norm,
            q_weight: &q_weight,
            k_weight: &k_weight,
            v_weight: &v_weight,
            q_norm_weight: Some(&q_norm),
            k_norm_weight: Some(&k_norm),
            o_weight: &o_weight,
            post_norm_weight: &post_norm,
            gate_weight: &gate_weight,
            up_weight: &up_weight,
            down_weight: &down_weight,
            heads_q: 1,
            heads_kv: 1,
            head_dim: 2,
            eps: 1e-6,
            attn_output_gate: true,
            rope: Some(CudaRopeTables {
                cos: &cos,
                sin: &sin,
                rotary_dim: 2,
            }),
        };

        let out = cuda_full_attention_layer(&input, &weights)?;
        let loss = cuda_sum_all(&out)?;

        assert_eq!(out.dims(), &[2, 2]);
        let out_values = out.to_vec_f32()?;
        assert!(
            out_values.iter().all(|value| value.is_finite()),
            "full attention layer output should stay finite: {out_values:?}"
        );
        let grads = cuda_backward(&loss)?;
        for (name, id) in [
            ("input", input_id),
            ("q", q_id),
            ("k", k_id),
            ("v", v_id),
            ("o", o_id),
            ("gate", gate_id),
            ("up", up_id),
            ("down", down_id),
        ] {
            let values = grads
                .get(id)
                .with_context(|| format!("missing {name} grad"))?
                .to_vec_f32()?;
            assert!(
                values.iter().all(|value| value.is_finite())
                    && values.iter().any(|value| value.abs() > 1e-7),
                "{name} grad should be finite and non-zero: {values:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_repeat_kv_heads_backward_sums_repeated_groups() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_repeat_kv_heads backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (2usize, 1usize, 2usize), &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let repeated = cuda_repeat_kv_heads(&param, 2)?;
        let weights = CudaTrainTensor::new(Tensor::from_vec(
            vec![10.0f32, 1.0, 20.0, 2.0, 30.0, 3.0, 40.0, 4.0],
            (4usize, 1usize, 2usize),
            &device,
        )?)?;
        let weighted = cuda_mul(&repeated, &weights)?;
        let loss = cuda_sum_all(&weighted)?;

        assert_eq!(repeated.dims(), &[4, 1, 2]);
        assert_eq!(repeated.to_vec_f32()?, vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
        assert_eq!(loss.to_vec_f32()?, vec![274.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![30.0, 3.0, 70.0, 7.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_narrow_last_dim_backward_pads_zero_gradients() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_narrow_last_dim backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (2usize, 4usize),
            &device,
        )?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let narrowed = cuda_narrow_last_dim(&param, 1, 2)?;
        let weights = CudaTrainTensor::new(Tensor::from_vec(
            vec![10.0f32, 20.0, 30.0, 40.0],
            (2usize, 2usize),
            &device,
        )?)?;
        let weighted = cuda_mul(&narrowed, &weights)?;
        let loss = cuda_sum_all(&weighted)?;

        assert_eq!(narrowed.dims(), &[2, 2]);
        assert_eq!(narrowed.to_vec_f32()?, vec![2.0, 3.0, 6.0, 7.0]);
        assert_eq!(loss.to_vec_f32()?, vec![540.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![0.0, 10.0, 20.0, 0.0, 0.0, 30.0, 40.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_narrow_rows_backward_pads_zero_gradients() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_narrow_rows backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (4usize, 2usize),
            &device,
        )?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let narrowed = cuda_narrow_rows(&param, 1, 2)?;
        let weights = CudaTrainTensor::new(Tensor::from_vec(
            vec![10.0f32, 20.0, 30.0, 40.0],
            (2usize, 2usize),
            &device,
        )?)?;
        let weighted = cuda_mul(&narrowed, &weights)?;
        let loss = cuda_sum_all(&weighted)?;

        assert_eq!(narrowed.dims(), &[2, 2]);
        assert_eq!(narrowed.to_vec_f32()?, vec![3.0, 4.0, 5.0, 6.0]);
        assert_eq!(loss.to_vec_f32()?, vec![500.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![0.0, 0.0, 10.0, 20.0, 30.0, 40.0, 0.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_cat_rows_backward_splits_gradients() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_cat_rows backward smoke: {err}");
                return Ok(());
            }
        };

        let first_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0],
            (2usize, 2usize),
            &device,
        )?;
        let first_id = first_tensor.id();
        let first = CudaTrainTensor::parameter(first_tensor, first_id)?;
        let second_tensor =
            Tensor::from_vec(vec![5.0f32, 6.0, 7.0, 8.0], (2usize, 2usize), &device)?;
        let second_id = second_tensor.id();
        let second = CudaTrainTensor::parameter(second_tensor, second_id)?;
        let cat = cuda_cat_rows(&[first, second])?;
        let weights = CudaTrainTensor::new(Tensor::from_vec(
            vec![1.0f32, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
            (4usize, 2usize),
            &device,
        )?)?;
        let weighted = cuda_mul(&cat, &weights)?;
        let loss = cuda_sum_all(&weighted)?;

        assert_eq!(cat.dims(), &[4, 2]);
        assert_eq!(
            cat.to_vec_f32()?,
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
        assert_eq!(loss.to_vec_f32()?, vec![650.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(first_id).context("missing first grad")?.to_vec_f32()?,
            vec![1.0, 10.0, 2.0, 20.0]
        );
        assert_eq!(
            grads
                .get(second_id)
                .context("missing second grad")?
                .to_vec_f32()?,
            vec![3.0, 30.0, 4.0, 40.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_causal_depthwise_conv1d_prefill_zero_state_matches_reference() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda conv1d prefill smoke: {err}");
                return Ok(());
            }
        };

        let input_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (3usize, 2usize),
            &device,
        )?;
        let input_id = input_tensor.id();
        let input = CudaTrainTensor::parameter(input_tensor, input_id)?;
        let weight_tensor = Tensor::from_vec(
            vec![1.0f32, 10.0, 100.0, 2.0, 20.0, 200.0],
            (2usize, 3usize),
            &device,
        )?;
        let weight_id = weight_tensor.id();
        let weight = CudaTrainTensor::parameter(weight_tensor, weight_id)?;
        let output = cuda_causal_depthwise_conv1d_prefill_zero_state(&input, &weight)?;
        let loss = cuda_sum_all(&output)?;

        assert_eq!(output.dims(), &[3, 2]);
        assert_eq!(
            output.to_vec_f32()?,
            vec![100.0, 400.0, 310.0, 840.0, 531.0, 1284.0]
        );
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(input_id).context("missing input grad")?.to_vec_f32()?,
            vec![111.0, 222.0, 110.0, 220.0, 100.0, 200.0]
        );
        assert_eq!(
            grads
                .get(weight_id)
                .context("missing weight grad")?
                .to_vec_f32()?,
            vec![1.0, 4.0, 9.0, 2.0, 6.0, 12.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_causal_depthwise_conv1d_next_state_zero_state_tracks_tail() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda conv1d state smoke: {err}");
                return Ok(());
            }
        };

        let input_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (3usize, 2usize),
            &device,
        )?;
        let input_id = input_tensor.id();
        let input = CudaTrainTensor::parameter(input_tensor, input_id)?;
        let state = cuda_causal_depthwise_conv1d_next_state_zero_state(&input, 3)?;
        assert_eq!(state.dims(), &[2, 2]);
        assert_eq!(state.to_vec_f32()?, vec![3.0, 4.0, 5.0, 6.0]);

        let short_tensor = Tensor::from_vec(vec![7.0f32, 8.0], (1usize, 2usize), &device)?;
        let short = CudaTrainTensor::new(short_tensor)?;
        let short_state = cuda_causal_depthwise_conv1d_next_state_zero_state(&short, 4)?;
        assert_eq!(short_state.dims(), &[3, 2]);
        assert_eq!(short_state.to_vec_f32()?, vec![0.0, 0.0, 0.0, 0.0, 7.0, 8.0]);

        let loss = cuda_sum_all(&state)?;
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(input_id).context("missing input grad")?.to_vec_f32()?,
            vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_causal_depthwise_conv1d_prefill_with_state_threads_state() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda conv1d stateful smoke: {err}");
                return Ok(());
            }
        };

        let input_tensor =
            Tensor::from_vec(vec![3.0f32, 4.0, 5.0, 6.0], (2usize, 2usize), &device)?;
        let input_id = input_tensor.id();
        let input = CudaTrainTensor::parameter(input_tensor, input_id)?;
        let weight_tensor = Tensor::from_vec(
            vec![1.0f32, 10.0, 100.0, 2.0, 20.0, 200.0],
            (2usize, 3usize),
            &device,
        )?;
        let weight_id = weight_tensor.id();
        let weight = CudaTrainTensor::parameter(weight_tensor, weight_id)?;
        let state_tensor =
            Tensor::from_vec(vec![0.5f32, 1.0, 1.5, 2.0], (2usize, 2usize), &device)?;
        let state_id = state_tensor.id();
        let state = CudaTrainTensor::parameter(state_tensor, state_id)?;

        let (output, next_state) =
            cuda_causal_depthwise_conv1d_prefill_with_state(&input, &weight, &state)?;
        let loss = cuda_sum_all(&output)?;

        assert_eq!(output.dims(), &[2, 2]);
        assert_eq!(output.to_vec_f32()?, vec![315.5, 842.0, 531.5, 1284.0]);
        assert_eq!(next_state.dims(), &[2, 2]);
        assert_eq!(next_state.to_vec_f32()?, vec![3.0, 4.0, 5.0, 6.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(input_id).context("missing input grad")?.to_vec_f32()?,
            vec![110.0, 220.0, 100.0, 200.0]
        );
        assert_eq!(
            grads
                .get(weight_id)
                .context("missing weight grad")?
                .to_vec_f32()?,
            vec![2.0, 4.5, 8.0, 3.0, 6.0, 10.0]
        );
        assert_eq!(
            grads.get(state_id).context("missing state grad")?.to_vec_f32()?,
            vec![1.0, 2.0, 11.0, 22.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_gdn_single_token_recurrence_matches_reference() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda GDN single-token smoke: {err}");
                return Ok(());
            }
        };

        let q_tensor = Tensor::from_vec(vec![0.5f32, -1.0], (1usize, 2usize), &device)?;
        let k_tensor = Tensor::from_vec(vec![1.5f32, 0.25], (1usize, 2usize), &device)?;
        let v_tensor =
            Tensor::from_vec(vec![0.1f32, -0.2, 0.3], (1usize, 3usize), &device)?;
        let beta_tensor = Tensor::from_vec(vec![0.4f32], (1usize, 1usize), &device)?;
        let g_tensor = Tensor::from_vec(vec![0.0f32], (1usize, 1usize), &device)?;
        let state_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2usize, 3usize),
            &device,
        )?;
        let q_id = q_tensor.id();
        let k_id = k_tensor.id();
        let v_id = v_tensor.id();
        let beta_id = beta_tensor.id();
        let g_id = g_tensor.id();
        let state_id = state_tensor.id();
        let q = CudaTrainTensor::parameter(q_tensor, q_id)?;
        let k = CudaTrainTensor::parameter(k_tensor, k_id)?;
        let v = CudaTrainTensor::parameter(v_tensor, v_id)?;
        let beta = CudaTrainTensor::parameter(beta_tensor, beta_id)?;
        let g = CudaTrainTensor::parameter(g_tensor, g_id)?;
        let state = CudaTrainTensor::parameter(state_tensor, state_id)?;

        let result = cuda_gdn_single_token_recurrence(&q, &k, &v, &beta, &g, &state)?;
        assert_eq!(result.out.dims(), &[1, 3]);
        assert_eq!(result.next_state.dims(), &[2, 3]);
        let expected_out = [-3.98f32, -4.89, -5.64];
        for (idx, (actual, expected)) in result
            .out
            .to_vec_f32()?
            .iter()
            .zip(expected_out.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-5,
                "GDN single-token out mismatch at {idx}: actual={actual} expected={expected}"
            );
        }
        let expected_state = [-0.44f32, -0.67, -0.42, 3.76, 4.555, 5.43];
        for (idx, (actual, expected)) in result
            .next_state
            .to_vec_f32()?
            .iter()
            .zip(expected_state.iter())
            .enumerate()
        {
            assert!(
                (actual - expected).abs() < 1e-5,
                "GDN single-token state mismatch at {idx}: actual={actual} expected={expected}"
            );
        }

        let loss = cuda_add(&cuda_sum_all(&result.out)?, &cuda_sum_all(&result.next_state)?)?;
        let grads = cuda_backward(&loss)?;
        assert!(grads.get(q_id).is_some());
        assert!(grads.get(k_id).is_some());
        assert!(grads.get(v_id).is_some());
        assert!(grads.get(beta_id).is_some());
        assert!(grads.get(g_id).is_some());
        assert!(grads.get(state_id).is_some());
        Ok(())
    }

    #[test]
    fn cuda_index_select_rows_backward_scatter_adds_duplicate_indices() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_index_select_rows backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (3usize, 2usize),
            &device,
        )?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let selected = cuda_index_select_rows(&param, &[2, 0, 2])?;
        let weights = CudaTrainTensor::new(Tensor::from_vec(
            vec![10.0f32, 1.0, 20.0, 2.0, 30.0, 3.0],
            (3usize, 2usize),
            &device,
        )?)?;
        let weighted = cuda_mul(&selected, &weights)?;
        let loss = cuda_sum_all(&weighted)?;

        assert_eq!(selected.dims(), &[3, 2]);
        assert_eq!(selected.to_vec_f32()?, vec![5.0, 6.0, 1.0, 2.0, 5.0, 6.0]);
        assert_eq!(loss.to_vec_f32()?, vec![248.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![20.0, 2.0, 0.0, 0.0, 40.0, 4.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_embedding_lookup_backward_scatter_adds_token_grads() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_embedding_lookup backward smoke: {err}");
                return Ok(());
            }
        };

        let table_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            (4usize, 2usize),
            &device,
        )?;
        let table_id = table_tensor.id();
        let table = CudaTrainTensor::parameter(table_tensor, table_id)?;
        let embedded = cuda_embedding_lookup(&table, &[2, 0, 2])?;
        let weights = CudaTrainTensor::new(Tensor::from_vec(
            vec![10.0f32, 1.0, 20.0, 2.0, 30.0, 3.0],
            (3usize, 2usize),
            &device,
        )?)?;
        let weighted = cuda_mul(&embedded, &weights)?;
        let loss = cuda_sum_all(&weighted)?;

        assert_eq!(embedded.dims(), &[3, 2]);
        assert_eq!(embedded.to_vec_f32()?, vec![5.0, 6.0, 1.0, 2.0, 5.0, 6.0]);
        assert_eq!(loss.to_vec_f32()?, vec![248.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(table_id).expect("table grad").to_vec_f32()?,
            vec![20.0, 2.0, 0.0, 0.0, 40.0, 4.0, 0.0, 0.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_permute_rh_to_hr_backward_inverts_permute() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_permute_rh_to_hr backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2usize, 3usize, 1usize),
            &device,
        )?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let permuted = cuda_permute_rh_to_hr(&param)?;
        let weights = CudaTrainTensor::new(Tensor::from_vec(
            vec![10.0f32, 1.0, 20.0, 2.0, 30.0, 3.0],
            (3usize, 2usize, 1usize),
            &device,
        )?)?;
        let weighted = cuda_mul(&permuted, &weights)?;
        let loss = cuda_sum_all(&weighted)?;

        assert_eq!(permuted.dims(), &[3, 2, 1]);
        assert_eq!(permuted.to_vec_f32()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(loss.to_vec_f32()?, vec![172.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![10.0, 20.0, 30.0, 1.0, 2.0, 3.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_permute_hr_to_rh_backward_inverts_permute() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_permute_hr_to_rh backward smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            (2usize, 3usize, 1usize),
            &device,
        )?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let permuted = cuda_permute_hr_to_rh(&param)?;
        let weights = CudaTrainTensor::new(Tensor::from_vec(
            vec![10.0f32, 1.0, 20.0, 2.0, 30.0, 3.0],
            (3usize, 2usize, 1usize),
            &device,
        )?)?;
        let weighted = cuda_mul(&permuted, &weights)?;
        let loss = cuda_sum_all(&weighted)?;

        assert_eq!(permuted.dims(), &[3, 2, 1]);
        assert_eq!(permuted.to_vec_f32()?, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(loss.to_vec_f32()?, vec![172.0]);
        let grads = cuda_backward(&loss)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![10.0, 20.0, 30.0, 1.0, 2.0, 3.0]
        );
        Ok(())
    }

    #[test]
    fn cuda_sdpa_unmasked_backward_reaches_qkv() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_sdpa_unmasked backward smoke: {err}");
                return Ok(());
            }
        };

        let q_tensor = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.4, 0.3, 0.0, 0.5, -0.3, 0.7],
            (2usize, 2usize, 2usize),
            &device,
        )?;
        let k_tensor = Tensor::from_vec(
            vec![0.1f32, 0.6, 0.8, -0.2],
            (2usize, 1usize, 2usize),
            &device,
        )?;
        let v_tensor = Tensor::from_vec(
            vec![1.0f32, -0.5, 0.25, 0.75],
            (2usize, 1usize, 2usize),
            &device,
        )?;
        let q_id = q_tensor.id();
        let k_id = k_tensor.id();
        let v_id = v_tensor.id();
        let q = CudaTrainTensor::parameter(q_tensor, q_id)?;
        let k = CudaTrainTensor::parameter(k_tensor, k_id)?;
        let v = CudaTrainTensor::parameter(v_tensor, v_id)?;

        let out = cuda_sdpa_unmasked(&q, &k, &v, 1.0)?;
        let loss = cuda_sum_all(&out)?;

        assert_eq!(out.dims(), &[2, 2, 2]);
        let out_values = out.to_vec_f32()?;
        assert!(
            out_values.iter().all(|value| value.is_finite()),
            "SDPA output should stay finite: {out_values:?}"
        );
        let grads = cuda_backward(&loss)?;
        for (name, id, expected_len) in [("q", q_id, 8usize), ("k", k_id, 4usize), ("v", v_id, 4usize)] {
            let values = grads
                .get(id)
                .with_context(|| format!("missing {name} grad"))?
                .to_vec_f32()?;
            assert_eq!(values.len(), expected_len, "{name} grad length mismatch");
            assert!(
                values.iter().all(|value| value.is_finite())
                    && values.iter().any(|value| value.abs() > 1e-6),
                "{name} grad should be finite and non-zero: {values:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_sdpa_prefill_causal_masks_future_keys_and_reaches_qkv() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda_sdpa_prefill_causal backward smoke: {err}");
                return Ok(());
            }
        };

        let q_tensor = Tensor::from_vec(
            vec![0.2f32, -0.1, 0.4, 0.3, 0.0, 0.5, -0.3, 0.7],
            (2usize, 2usize, 2usize),
            &device,
        )?;
        let k_tensor = Tensor::from_vec(
            vec![0.1f32, 0.6, 0.8, -0.2],
            (2usize, 1usize, 2usize),
            &device,
        )?;
        let v_tensor = Tensor::from_vec(
            vec![1.0f32, -0.5, 0.25, 0.75],
            (2usize, 1usize, 2usize),
            &device,
        )?;
        let q_id = q_tensor.id();
        let k_id = k_tensor.id();
        let v_id = v_tensor.id();
        let q = CudaTrainTensor::parameter(q_tensor, q_id)?;
        let k = CudaTrainTensor::parameter(k_tensor, k_id)?;
        let v = CudaTrainTensor::parameter(v_tensor, v_id)?;

        let out = cuda_sdpa_prefill_causal(&q, &k, &v, 1.0)?;
        let loss = cuda_sum_all(&out)?;

        assert_eq!(out.dims(), &[2, 2, 2]);
        let out_values = out.to_vec_f32()?;
        assert_eq!(&out_values[..4], &[1.0, -0.5, 1.0, -0.5]);
        assert!(
            out_values.iter().all(|value| value.is_finite()),
            "causal SDPA output should stay finite: {out_values:?}"
        );
        let grads = cuda_backward(&loss)?;
        for (name, id, expected_len) in [("q", q_id, 8usize), ("k", k_id, 4usize), ("v", v_id, 4usize)] {
            let values = grads
                .get(id)
                .with_context(|| format!("missing {name} grad"))?
                .to_vec_f32()?;
            assert_eq!(values.len(), expected_len, "{name} grad length mismatch");
            assert!(
                values.iter().all(|value| value.is_finite())
                    && values.iter().any(|value| value.abs() > 1e-6),
                "{name} grad should be finite and non-zero: {values:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_native_sgd_step_decreases_sum_square_loss() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda native SGD smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::new(vec![2.0f32, -4.0], &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;

        let before = cuda_sum_all(&cuda_mul(&param, &param)?)?;
        let before_loss = before.to_vec_f32()?[0];
        let grads = cuda_backward(&before)?;
        assert_eq!(
            grads.get(param_id).expect("param grad").to_vec_f32()?,
            vec![4.0, -8.0]
        );

        let updated = cuda_sgd_step_from_store(&[param.clone()], &grads, 0.1)?;
        assert_eq!(updated, 1);
        assert_eq!(param.to_vec_f32()?, vec![1.6, -3.2]);

        let after = cuda_sum_all(&cuda_mul(&param, &param)?)?;
        let after_loss = after.to_vec_f32()?[0];
        assert!(
            after_loss < before_loss,
            "expected SGD to reduce loss: before={before_loss} after={after_loss}"
        );
        Ok(())
    }

    #[test]
    fn cuda_native_adamw_step_decreases_sum_square_loss() -> Result<()> {
        let device = match Device::new_cuda(0) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("CUDA unavailable, skipping cuda native AdamW smoke: {err}");
                return Ok(());
            }
        };

        let param_tensor = Tensor::new(vec![2.0f32, -4.0], &device)?;
        let param_id = param_tensor.id();
        let param = CudaTrainTensor::parameter(param_tensor, param_id)?;
        let mut states = HashMap::new();
        states.insert(param_id, CudaAdamWState::zeros_like(&param)?);

        let before = cuda_sum_all(&cuda_mul(&param, &param)?)?;
        let before_loss = before.to_vec_f32()?[0];
        let grads = cuda_backward(&before)?;
        let updated = cuda_adamw_step_from_store(
            &[param.clone()],
            &grads,
            &mut states,
            CudaAdamWConfig {
                lr: 0.1,
                ..CudaAdamWConfig::default()
            },
        )?;
        assert_eq!(updated, 1);

        let state = states.get(&param_id).expect("adamw state");
        assert_eq!(state.step, 1);
        assert!(state.first_moment.to_vec_f32()?.iter().any(|v| v.abs() > 0.0));
        assert!(state.second_moment.to_vec_f32()?.iter().any(|v| v.abs() > 0.0));

        let after = cuda_sum_all(&cuda_mul(&param, &param)?)?;
        let after_loss = after.to_vec_f32()?[0];
        assert!(
            after_loss < before_loss,
            "expected AdamW to reduce loss: before={before_loss} after={after_loss}"
        );
        Ok(())
    }
}
