//! CUDA-native training tensor shell.
//!
//! This is the first CUDA-specific training-stack boundary. It deliberately
//! stays small: candle CUDA tensors remain the canonical storage, while this
//! wrapper enforces CUDA residency before calling backend-owned training
//! kernels. Higher-level native CUDA training can build on this without
//! accepting CPU tensors by accident.

use anyhow::{Context, Result, ensure};
use candle_core::{D, DType, Device, Tensor, TensorId};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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
