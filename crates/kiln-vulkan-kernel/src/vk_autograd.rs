//! `vk_backward`: traverse a `VkTensor` graph in reverse and return
//! per-parameter gradients.
//!
//! Eager-tape design (PyTorch-style): each forward op produces a
//! `VkTensor` carrying an `Arc<dyn VkBackwardOp>` that owns Arc
//! refs to its inputs. `vk_backward(&loss)` does a DFS over the graph
//! to topologically order all reachable nodes, then walks them in
//! reverse, accumulating gradients in a per-op-id map. Parameter leaves
//! (those with `param_id: Some(_)`) accumulate into the returned
//! `VkGradStore`, keyed by candle's `TensorId` so the existing
//! optimizer dispatch path can consume the grads.
//!
//! Multi-use accumulation uses `vk_ops::elementwise::vk_add_no_grad`,
//! which is a forward-only elementwise add kernel that does not record
//! a backward op (we don't differentiate through gradient accumulation).

use crate::vk_ops::elementwise::vk_add_no_grad;
use crate::vk_ops::reduce::vk_ones_like;
use crate::vk_tensor::{VkTensor, VkTensorInner};
use anyhow::{Context, Result};
use candle_core::TensorId;
use std::collections::{HashMap, HashSet};

/// Per-parameter gradient store keyed by `TensorId`. Matches the
/// existing optimizer dispatch's input contract — pass these grads
/// into `apply_adamw_update` / `apply_sgd_update` via the registry.
#[derive(Debug, Default)]
pub struct VkGradStore {
    grads: HashMap<TensorId, VkTensor>,
}

impl VkGradStore {
    pub fn new() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }

    pub fn insert(&mut self, id: TensorId, t: VkTensor) {
        self.grads.insert(id, t);
    }

    pub fn get(&self, id: TensorId) -> Option<&VkTensor> {
        self.grads.get(&id)
    }

    pub fn remove(&mut self, id: TensorId) -> Option<VkTensor> {
        self.grads.remove(&id)
    }

    pub fn into_inner(self) -> HashMap<TensorId, VkTensor> {
        self.grads
    }

    pub fn iter(&self) -> impl Iterator<Item = (&TensorId, &VkTensor)> {
        self.grads.iter()
    }

    pub fn len(&self) -> usize {
        self.grads.len()
    }

    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }
}

/// Walk the autograd graph rooted at `loss` and return per-parameter
/// gradients.
///
/// `loss` must be a scalar (single-element) VkTensor. Internally seeds
/// `grad_loss = ones_like(loss)` and walks the tape in reverse topo order.
pub fn vk_backward(loss: &VkTensor) -> Result<VkGradStore> {
    anyhow::ensure!(
        loss.num_elements() == 1,
        "vk_backward: loss must be scalar (got shape {:?})",
        loss.shape()
    );

    // Topo sort: DFS from loss collecting all reachable VkTensors with grad_fn.
    let mut order: Vec<VkTensor> = Vec::new();
    let mut visited: HashSet<u64> = HashSet::new();
    let mut leaves: Vec<VkTensor> = Vec::new();
    collect_topo(loss, &mut visited, &mut order, &mut leaves);

    // Per-node accumulated gradient map, keyed by op_id.
    let mut grads: HashMap<u64, VkTensor> = HashMap::new();

    // Seed: d(loss)/d(loss) = 1
    let ones = vk_ones_like(loss).context("vk_backward: seed ones_like")?;
    grads.insert(loss.op_id(), ones);

    // Process in reverse topo order (consumers before producers).
    for t in order.iter().rev() {
        let Some(grad_at_out) = grads.remove(&t.op_id()) else {
            continue;
        };
        let Some(gf) = t.grad_fn() else {
            continue;
        };
        let input_grads = gf
            .backward(&grad_at_out)
            .with_context(|| format!("vk_backward: {} bwd", gf.op_name()))?;
        let inputs = gf.input_refs();
        anyhow::ensure!(
            inputs.len() == input_grads.len(),
            "vk_backward: {} returned {} grads for {} inputs",
            gf.op_name(),
            input_grads.len(),
            inputs.len()
        );
        for (input, maybe_grad) in inputs.iter().zip(input_grads.into_iter()) {
            let Some(g) = maybe_grad else { continue };
            if !input.requires_grad() && input.grad_fn().is_none() && input.param_id().is_none() {
                continue;
            }
            anyhow::ensure!(
                g.shape() == input.shape(),
                "vk_backward: {} produced grad of shape {:?} for input of shape {:?}",
                gf.op_name(),
                g.shape(),
                input.shape()
            );
            match grads.remove(&input.op_id()) {
                Some(existing) => {
                    let summed =
                        vk_add_no_grad(&existing, &g).context("vk_backward: grad accumulation")?;
                    grads.insert(input.op_id(), summed);
                }
                None => {
                    grads.insert(input.op_id(), g);
                }
            }
        }
    }

    // Collect parameter-leaf grads into the public store.
    let mut store = VkGradStore::new();
    for leaf in leaves {
        if let Some(pid) = leaf.param_id() {
            if let Some(g) = grads.remove(&leaf.op_id()) {
                store.insert(pid, g);
            }
        }
    }
    Ok(store)
}

/// DFS topo walk. `order` ends up in producer-before-consumer order
/// (so iterating reversed visits consumers first). `leaves` collects
/// any tensor without `grad_fn` that is reachable from `loss` — used
/// to map per-op-id grads back to parameter ids at the end.
fn collect_topo(
    t: &VkTensor,
    visited: &mut HashSet<u64>,
    order: &mut Vec<VkTensor>,
    leaves: &mut Vec<VkTensor>,
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

// VkTensorInner field punning is kept private — exposed via module siblings.
fn _vk_tensor_inner_check(_t: &VkTensorInner) {}
