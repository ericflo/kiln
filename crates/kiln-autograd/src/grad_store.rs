//! `GradStore` — per-parameter gradient bucket, keyed on `TensorId`.
//!
//! Lifted from `vk_autograd::VkGradStore` (Phase 0.1's audit shows 6
//! candle `GradStore` references that point to this lift target).

use std::collections::hash_map::Iter;
use std::collections::HashMap;

use kiln_tensor::{Tensor, TensorId};

/// Gradient bucket. Each parameter's gradient lives behind its
/// `TensorId` (which is stable across storage-variant transitions
/// per kiln-param's anti-pattern 11 contract).
#[derive(Debug, Default)]
pub struct GradStore {
    grads: HashMap<TensorId, Tensor>,
}

impl GradStore {
    /// Construct an empty store.
    pub fn new() -> Self {
        GradStore {
            grads: HashMap::new(),
        }
    }

    /// Insert / replace a gradient for `id`.
    pub fn insert(&mut self, id: TensorId, t: Tensor) {
        self.grads.insert(id, t);
    }

    /// Borrow a gradient.
    pub fn get(&self, id: TensorId) -> Option<&Tensor> {
        self.grads.get(&id)
    }

    /// Remove and return a gradient.
    pub fn remove(&mut self, id: TensorId) -> Option<Tensor> {
        self.grads.remove(&id)
    }

    /// True iff this `id` has an accumulated gradient.
    pub fn contains(&self, id: TensorId) -> bool {
        self.grads.contains_key(&id)
    }

    /// Number of parameters with gradients.
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// True iff the store has no gradients.
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    /// Iterate over `(id, grad)` pairs.
    pub fn iter(&self) -> Iter<'_, TensorId, Tensor> {
        self.grads.iter()
    }

    /// Consume and return the underlying `HashMap`. Used by callers
    /// that want to drain into an optimizer step.
    pub fn into_inner(self) -> HashMap<TensorId, Tensor> {
        self.grads
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_tensor::{DType, Tensor};

    fn t() -> Tensor {
        Tensor::zeros_cpu(vec![2, 2], DType::F32)
    }

    #[test]
    fn empty_store() {
        let s = GradStore::new();
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.iter().count(), 0);
    }

    #[test]
    fn insert_and_get() {
        let mut s = GradStore::new();
        let g = t();
        let id = TensorId::from_raw(42);
        s.insert(id, g);
        assert!(s.contains(id));
        assert_eq!(s.len(), 1);
        assert!(s.get(id).is_some());
    }

    #[test]
    fn remove_returns_owned() {
        let mut s = GradStore::new();
        let id = TensorId::from_raw(7);
        s.insert(id, t());
        let _g = s.remove(id).unwrap();
        assert!(!s.contains(id));
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn insert_replaces_existing() {
        let mut s = GradStore::new();
        let id = TensorId::from_raw(1);
        s.insert(id, t());
        s.insert(id, t()); // replace
        assert_eq!(s.len(), 1);
    }

    #[test]
    fn into_inner_drains() {
        let mut s = GradStore::new();
        s.insert(TensorId::from_raw(1), t());
        s.insert(TensorId::from_raw(2), t());
        let m = s.into_inner();
        assert_eq!(m.len(), 2);
    }
}
