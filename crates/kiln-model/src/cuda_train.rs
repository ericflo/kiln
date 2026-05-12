//! CUDA-native training tensor shell.
//!
//! This is the first CUDA-specific training-stack boundary. It deliberately
//! stays small: candle CUDA tensors remain the canonical storage, while this
//! wrapper enforces CUDA residency before calling backend-owned training
//! kernels. Higher-level native CUDA training can build on this without
//! accepting CPU tensors by accident.

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor};

#[derive(Debug, Clone)]
pub struct CudaTrainTensor {
    tensor: Tensor,
}

impl CudaTrainTensor {
    pub fn new(tensor: Tensor) -> Result<Self> {
        ensure!(
            matches!(tensor.device(), Device::Cuda(_)),
            "CudaTrainTensor requires a CUDA tensor, got {:?}",
            tensor.device()
        );
        Ok(Self { tensor })
    }

    pub fn zeros_like(reference: &Tensor) -> Result<Self> {
        ensure!(
            matches!(reference.device(), Device::Cuda(_)),
            "CudaTrainTensor::zeros_like requires a CUDA tensor, got {:?}",
            reference.device()
        );
        let tensor = Tensor::zeros(reference.shape().clone(), reference.dtype(), reference.device())
            .context("alloc CUDA training tensor")?;
        Ok(Self { tensor })
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
}
