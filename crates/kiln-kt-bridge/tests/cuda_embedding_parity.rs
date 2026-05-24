//! Parity test: kt CUDA embedding lookup vs kt CPU reference.
//!
//! Phase 4 substrate validation. `EmbeddingOp::cuda_fwd` routes
//! through `cuda_index_select_dim0`; this test confirms the gather
//! produces byte-identical output to the CPU reference for both
//! 1-D and 2-D token_ids shapes.



use candle_core::backend::BackendDevice;
use candle_core::{DType as CandleDType, Device as CandleDevice, Tensor as CandleTensor};

use kiln_tensor::{ops, Tensor};

fn try_cuda() -> Option<CandleDevice> {
    CandleDevice::new_cuda(0).ok()
}

fn make_weights_f32_pattern(vocab: usize, hidden: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(vocab * hidden);
    for v in 0..vocab {
        for h in 0..hidden {
            // Deterministic per-(vocab, hidden) value — easy to debug.
            out.push((v * 1000 + h) as f32 / 1000.0);
        }
    }
    out
}

fn cpu_reference(
    weights_data: &[f32],
    vocab: usize,
    hidden: usize,
    token_ids: &[u32],
) -> Vec<f32> {
    let w = Tensor::from_slice(weights_data, vec![vocab, hidden]).unwrap();
    let ids = Tensor::from_slice(token_ids, vec![token_ids.len()]).unwrap();
    let y = ops::embedding(&w, &ids).unwrap();
    let cpu = y
        .storage()
        .as_any()
        .downcast_ref::<kiln_tensor::CpuStorage>()
        .unwrap();
    let bytes = cpu.as_bytes();
    let mut out = Vec::with_capacity(token_ids.len() * hidden);
    for i in 0..(token_ids.len() * hidden) {
        out.push(f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()));
    }
    out
}

#[test]
fn cuda_embedding_1d_token_ids_f32() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let vocab = 64usize;
    let hidden = 32usize;
    let weights_data = make_weights_f32_pattern(vocab, hidden);
    let token_ids: Vec<u32> = vec![0, 5, 13, 27, 63, 2];

    let w_cd = CandleTensor::from_vec(weights_data.clone(), (vocab, hidden), &dev).unwrap();
    let ids_cd = CandleTensor::from_vec(token_ids.clone(), (token_ids.len(),), &dev).unwrap();
    let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&w_cd).unwrap();
    let ids_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&ids_cd).unwrap();

    let out_kt = ops::embedding(&w_kt, &ids_kt).expect("embedding dispatch");

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .reshape((token_ids.len() * hidden,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let ref_v = cpu_reference(&weights_data, vocab, hidden, &token_ids);

    assert_eq!(got.len(), ref_v.len());
    for (i, (a, b)) in ref_v.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "idx={i} ref={a} got={b}"
        );
    }
}

#[test]
fn cuda_embedding_2d_token_ids_bf16() {
    let Some(dev) = try_cuda() else {
        eprintln!("CUDA not available; skipping");
        return;
    };

    let vocab = 128usize;
    let hidden = 16usize;
    let batch = 2usize;
    let seq = 4usize;

    let weights_data = make_weights_f32_pattern(vocab, hidden);
    let token_ids: Vec<u32> = vec![1, 7, 33, 127, 0, 5, 11, 21];
    assert_eq!(token_ids.len(), batch * seq);

    let w_cd = CandleTensor::from_vec(weights_data.clone(), (vocab, hidden), &dev)
        .unwrap()
        .to_dtype(CandleDType::BF16)
        .unwrap();
    let ids_cd = CandleTensor::from_vec(token_ids.clone(), (batch, seq), &dev).unwrap();
    let w_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&w_cd).unwrap();
    let ids_kt = kiln_kt_bridge::kt_tensor_from_candle_cuda_borrow(&ids_cd).unwrap();

    let out_kt = ops::embedding(&w_kt, &ids_kt).expect("embedding 2D dispatch");
    assert_eq!(out_kt.shape(), &[batch, seq, hidden]);

    let cuda_dev = match dev {
        CandleDevice::Cuda(ref c) => c,
        _ => unreachable!(),
    };
    cuda_dev.synchronize().unwrap();

    let got: Vec<f32> = kiln_kt_bridge::kt_tensor_to_candle_cuda_copy(&out_kt)
        .unwrap()
        .to_dtype(CandleDType::F32)
        .unwrap()
        .reshape((token_ids.len() * hidden,))
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // Reference: gather the same rows from the F32 weights, then
    // bf16-quantize to match the kt path's dtype.
    let ref_f32 = cpu_reference(&weights_data, vocab, hidden, &token_ids);
    let mut ref_bf16_then_f32 = Vec::with_capacity(ref_f32.len());
    for v in &ref_f32 {
        ref_bf16_then_f32.push(half::bf16::from_f32(*v).to_f32());
    }

    for (i, (a, b)) in ref_bf16_then_f32.iter().zip(got.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "idx={i} ref(bf16)={a} got={b}"
        );
    }
}
