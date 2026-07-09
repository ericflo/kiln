//! Phase R.3 smoke tests — Tensor <-> ROCm device round-trips on a real AMD
//! GPU. Normal developer runs skip when no ROCm device is present; runs with
//! `KILN_QUALIFICATION=1` fail so missing hardware cannot look like evidence.
//!
//! Run with: `cargo test -p kiln-tensor --features rocm --test rocm_storage_smoke`
#![cfg(feature = "rocm")]

use half::bf16;
use kiln_tensor::{DType, Device, Tensor};

fn qualification_required(value: Option<&str>) -> bool {
    value == Some("1")
}

fn no_rocm() -> bool {
    if !kiln_tensor::rocm_is_available() {
        if qualification_required(std::env::var("KILN_QUALIFICATION").ok().as_deref()) {
            panic!("ROCm device unavailable while KILN_QUALIFICATION=1");
        }
        eprintln!("no ROCm device available; skipping R.3 smoke test");
        true
    } else {
        false
    }
}

#[test]
fn qualification_mode_is_exact_opt_in() {
    assert!(qualification_required(Some("1")));
    assert!(!qualification_required(None));
    assert!(!qualification_required(Some("")));
    assert!(!qualification_required(Some("0")));
    assert!(!qualification_required(Some("true")));
}

#[test]
fn zeros_on_rocm_then_host() {
    if no_rocm() {
        return;
    }
    let t = Tensor::zeros_on(Device::Rocm(0), vec![4, 8], DType::F32).expect("zeros_on rocm");
    assert_eq!(t.device(), Device::Rocm(0));
    let host = kiln_tensor::rocm_to_host_copy(&t).expect("rocm_to_host_copy");
    assert_eq!(host.device(), Device::Cpu);
    let v = host.to_vec::<f32>().expect("to_vec");
    assert_eq!(v.len(), 32);
    assert!(
        v.iter().all(|&x| x == 0.0),
        "zeros_on must produce all zeros"
    );
}

#[test]
fn from_vec_on_rocm_roundtrip() {
    if no_rocm() {
        return;
    }
    let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.5).collect();
    let t =
        Tensor::from_vec_on(Device::Rocm(0), data.clone(), vec![4, 8]).expect("from_vec_on rocm");
    assert_eq!(t.device(), Device::Rocm(0));
    let host = kiln_tensor::rocm_to_host_copy(&t).expect("rocm_to_host_copy");
    let got = host.to_vec::<f32>().expect("to_vec");
    assert_eq!(got, data, "H2D -> D2H must round-trip exactly");
}

#[test]
fn rocm_contiguous_materializes_transpose() {
    if no_rocm() {
        return;
    }
    // [[0,1,2],[3,4,5]] on device, transpose to a non-contiguous [3,2] view,
    // then force a contiguous copy through the hipcc contiguity kernel.
    let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
    let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![2, 3]).expect("from_vec_on");
    let tt = t.transpose(0, 1).expect("transpose"); // [3,2], non-contiguous
    assert_eq!(tt.device(), Device::Rocm(0));
    let host = kiln_tensor::rocm_to_host_copy(&tt).expect("to host (forces contiguous kernel)");
    let got = host.to_vec::<f32>().expect("to_vec");
    // transposed [[0,3],[1,4],[2,5]] -> flat [0,3,1,4,2,5]
    assert_eq!(got, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
}

fn patterned_bf16(i: usize) -> bf16 {
    let v = (((i * 37 + 11) % 2048) as f32) / 128.0 - 8.0;
    bf16::from_f32(v)
}

fn assert_bf16_narrow_copy(shape: [usize; 3], axis1_start: usize, axis1_len: usize) {
    let total = shape.iter().product();
    let data: Vec<bf16> = (0..total).map(patterned_bf16).collect();
    let t = Tensor::from_vec_on(Device::Rocm(0), data, shape.to_vec()).expect("from_vec_on bf16");
    assert!(t.all_finite().expect("full tensor finite check"));

    let view = t.narrow(1, axis1_start, axis1_len).expect("axis-1 narrow");
    assert!(
        !view.is_contiguous(),
        "axis-1 narrow should require a stride-aware ROCm copy"
    );
    let copied = kiln_tensor::rocm_contiguous(&view).expect("rocm_contiguous narrow");
    assert_eq!(copied.shape(), &[shape[0], axis1_len, shape[2]]);
    assert!(copied.is_contiguous());
    assert_eq!(copied.layout().start_offset(), 0);
    assert!(copied.all_finite().expect("narrow copy finite check"));

    let host = kiln_tensor::rocm_to_host_copy(&copied).expect("narrow copy to host");
    let got = host.to_vec::<bf16>().expect("to_vec bf16");
    for b in 0..shape[0] {
        for s in 0..axis1_len {
            for d in 0..shape[2] {
                let got_idx = (b * axis1_len + s) * shape[2] + d;
                let src_idx = (b * shape[1] + axis1_start + s) * shape[2] + d;
                assert_eq!(
                    got[got_idx],
                    patterned_bf16(src_idx),
                    "bf16 narrow copy mismatch at b={b} s={s} d={d}"
                );
            }
        }
    }
}

#[test]
fn rocm_contiguous_materializes_nonzero_axis1_bf16_narrow() {
    if no_rocm() {
        return;
    }
    assert_bf16_narrow_copy([3, 4097, 16], 2048, 513);
}

#[test]
fn rocm_contiguous_materializes_large_axis0_bf16_row_slice() {
    if no_rocm() {
        return;
    }

    let rows = 8192usize;
    let hidden = 2560usize;
    let start = 3072usize;
    let len = 512usize;
    let data: Vec<bf16> = (0..rows * hidden).map(patterned_bf16).collect();
    let t = Tensor::from_vec_on(Device::Rocm(0), data, vec![rows, hidden])
        .expect("from_vec_on large bf16 row source");
    assert!(t.all_finite().expect("source finite check"));

    let view = t.narrow(0, start, len).expect("axis-0 row narrow");
    assert!(
        !view.is_contiguous(),
        "nonzero-offset row slice should be materialized before matmul"
    );
    let copied = kiln_tensor::rocm_contiguous(&view).expect("rocm_contiguous row slice");
    assert_eq!(copied.shape(), &[len, hidden]);
    assert!(copied.is_contiguous());
    assert!(copied.all_finite().expect("row-slice finite check"));

    let host = kiln_tensor::rocm_to_host_copy(&copied).expect("row-slice to host");
    let got = host.to_vec::<bf16>().expect("to_vec bf16");
    for row in 0..len {
        for dim in 0..hidden {
            let got_idx = row * hidden + dim;
            let src_idx = (start + row) * hidden + dim;
            assert_eq!(
                got[got_idx],
                patterned_bf16(src_idx),
                "bf16 row-slice copy mismatch at row={row} dim={dim}"
            );
        }
    }
}

#[test]
fn rocm_is_finite_catches_nonzero_axis1_bf16_narrow_nan() {
    if no_rocm() {
        return;
    }
    let shape = [3usize, 4097, 16];
    let nan_src_idx = (shape[1] + 2048) * shape[2] + 7;
    let mut data: Vec<bf16> = (0..shape.iter().product()).map(patterned_bf16).collect();
    data[nan_src_idx] = bf16::NAN;
    let t = Tensor::from_vec_on(Device::Rocm(0), data, shape.to_vec()).expect("from_vec_on bf16");
    assert!(
        !t.all_finite().expect("full tensor finite check"),
        "full BF16 finite check must catch an injected NaN"
    );

    let view = t.narrow(1, 2048, 513).expect("axis-1 narrow");
    let copied = kiln_tensor::rocm_contiguous(&view).expect("rocm_contiguous narrow");
    assert!(
        !copied.all_finite().expect("narrow copy finite check"),
        "narrow BF16 finite check must catch an injected NaN"
    );
}

#[test]
fn rocm_contiguous_materializes_full_attention_q_bf16_transpose() {
    if no_rocm() {
        return;
    }

    let shape = [1usize, 8192, 16, 256];
    let total = shape.iter().product();
    let data: Vec<bf16> = (0..total).map(patterned_bf16).collect();
    let t = Tensor::from_vec_on(Device::Rocm(0), data, shape.to_vec()).expect("from_vec_on bf16");
    assert!(t.all_finite().expect("source finite check"));

    let transposed = t
        .transpose(1, 2)
        .expect("transpose seq/head axes")
        .contiguous()
        .expect("contiguous full-attention q layout");
    assert_eq!(transposed.shape(), &[1, 16, 8192, 256]);
    assert!(transposed.is_contiguous());
    assert!(transposed.all_finite().expect("transposed finite check"));

    let host = kiln_tensor::rocm_to_host_copy(&transposed).expect("transposed to host");
    let got = host.to_vec::<bf16>().expect("to_vec bf16");
    for &(head, seq, dim) in &[
        (0usize, 0usize, 0usize),
        (1, 47, 186),
        (7, 2048, 31),
        (9, 5921, 42),
        (10, 1943, 86),
        (15, 8191, 255),
    ] {
        let got_idx = ((head * shape[1] + seq) * shape[3]) + dim;
        let src_idx = ((seq * shape[2] + head) * shape[3]) + dim;
        assert_eq!(
            got[got_idx],
            patterned_bf16(src_idx),
            "bf16 transpose mismatch at head={head} seq={seq} dim={dim}"
        );
    }
}

#[test]
fn rocm_concat_large_sequence_tiles_preserves_bf16_rows() {
    if no_rocm() {
        return;
    }

    let tile = 2048usize;
    let hidden = 1024usize;
    let n_tiles = 4usize;
    let mut pieces = Vec::with_capacity(n_tiles);
    for tile_idx in 0..n_tiles {
        let elems = tile * hidden;
        let data: Vec<bf16> = (0..elems)
            .map(|i| patterned_bf16(tile_idx * elems + i))
            .collect();
        pieces.push(
            Tensor::from_vec_on(Device::Rocm(0), data, vec![1, tile, hidden])
                .expect("from_vec_on bf16 tile"),
        );
    }
    let refs: Vec<&Tensor> = pieces.iter().collect();
    let out = Tensor::cat(&refs, 1).expect("rocm sequence-tile concat");
    assert_eq!(out.shape(), &[1, tile * n_tiles, hidden]);
    assert!(out.all_finite().expect("concat output finite check"));

    let host = kiln_tensor::rocm_to_host_copy(&out).expect("concat to host");
    let got = host.to_vec::<bf16>().expect("to_vec bf16");
    for &(seq, dim) in &[
        (0usize, 0usize),
        (tile - 1, hidden - 1),
        (tile, 0),
        (tile + 137, 511),
        (2 * tile - 1, 17),
        (3 * tile + 29, hidden - 2),
        (n_tiles * tile - 1, hidden - 1),
    ] {
        let flat = seq * hidden + dim;
        assert_eq!(
            got[flat],
            patterned_bf16(flat),
            "concat mismatch at seq={seq} dim={dim}"
        );
    }
}

#[test]
fn rocm_concat_production_sequence_tiles_preserves_bf16_rows() {
    if no_rocm() {
        return;
    }

    let hidden = 2560usize;
    let tiles = [8192usize, 8192, 8192, 8192, 7546];
    let mut pieces = Vec::with_capacity(tiles.len());
    let mut seq_base = 0usize;
    for &tile_len in &tiles {
        let elems = tile_len * hidden;
        let data: Vec<bf16> = (0..elems)
            .map(|i| patterned_bf16(seq_base * hidden + i))
            .collect();
        pieces.push(
            Tensor::from_vec_on(Device::Rocm(0), data, vec![1, tile_len, hidden])
                .expect("from_vec_on bf16 production tile"),
        );
        seq_base += tile_len;
    }

    let refs: Vec<&Tensor> = pieces.iter().collect();
    let out = Tensor::cat(&refs, 1).expect("rocm production sequence-tile concat");
    assert_eq!(out.shape(), &[1, tiles.iter().sum::<usize>(), hidden]);
    assert!(
        out.all_finite()
            .expect("production concat output finite check")
    );

    let host = kiln_tensor::rocm_to_host_copy(&out).expect("production concat to host");
    let got = host.to_vec::<bf16>().expect("to_vec bf16");
    for &(seq, dim) in &[
        (0usize, 0usize),
        (8191, hidden - 1),
        (8192, 0),
        (12014, 968),
        (16384, 0),
        (32768, 0),
        (40313, hidden - 1),
    ] {
        let flat = seq * hidden + dim;
        assert_eq!(
            got[flat],
            patterned_bf16(flat),
            "production concat mismatch at seq={seq} dim={dim}"
        );
    }
}

#[test]
#[ignore = "allocates a production-sized Q tile source (~394 MB) for long-context ROCm debugging"]
fn rocm_contiguous_materializes_production_qtile_bf16_narrow() {
    if no_rocm() {
        return;
    }
    assert_bf16_narrow_copy([16, 48092, 256], 2048, 2048);
}

#[test]
#[ignore = "allocates a production-sized Q tile source (~394 MB) for long-context ROCm debugging"]
fn rocm_is_finite_catches_production_qtile_bf16_nan() {
    if no_rocm() {
        return;
    }
    let shape = [16usize, 48092, 256];
    let nan_src_idx = (3 * shape[1] + 2048) * shape[2] + 17;
    let mut data: Vec<bf16> = (0..shape.iter().product()).map(patterned_bf16).collect();
    data[nan_src_idx] = bf16::NAN;
    let t = Tensor::from_vec_on(Device::Rocm(0), data, shape.to_vec()).expect("from_vec_on bf16");
    assert!(
        !t.all_finite().expect("full tensor finite check"),
        "production-sized BF16 finite check must catch an injected NaN"
    );

    let view = t.narrow(1, 2048, 2048).expect("axis-1 narrow");
    let copied = kiln_tensor::rocm_contiguous(&view).expect("rocm_contiguous narrow");
    assert!(
        !copied.all_finite().expect("narrow copy finite check"),
        "production-sized BF16 narrow finite check must catch an injected NaN"
    );
}
