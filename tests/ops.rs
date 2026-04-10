//! Backend-agnostic op correctness tests.
//!
//! Every test compares backend output against a plain-Rust CPU reference on
//! random inputs. The tests take `&impl Backend`, so adding a new backend
//! just means adding another `#[test]` that instantiates it and forwards to
//! the same check functions.

use half::{bf16, f16};
use llm_rs::backend::Backend;
use llm_rs::tensor::{DType, TensorView};

// ---------- tiny deterministic PRNG ----------

struct Xor(u64);
impl Xor {
    fn new(seed: u64) -> Self { Xor(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1) }
    fn next(&mut self) -> u64 {
        let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x; x
    }
    fn u8(&mut self) -> u8 { self.next() as u8 }
    fn f32m1p1(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

// ---------- Q4_K CPU reference ----------

fn unpack_q4k_scale(scales: &[u8], sb: usize) -> (f32, f32) {
    let (sc, m);
    if sb < 4 {
        sc = (scales[sb] & 63) as u32;
        m  = (scales[sb + 4] & 63) as u32;
    } else {
        sc = ((scales[sb + 4] & 0x0F) as u32) | (((scales[sb - 4] >> 6) as u32) << 4);
        m  = ((scales[sb + 4] >> 4) as u32)   | (((scales[sb] >> 6) as u32)     << 4);
    }
    (sc as f32, m as f32)
}

fn dequant_q4k_row(row_bytes: &[u8], out: &mut [f32]) {
    assert!(out.len() % 256 == 0);
    for (b, blk) in row_bytes.chunks_exact(144).enumerate() {
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin = f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        for sb in 0..8 {
            let (sc, m) = unpack_q4k_scale(scales, sb);
            let pair = sb / 2;
            let hi = sb % 2 == 1;
            for j in 0..32 {
                let packed = qs[pair * 32 + j];
                let q = if hi { (packed >> 4) as f32 } else { (packed & 0x0F) as f32 };
                out[b * 256 + sb * 32 + j] = d * sc * q - dmin * m;
            }
        }
    }
}

// ---------- random data generators ----------

/// Random Q4_K weight bytes with sane (small positive) d/dmin so dequantized
/// values stay in a numerically reasonable range. Raw-random d/dmin would
/// include NaN/Inf f16 values and blow up the reference.
fn random_q4k_bytes(n_elements: usize, rng: &mut Xor) -> Vec<u8> {
    let n_blocks = n_elements / 256;
    let mut out = vec![0u8; n_blocks * 144];
    for blk in out.chunks_exact_mut(144) {
        let d = f16::from_f32(0.02 + (rng.u8() as f32 / 255.0) * 0.03);
        let dmin = f16::from_f32(0.01 + (rng.u8() as f32 / 255.0) * 0.02);
        blk[0..2].copy_from_slice(&d.to_le_bytes());
        blk[2..4].copy_from_slice(&dmin.to_le_bytes());
        for b in &mut blk[4..] { *b = rng.u8(); }
    }
    out
}

fn random_bf16_bytes(n_elements: usize, rng: &mut Xor) -> Vec<u8> {
    let mut out = vec![0u8; n_elements * 2];
    for pair in out.chunks_exact_mut(2) {
        pair.copy_from_slice(&bf16::from_f32(rng.f32m1p1()).to_le_bytes());
    }
    out
}

// ---------- the generic check ----------

/// Q4_K matmul: random weights × random BF16 input, compare vs CPU dequant+GEMM.
/// `in_f` must be a multiple of 256 (Q4_K block size).
fn check_matmul_q4k<B: Backend>(backend: &B, in_f: usize, out_f: usize, seq_len: usize) {
    assert!(in_f % 256 == 0);
    let mut rng = Xor::new(0xC0FFEE ^ ((in_f * out_f * seq_len) as u64));

    let w_bytes = random_q4k_bytes(in_f * out_f, &mut rng);
    let i_bytes = random_bf16_bytes(in_f * seq_len, &mut rng);

    let w_shape = [in_f as u64, out_f as u64];
    let weight = backend.upload_tensor(&TensorView {
        name: "w", dtype: DType::Q4K,
        shape: &w_shape, data: &w_bytes,
    });
    let i_shape_1d = [in_f as u64];
    let i_shape_2d = [in_f as u64, seq_len as u64];
    let input = backend.upload_tensor(&TensorView {
        name: "x", dtype: DType::BF16,
        shape: if seq_len == 1 { &i_shape_1d } else { &i_shape_2d }, data: &i_bytes,
    });
    let out_shape: Vec<u64> = if seq_len == 1 {
        vec![out_f as u64]
    } else {
        vec![out_f as u64, seq_len as u64]
    };
    let mut output = backend.alloc(&out_shape, DType::BF16);

    // CPU reference.
    let mut w_f32 = vec![0.0f32; in_f * out_f];
    let row_bytes_len = (in_f / 256) * 144;
    for row in 0..out_f {
        let src = &w_bytes[row * row_bytes_len..(row + 1) * row_bytes_len];
        dequant_q4k_row(src, &mut w_f32[row * in_f..(row + 1) * in_f]);
    }
    let i_f32: Vec<f32> = i_bytes.chunks_exact(2)
        .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect();

    let mut expected = vec![0.0f32; out_f * seq_len];
    for s in 0..seq_len {
        for row in 0..out_f {
            let mut acc = 0.0f32;
            for i in 0..in_f {
                acc += w_f32[row * in_f + i] * i_f32[s * in_f + i];
            }
            expected[s * out_f + row] = acc;
        }
    }

    backend.matmul(&mut output, &weight, &input);
    backend.sync();
    let got = backend.read_to_vec_f32(&output);

    // Output stored as BF16 (ulp ≈ |x|/128). Small floor for near-zero results.
    let mut max_rel = 0.0f32;
    for (i, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
        let tol = (e.abs() / 64.0).max(0.02 * (in_f as f32).sqrt());
        let err = (e - g).abs();
        let rel = err / e.abs().max(1e-6);
        if rel > max_rel { max_rel = rel; }
        assert!(
            err < tol,
            "q4k matmul mismatch at {i} (in_f={in_f} out_f={out_f} seq={seq_len}): \
             expected {e}, got {g}, err {err} > tol {tol}"
        );
    }
    eprintln!("q4k matmul {in_f}×{out_f} seq={seq_len}: max_rel_err={max_rel:.5}");
}

// ---------- Metal backend bindings ----------

#[cfg(target_os = "macos")]
mod metal {
    use super::*;
    use llm_rs::backend::metal::MetalBackend;

    #[test]
    fn q4k_matmul_decode_small() {
        let backend = MetalBackend::new();
        check_matmul_q4k(&backend, 256, 8, 1);
        check_matmul_q4k(&backend, 512, 16, 1);
        check_matmul_q4k(&backend, 1024, 64, 1);
    }

    #[test]
    fn q4k_matmul_decode_large() {
        let backend = MetalBackend::new();
        check_matmul_q4k(&backend, 4096, 4096, 1);
    }

    #[test]
    fn q4k_matmul_prefill() {
        let backend = MetalBackend::new();
        check_matmul_q4k(&backend, 256, 8, 4);
        check_matmul_q4k(&backend, 1024, 64, 7);
        check_matmul_q4k(&backend, 4096, 1024, 16);
    }
}
