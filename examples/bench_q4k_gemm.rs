/// Q4_K GEMM (prefill) benchmark. Same GPU-counter timing as bench_q4k_gemv,
/// but with seq_len > 1 to exercise gemm_q4k.
use half::{bf16, f16};
use llm_rs::backend::metal::MetalBackend;
use llm_rs::backend::Backend;
use llm_rs::tensor::{DType, TensorView};

const PEAK_BW: f64 = 200.0;

struct Xor(u64);
impl Xor {
    fn next(&mut self) -> u64 {
        let mut x = self.0; x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.0 = x; x
    }
    fn f32m1p1(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    }
}

fn random_q4k(n_elements: usize, rng: &mut Xor) -> Vec<u8> {
    let n_blocks = n_elements / 256;
    let mut out = vec![0u8; n_blocks * 144];
    for blk in out.chunks_exact_mut(144) {
        let d = f16::from_f32(0.02 + (rng.next() as u8 as f32 / 255.0) * 0.03);
        let dmin = f16::from_f32(0.01 + (rng.next() as u8 as f32 / 255.0) * 0.02);
        blk[0..2].copy_from_slice(&d.to_le_bytes());
        blk[2..4].copy_from_slice(&dmin.to_le_bytes());
        for b in &mut blk[4..] { *b = rng.next() as u8; }
    }
    out
}

fn random_bf16(n_elements: usize, rng: &mut Xor) -> Vec<u8> {
    let mut out = vec![0u8; n_elements * 2];
    for pair in out.chunks_exact_mut(2) {
        pair.copy_from_slice(&bf16::from_f32(rng.f32m1p1()).to_le_bytes());
    }
    out
}

const SHAPES: &[(usize, usize, &str)] = &[
    (4096,  14336, "w1 / w3 (up)  "),
    (14336, 4096,  "w2 (down)     "),
];

fn bench(backend: &MetalBackend, m: usize, n: usize, seq: usize, iters: usize) -> (f64, f64) {
    let mut rng = Xor(0xC0FFEE ^ ((m * n * seq) as u64));
    let w_bytes = random_q4k(m * n, &mut rng);
    let i_bytes = random_bf16(m * seq, &mut rng);
    let w_shape = [m as u64, n as u64];
    let weight = backend.upload_tensor(&TensorView {
        name: "w", dtype: DType::Q4K,
        shape: &w_shape, data: &w_bytes,
    });
    let i_shape = [m as u64, seq as u64];
    let input = backend.upload_tensor(&TensorView {
        name: "x", dtype: DType::BF16,
        shape: &i_shape, data: &i_bytes,
    });
    let mut output = backend.alloc(&[n as u64, seq as u64], DType::BF16);

    for _ in 0..3 { backend.matmul(&mut output, &weight, &input); }
    backend.sync();

    backend.reset_gpu_secs();
    for _ in 0..iters { backend.matmul(&mut output, &weight, &input); }
    backend.sync();
    let elapsed = backend.gpu_secs_total();

    let per_iter = elapsed / iters as f64;
    let weight_bytes = DType::Q4K.storage_size(m * n) as f64;
    let bw = weight_bytes / per_iter / 1e9;
    (per_iter * 1e3, bw)
}

fn main() {
    let iters = 50;
    let backend = MetalBackend::new();
    println!("Q4_K GEMM (prefill) benchmark  (iters={iters}, peak={PEAK_BW} GB/s)");
    println!("  {:<16} {:>6} x {:<6}  {:>4}  {:>9}  {:>11}  {:>7}", "shape", "in", "out", "seq", "ms/call", "GB/s", "% SOL");
    println!("  {}", "-".repeat(70));
    for seq in [1usize, 8, 64, 256, 512] {
        for (m, n, label) in SHAPES {
            let (ms, bw) = bench(&backend, *m, *n, seq, iters);
            let sol = bw / PEAK_BW * 100.0;
            println!("  {label:<16} {m:>6} x {n:<6}  {seq:>4}  {ms:7.3}ms  {bw:7.1} GB/s  {sol:5.1}%");
        }
    }
}
