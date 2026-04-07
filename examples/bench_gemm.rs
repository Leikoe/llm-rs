/// GEMM benchmark: measures matmul throughput for different dtypes and shapes.
///
/// Usage:
///   cargo run --release --example bench_gemm
///   cargo run --release --example bench_gemm -- --m 4096 --n 4096 --seq 64
///   cargo run --release --example bench_gemm -- --dtype bf16
use std::time::Instant;

use llm_rs::backend::metal::MetalBackend;
use llm_rs::backend::Backend;
use llm_rs::tensor::DType;

struct BenchConfig {
    m: usize,      // in_features (contiguous dim of weight)
    n: usize,      // out_features (rows of weight)
    seq_len: usize, // number of input columns
    dtype: DType,   // weight dtype
    iters: usize,
}

fn bench_matmul(backend: &MetalBackend, cfg: &BenchConfig) -> (f64, f64, f64) {
    let weight = backend.alloc(&[cfg.m as u64, cfg.n as u64], cfg.dtype);
    let input = backend.alloc(&[cfg.m as u64, cfg.seq_len as u64], DType::BF16);
    let mut output = backend.alloc(&[cfg.n as u64, cfg.seq_len as u64], DType::BF16);

    for _ in 0..3 { backend.matmul(&mut output, &weight, &input); }
    backend.sync();

    let start = Instant::now();
    for _ in 0..cfg.iters { backend.matmul(&mut output, &weight, &input); }
    backend.sync();
    let elapsed = start.elapsed().as_secs_f64();

    let per_iter = elapsed / cfg.iters as f64;
    let weight_bytes = cfg.dtype.storage_size(cfg.m * cfg.n);
    let input_bytes = cfg.m * cfg.seq_len * 2;
    let output_bytes = cfg.n * cfg.seq_len * 2;
    let total_bytes = (weight_bytes + input_bytes + output_bytes) as f64;
    let flops = (2 * cfg.m * cfg.n * cfg.seq_len) as f64;

    (per_iter * 1e3, total_bytes / per_iter / 1e9, flops / per_iter / 1e9)
}

fn bench_matvec_seq(backend: &MetalBackend, cfg: &BenchConfig) -> (f64, f64, f64) {
    let weight = backend.alloc(&[cfg.m as u64, cfg.n as u64], cfg.dtype);
    let input = backend.alloc(&[cfg.m as u64], DType::BF16);
    let mut output = backend.alloc(&[cfg.n as u64], DType::BF16);

    for _ in 0..3 {
        for _ in 0..cfg.seq_len { backend.matmul(&mut output, &weight, &input); }
    }
    backend.sync();

    let start = Instant::now();
    for _ in 0..cfg.iters {
        for _ in 0..cfg.seq_len { backend.matmul(&mut output, &weight, &input); }
    }
    backend.sync();
    let elapsed = start.elapsed().as_secs_f64();

    let per_iter = elapsed / cfg.iters as f64;
    let weight_bytes = cfg.dtype.storage_size(cfg.m * cfg.n);
    let io_bytes = (cfg.m + cfg.n) * 2;
    let total_bytes = (weight_bytes + io_bytes) as f64 * cfg.seq_len as f64;
    let flops = (2 * cfg.m * cfg.n * cfg.seq_len) as f64;

    (per_iter * 1e3, total_bytes / per_iter / 1e9, flops / per_iter / 1e9)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut m = 4096;
    let mut n = 4096;
    let mut seq_len = 32;
    let mut dtype_str = "bf16".to_string();
    let mut iters = 100;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--m" => { m = args[i + 1].parse().unwrap(); i += 2; }
            "--n" => { n = args[i + 1].parse().unwrap(); i += 2; }
            "--seq" => { seq_len = args[i + 1].parse().unwrap(); i += 2; }
            "--dtype" => { dtype_str = args[i + 1].clone(); i += 2; }
            "--iters" => { iters = args[i + 1].parse().unwrap(); i += 2; }
            _ => { eprintln!("Unknown arg: {}", args[i]); i += 1; }
        }
    }

    let dtype = match dtype_str.as_str() {
        "f32" => DType::F32,
        "bf16" => DType::BF16,
        "f16" => DType::F16,
        "q4_0" => DType::Q4_0,
        "q8_0" => DType::Q8_0,
        "q4k" => DType::Q4K,
        "q6k" => DType::Q6K,
        _ => panic!("Unknown dtype: {dtype_str}"),
    };

    let backend = MetalBackend::new();

    println!("GEMM benchmark: [{m}, {n}] {dtype_str} × [{m}, {seq_len}] bf16");
    println!("  weight: {:.1} MB, iters: {iters}", dtype.storage_size(m * n) as f64 / 1e6);
    println!();

    let cfg = BenchConfig { m, n, seq_len, dtype, iters };

    let (ms, bw, gf) = bench_matmul(&backend, &cfg);
    println!("  GEMM:        {ms:7.2}ms  {bw:6.1} GB/s  {gf:7.1} GFLOP/s");

    let (ms2, bw2, gf2) = bench_matvec_seq(&backend, &cfg);
    println!("  seq. matvec: {ms2:7.2}ms  {bw2:6.1} GB/s  {gf2:7.1} GFLOP/s");

    println!();
    println!("  GEMM speedup: {:.2}×", ms2 / ms);
}
