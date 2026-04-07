# llm-rs

High-performance LLM inference engine in Rust targeting Apple Metal (primary) and NVIDIA CUDA (future).

## Principles

**Correctness, speed, simplicity.** In that order, but never trade simplicity for nothing. "An idiot admires complexity, a genius admires simplicity." Every line of code should be obvious. No abstractions that don't pay for themselves. No cleverness for its own sake. If the simple solution is fast enough, it's the right solution.

- Write the dumbest correct thing first. Optimize with data, not speculation.
- Fewer lines > more lines. Fewer abstractions > more abstractions. Fewer dependencies > more dependencies.
- If you need a comment to explain what the code does, the code is too clever.

## Taste

We are building software with **taste**. The bar isn't "it works" — it's
beautifully simple, immaculately designed, immaculately implemented. Code
should feel inevitable: the kind of thing where, after reading it, you can't
imagine it being written any other way.

**Naming matters more than anything else.** Names are the base of all
understanding and communication. A well-named type, function, or variable
makes the code teach itself. A badly-named one creates confusion that no
amount of documentation can fix. Spend disproportionate time on names. If
you can't name a thing well, you probably don't understand it well enough
to write it yet.

- Prefer concrete, specific names over generic ones (`Session`, not `Context`).
- Prefer the shortest name that's still unambiguous in its scope.
- A function name should describe what it returns or what it does, not how.
- If renaming a thing would make the code clearer, rename it. Always.

## Simplicity budget

We are explicitly NOT trying to become vLLM, llama.cpp, or any other engine
that grew into a maintenance nightmare. The goal is a fast, focused LLM
inference engine that one person can hold in their head. Track size and push
back on growth that doesn't pay for itself.

| Metric                       | Current | Soft cap |
|------------------------------|---------|----------|
| Rust LOC (`src/**/*.rs`)     |   2651  |   6000   |
| Metal LOC (`shaders/*.metal`)|   1498  |   3000   |
| Non-Apple deps (Cargo.toml)  |     3   |     5    |

When you cross a soft cap, stop and ask whether the new code is paying for
its weight. Deletion is a feature. If a roadmap item ships with a net
line-count increase, the PR description should justify why.

Update the table when committing changes that materially shift the counts:
```bash
find src -name '*.rs' | xargs wc -l | tail -1
find shaders -name '*.metal' | xargs wc -l | tail -1
```

## Current Scope (v0.1)
- **Models:** LLaMA family (LLaMA 3.x, Mistral) -- dense only
- **Weights:** GGUF format, direct file read
- **Quantizations:** BF16, FP16, F32, Q4_0, Q8_0, Q4_K, Q6_K
- **Backend:** Metal (primary, ~62 tok/s 1B BF16, ~11.5 tok/s 8B Q4_K), CPU fallback
- **CLI:** Interactive chat + single-shot completion
- **Tokenizer:** BPE from GGUF metadata, GPT-2 byte-level encoding with special token support (no external crate)

## Architecture
- `src/gguf/` -- GGUF parser, direct file read into owned buffer
- `src/tokenizer/` -- BPE tokenizer (GPT-2 byte-level for LLaMA 3, sentencepiece for LLaMA 2)
- `src/tensor/` -- DType enum, TensorView (borrows from GgufFile buffer)
- `src/backend/` -- Backend trait (~12 ops), CPU impl, Metal impl
- `src/model/` -- ModelConfig, LLaMA forward pass with GQA
- `src/kv_cache/` -- Per-layer KV cache, flat [max_seq_len, kv_dim] buffers
- `src/sampler/` -- Temperature, top-k, top-p, custom xorshift64 PRNG
- `src/cli/` -- Clap-based CLI: `complete` and `chat` subcommands
- `shaders/` -- Metal compute shaders (ops.metal: embed, matvec, SIMD matvec with multi-row+register caching, rms_norm, rope, softmax, silu, elementwise, flash attention)
- `build.rs` -- AOT Metal shader compilation (.metal -> .air -> .metallib via xcrun metal4.0, embedded in binary)
- `scripts/` -- Profiling tools (profile.py: parse Metal System Trace from xctrace)

## Key Design Decisions

### Dependencies (3 non-Apple crates)
- `clap` 4 (derive) -- CLI parsing
- `byteorder` -- LE binary reads for GGUF parsing
- `half` -- f16/bf16 types for quantization block scales

Metal bindings: `objc2` + `objc2-metal` + `objc2-foundation` (macOS-only, behind `metal` feature flag).

No `serde`, `tokio`, `rand`, `ndarray`, or any ML framework.

### GGML Conventions (critical)
- **Dimension order:** shapes are `[contiguous_dim, outer_dim]` (innermost first, reversed from standard row-major). A weight matrix `[2048, 512]` means 512 rows of 2048 elements.
- **Embedding:** shape `[dim, vocab_size]` = vocab_size rows of dim elements. Row `token_id` starts at offset `token_id * dim`.
- **Weight matrices:** shape `[in_features, out_features]` = out_features rows of in_features elements. For matvec: iterate over rows (out_features), dot each row with input.

### Backend Trait
```rust
pub trait Backend {
    fn upload_tensor(&self, tv: &TensorView) -> DeviceBuffer;
    fn alloc(&self, shape: &[u64], dtype: DType) -> DeviceBuffer;
    fn argmax(&self, logits: &DeviceBuffer) -> u32;
    fn embed(&self, out: &mut DeviceBuffer, table: &DeviceBuffer, tokens: &[u32]);
    fn matmul(&self, out: &mut DeviceBuffer, weight: &DeviceBuffer, input: &DeviceBuffer);
    fn rms_norm(&self, out: &mut DeviceBuffer, input: &DeviceBuffer, weight: &DeviceBuffer, eps: f32);
    fn rope(&self, q: &mut DeviceBuffer, k: &mut DeviceBuffer, start_pos: usize, head_dim: usize, theta: f32);
    fn attention(&self, out: &mut DeviceBuffer, q: &DeviceBuffer, k_cache: &DeviceBuffer, v_cache: &DeviceBuffer,
                 start_pos: usize, n_heads: usize, n_kv_heads: usize, head_dim: usize);
    fn silu(&self, x: &mut DeviceBuffer);
    fn mul(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer);
    fn add(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer);
    fn copy_into(&self, dst: &mut DeviceBuffer, src: &DeviceBuffer, dst_offset_elements: usize);
    fn read_to_vec_f32(&self, buf: &DeviceBuffer) -> Vec<f32>;
}
```
Every op is shape-agnostic w.r.t. seq_len: decode (`seq_len == 1`) and prefill (`seq_len > 1`)
take exactly the same call. The backend reads `seq_len` from `input.shape[1]` (or 1 if 1D) and
dispatches GEMV vs GEMM, single-query flash attention vs causal attention. The model layer
doesn't know or care. `DeviceBuffer` uses `Box<dyn Any>` for type erasure.

### LLaMA Forward Pass (single token at position `pos`)
```
x = embed(token)
for layer in 0..n_layers:
    x_norm = rms_norm(x, attn_norm)
    q, k, v = wq @ x_norm, wk @ x_norm, wv @ x_norm
    rope(q, k, pos)                          # interleaved pairs: (x[2i], x[2i+1])
    kv_cache.store(layer, pos, k, v)
    attn_out = grouped_query_attention(q, kv_cache, layer, 0..pos+1)
    attn_out = wo @ attn_out
    x = x + attn_out                         # residual
    x_norm = rms_norm(x, ffn_norm)
    x = x + w2 @ (silu(w1 @ x_norm) * (w3 @ x_norm))  # SwiGLU + residual
x = rms_norm(x, output_norm)
logits = output @ x
```
GQA: multiple query heads share KV heads (ratio = n_heads / n_kv_heads).

### KV Cache
Flat `[max_seq_len, kv_dim]` buffer per layer per K/V. Write at `pos * kv_dim`, read `0..pos+1`. No paging in v0.1.

### Weight Loading
1. Read entire GGUF file into owned `Vec<u8>` (single `read_exact` — we need all weights)
2. Parse header, then `TensorView` = `&[u8]` slice into the owned buffer
3. CPU backend stores raw pointer into GgufFile buffer (no copy)
4. Metal: `newBufferWithBytesNoCopy` can wrap the buffer on UMA (Apple Silicon)

### Metal Performance Strategy (Phase 2)
1. Pre-allocate all activation buffers at init -- zero allocations during inference
2. Encode entire forward pass into single `MTLCommandBuffer` with `memoryBarrier` between dependent dispatches, `commit()` once, `waitUntilCompleted()` only after logits readback
3. AOT shader compilation: `.metal` → `.air` → `.metallib` at build time (metal4.0, -O2), embedded in binary via `include_bytes!`, loaded with `newLibraryWithURL` — zero source JIT at runtime
4. Multi-row SIMD matvec: NR=2 rows per simdgroup with register-cached input (amortizes input reads across rows). Float4 wide loads for BF16/F16. Paired nibble processing for Q4_K.
5. BF16 matvec: 167 GB/s (84% SOL), Q4_K: 53 GB/s (27% SOL), Q6_K: 62 GB/s (31% SOL) on M1 Pro
6. `LLM_PERF=1` env var enables per-kernel GPU timing via `MTLCommandBuffer.GPUStartTime()/GPUEndTime()` with BW and FLOPS vs SOL reporting

### CPU Backend
Accelerate BLAS for FP32 matmul, NEON SIMD for quantized dot products. BF16/FP16 with FP32 accumulation via AMX coprocessor.

## Build & Run
```bash
cargo build --release
./target/release/llm-rs -m models/Llama-3.2-1B-Instruct-BF16.gguf complete -p "Hello" -n 30 --temperature 0
./target/release/llm-rs -m models/Llama-3.1-8B-Instruct-Q4_K_M.gguf chat
```

## Profiling (Metal GPU)
```bash
# Capture a Metal System Trace (opens in Instruments)
xcrun xctrace record --template 'Metal System Trace' --output /tmp/llm-rs-profile.trace \
  --launch -- ./target/release/llm-rs -m models/Llama-3.2-1B-Instruct-BF16.gguf complete -p "Hello" -n 10 --temperature 0
open /tmp/llm-rs-profile.trace

# Export trace data from CLI (no Instruments UI needed)
# Table of contents (list all available tables):
xctrace export --input /tmp/llm-rs-profile.trace --toc
# Per-command-buffer GPU timing:
xctrace export --input /tmp/llm-rs-profile.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-application-intervals"]'
# Per-encoder timing:
xctrace export --input /tmp/llm-rs-profile.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-application-encoders-list"]'
# GPU execution intervals:
xctrace export --input /tmp/llm-rs-profile.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'
# Shader list (kernel names + binary sizes):
xctrace export --input /tmp/llm-rs-profile.trace --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-shader-profiler-shader-list"]'
```
Note: Per-kernel GPU timing requires enabling "Shader Timeline" in the Metal Application instrument settings within Instruments.app.

## Implementation Phases

### Phase 1: Foundation (done)
- GGUF parser, metadata, tensor views
- BPE tokenizer from GGUF metadata (GPT-2 byte-level encoding, special token support)
- DType enum, TensorView with GGML dimension convention
- CPU backend with naive loops
- LLaMA forward pass with GQA, pre-allocated activation buffers
- CLI with `complete` and `chat` subcommands, sampler
- Validated on LLaMA 3.2 1B Instruct BF16 (~0.7 tok/s CPU)

### Phase 2: Metal (done)
- MetalBackend: device, queue, AOT shader compilation (metal4.0), pipeline cache
- Multi-row SIMD matvec kernels (NR=2 rows/simdgroup, register-cached input, float4 loads for BF16/F16)
- Flash Attention 2 GPU kernel (online softmax, eliminates per-layer GPU round-trips)
- Lazy command buffer batching (single MTLCommandBuffer per forward pass)
- Q4_K and Q6_K quantization support (CPU + Metal)
- Prefill optimization (skip output projection for non-final tokens)
- Chat template with special token encoding for LLaMA 3
- Per-kernel GPU profiling (LLM_PERF=1, GPU timestamps, BW/FLOPS vs SOL)
- 1B BF16: ~62 tok/s (167 GB/s, 84% SOL), 8B Q4_K_M: ~11.5 tok/s on M1 Pro

### Phase 3: Polish (next)
- Error handling (replace panics with `Result`)
- Batched prefill (GEMM instead of sequential matvec)
- CPU backend optimization (Accelerate BLAS, NEON SIMD)
- MTLBinaryArchive for full GPU ISA pre-compilation (eliminate first-run deferred compilation)

## Verification
1. **GGUF parser:** tensor count/shapes/metadata match llama.cpp output
2. **Tokenizer:** encode/decode round-trip matches llama.cpp
3. **CPU forward pass:** greedy output matches llama.cpp for same model + prompt
4. **Metal forward pass:** output matches CPU backend token-for-token ✓ (verified 1B BF16)
5. **End-to-end:** coherent chat with 8B Q4_K_M model at ~9 tok/s ✓
