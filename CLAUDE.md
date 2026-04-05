# llm-rs

High-performance LLM inference engine in Rust targeting Apple Metal (primary) and NVIDIA CUDA (future).

## Principles

**Correctness, speed, simplicity.** In that order, but never trade simplicity for nothing. "An idiot admires complexity, a genius admires simplicity." Every line of code should be obvious. No abstractions that don't pay for themselves. No cleverness for its own sake. If the simple solution is fast enough, it's the right solution.

- Write the dumbest correct thing first. Optimize with data, not speculation.
- Fewer lines > more lines. Fewer abstractions > more abstractions. Fewer dependencies > more dependencies.
- If you need a comment to explain what the code does, the code is too clever.

## Current Scope (v0.1)
- **Models:** LLaMA family (LLaMA 3.x, Mistral) -- dense only
- **Weights:** GGUF format, direct file read
- **Quantizations:** BF16, FP16, F32, Q4_0, Q8_0, Q4_K, Q6_K
- **Backend:** Metal (primary, ~50 tok/s 1B, ~9 tok/s 8B), CPU fallback
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
- `shaders/` -- Metal compute shaders (ops.metal: embed, matvec, SIMD matvec, rms_norm, rope, softmax, silu, elementwise, GQA attention)
- `build.rs` -- Metal shader compilation (.metal -> .air -> .metallib via xcrun)

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
    fn matvec_mul(&self, out: &mut DeviceBuffer, weight: &DeviceBuffer, input: &DeviceBuffer);
    fn matmul(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer);
    fn rms_norm(&self, out: &mut DeviceBuffer, input: &DeviceBuffer, weight: &DeviceBuffer, eps: f32);
    fn rope(&self, q: &mut DeviceBuffer, k: &mut DeviceBuffer, pos: usize, head_dim: usize, theta: f32);
    fn softmax(&self, x: &mut DeviceBuffer, len: usize);
    fn silu(&self, x: &mut DeviceBuffer);
    fn mul(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer);
    fn add(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer);
    fn embed(&self, out: &mut DeviceBuffer, table: &DeviceBuffer, token_id: u32);
    fn copy_into(&self, dst: &mut DeviceBuffer, src: &DeviceBuffer, dst_offset_elements: usize);
    fn read_to_vec_f32(&self, buf: &DeviceBuffer) -> Vec<f32>;
}
```
Operations map 1:1 to transformer forward pass. `DeviceBuffer` uses `Box<dyn Any>` for type erasure.

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
3. BF16/FP16 inputs with FP32 accumulation in matmul kernels
4. Quantized matvec: SIMD group reductions for memory bandwidth
5. Prefill (seq_len > 1): GEMM; consider MPS for FP16

### CPU Backend
Accelerate BLAS for FP32 matmul, NEON SIMD for quantized dot products. BF16/FP16 with FP32 accumulation via AMX coprocessor.

## Build & Run
```bash
cargo build --release
./target/release/llm-rs -m models/Llama-3.2-1B-Instruct-BF16.gguf complete -p "Hello" -n 30 --temperature 0
./target/release/llm-rs -m models/Llama-3.1-8B-Instruct-Q4_K_M.gguf chat
```

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
- MetalBackend: device, queue, runtime shader compilation, pipeline cache
- SIMD matvec kernels (32-lane cooperative dot products via simd_sum)
- GPU GQA attention kernel (eliminates per-layer GPU round-trips)
- Lazy command buffer batching (single MTLCommandBuffer per forward pass)
- Q4_K and Q6_K quantization support (CPU + Metal)
- Prefill optimization (skip output projection for non-final tokens)
- Chat template with special token encoding for LLaMA 3
- 1B BF16: ~50 tok/s, 8B Q4_K_M: ~9 tok/s on M1 Pro

### Phase 3: Polish (next)
- Error handling (replace panics with `Result`)
- Performance profiling with Instruments
- Batched prefill (GEMM instead of sequential matvec)
- CPU backend optimization (Accelerate BLAS, NEON SIMD)

## Verification
1. **GGUF parser:** tensor count/shapes/metadata match llama.cpp output
2. **Tokenizer:** encode/decode round-trip matches llama.cpp
3. **CPU forward pass:** greedy output matches llama.cpp for same model + prompt
4. **Metal forward pass:** output matches CPU backend token-for-token ✓ (verified 1B BF16)
5. **End-to-end:** coherent chat with 8B Q4_K_M model at ~9 tok/s ✓
