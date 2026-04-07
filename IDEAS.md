# Ideas / Future Work

## Single-user (BS=1) latency optimizations

Local LLM users on non-optimal hardware care about latency, not throughput. Keep
v0.1 as a single BS=1-optimized path. Don't split into latency vs throughput
engines until we actually add a serving layer.

- **Quantized KV cache (Q8/Q4 K and V)** — biggest win at long contexts. KV
  dominates memory; shrinking it frees bandwidth and lets users run longer
  prompts on the same hardware.
- **Speculative decoding with a tiny draft model** — 1.5–2.5× decode speedup,
  pure latency win, BS=1 friendly.
- **Flash-decoding / split-K attention** — splits the KV sequence dim across
  SMs/simdgroups so a single query saturates the GPU at long context. Pure
  BS=1 optimization.
- **Better prefill** — MMA tiles, persistent kernels. Batched GEMM already done.
- **Weight streaming / partial offload** — for users whose model doesn't fit
  in VRAM/UMA.

## Prefix caching (worth doing even at BS=1)

Interactive chat reuses the system prompt + conversation history every turn.
This is the one "serving" feature that's also a pure single-user latency win.
Doable on top of our flat KV cache: hash chunks of tokens, store
`hash → (layer, offset)` ranges, copy/skip recomputation on match. Do this
*before* paged attention. Critical: must be designed so it doesn't cost
anything when hit rate is 0% — vLLM V0's prefix cache hurt cold workloads
because it wasn't integrated with the scheduler.

## Throughput path (only if/when we add a server)

- Continuous batching
- Paged attention (only useful with multiple concurrent sequences of varying
  length — strictly overhead at BS=1 vs our flat contiguous KV cache)
- Lift the 4096 max_seq_len cap (`src/model/mod.rs:64`) — currently clamped to
  avoid huge KV allocation; revisit once paged attention lands.

## Lessons from vLLM V0 → V1 → MRV2 (don't repeat their mistakes)

Sources:
- V1: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html
- MRV2 (March 2026): https://vllm.ai/blog/mrv2

1. **Don't bolt features on in isolation.** vLLM V0's prefix caching, chunked
   prefill, and spec decoding were each developed standalone and didn't
   compose. Design the KV cache + scheduler abstraction once with all three
   in mind, even if we only ship one initially.
2. **Unify prefill and decode.** A step is just `{request: n_tokens}`. Don't
   bake a hard prefill/decode split into the model code — `forward_prefill`
   and `forward_kv_only` should collapse into one path parameterized by
   token count. Makes chunked prefill, spec decode, and continuous batching
   trivial later.
3. **Persistent batch / diff-only state.** vLLM V1 caches input tensors and
   applies diffs every step instead of rebuilding. Our equivalent: reuse
   encoded MTLCommandBuffers and only patch position/token, rather than
   re-encoding the full forward pass each step. Applies even at BS=1.
4. **CPU overhead matters even with a fast GPU.** vLLM V0's Python host loop
   was ~5ms — same order as the GPU step on H100. We're in Rust so we start
   ahead, but the principle holds: zero per-step allocations, no string
   formatting, no hash lookups in the hot path.
5. **Stateful workers, incremental updates.** When we eventually add a
   server: workers cache request state locally, scheduler sends only diffs.
   Don't co-locate scheduler with worker 0 for IPC reasons (V0's mistake —
   created asymmetric TP).
6. **Stable state table, per-step gather (MRV2).** Don't let a request's
   *index* become its *storage offset*. Persistent state lives in a
   fixed-size table with stable rows; per step, gather only the active rows
   into the input tensors. V1's mistake was coupling block-table layout to
   request order — made insertion/removal/reorder painful. Cheap to get
   right early, expensive to fix later.
7. **Build per-step inputs on the GPU (MRV2).** Position arrays, attn
   metadata, sampling outputs should never round-trip through the CPU.
   We mostly do this already in our Metal kernels — keep it that way.
   Critical for spec-decode: the rejection mask must be consumable by the
   next step's input prep without a sync.
8. **Async is a core invariant, not an optimization (MRV2).** Once we add
   any pipelining ("schedule step N+1 while GPU runs step N"), commit to
   "zero CPU↔GPU sync in the steady state" as a hard rule. Retrofitting
   async after the fact was V1's biggest source of complexity.

## Misc

- MTLBinaryArchive for full GPU ISA pre-compilation (eliminate first-run
  deferred compilation)
- Replace remaining panics with `Result`
- CPU backend optimization (Accelerate BLAS, NEON SIMD for quantized ops)
