# Roadmap

Ordered list of improvements. Each builds on the previous so the
abstractions stay aligned with what the next item actually needs.
Lessons taken from vLLM's V0 → V1 → MRV2 evolution (see `IDEAS.md`).

## 1. Unify forward_kv_only / forward_prefill / forward — DONE (b17e8e3)

A step is just `forward(tokens, start_pos, want_logits)`. Decode is
`tokens.len() == 1`. Prefill is `tokens.len() > 1`. Removes the artificial
prefill/decode split that V0 baked in and had to undo in V1.

## 2. Separate request state from model state — NEXT

Today `LlamaModel` holds weights, KV cache, and scratch activations all in
one struct. Split into:

- `LlamaModel` = weights + config (immutable, shared)
- `RequestState` = KV cache slot + position + token history
- `Activations` = scratch buffers (one set, reused per step)

No behavior change at BS=1, but the moment we add prefix caching or a
serving layer this split is non-negotiable. MRV2's lesson: don't let a
request's *index* become its *storage offset*.

## 3. Persistent / diffable command buffer

Re-encoding the full forward pass per token is wasteful at small models.
Cache the encoded `MTLCommandBuffer` and only patch the position uniform
and token id between steps. vLLM V1's "persistent batch" idea, applies
even at BS=1.

## 4. Prefix caching on the flat KV cache

Hash chunks of input tokens (16/32 per block), store
`hash → (layer, offset_range)`, reuse on match. Critical: zero cost on a
miss (V0's mistake). For chat, every turn reuses system prompt + history
verbatim — huge user-visible win. Needs #2 done first.

## 5. Speculative decoding

1.5–2.5× decode speedup, BS=1 friendly. Llama 3.2 1B drafting for 8B is
the obvious pairing. Needs #1 (run k tokens at pos p) and benefits from
#3 (reuse the verify pass encode).
