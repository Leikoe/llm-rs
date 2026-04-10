//! Qwen3.5: hybrid GDN (Gated Delta Net) + GQA attention.
//!
//! 75% of layers use GDN linear attention, every 4th layer uses standard GQA.
//! GDN layers have no KV cache — they maintain a recurrent state matrix instead.
//! Full attention layers have no FFN — only QK-normed GQA with RoPE.
//!
//! ## GDN layer forward pass
//!
//! ```text
//! x_norm = rms_norm(x)
//! qkv = silu(causal_conv1d(attn_qkv @ x_norm, conv_state))
//! q, k, v = split(qkv)
//! q, k = l2_norm(q), l2_norm(k)
//! g = -exp(A_log) * softplus(alpha @ x_norm + dt_bias)   // decay gate
//! beta = sigmoid(beta_proj @ x_norm)                     // write gate
//! o, ssm_state = gated_delta_rule(q, k, v, g, beta, ssm_state)
//! o = gated_rms_norm(o, gate @ x_norm)
//! x = x + out_proj @ o
//! x_norm = rms_norm(x)
//! x = x + swiglu_ffn(x_norm)
//! ```
//!
//! ## Full attention layer forward pass (every 4th layer, NO FFN)
//!
//! ```text
//! q, k, v = wq @ x, wk @ x, wv @ x
//! q, k = qk_norm(q), qk_norm(k)
//! q, k = rope(q, k)                                     // partial RoPE (25% of dims)
//! o = gqa_attention(q, k, v, kv_cache)
//! x = x + wo @ o
//! ```
//!
//! ## New ops needed
//!
//! - causal_conv1d: sliding-window conv with persistent state (kernel=4)
//! - l2_norm: per-head L2 normalization of Q and K
//! - gated_delta_rule: linear recurrence `state = g*state + beta*(k⊗v)`, `out = q@state`
//! - gated_rms_norm: `rms_norm(x) * silu(z)`
//! - softplus: `ln(1 + exp(x))` for decay gate
//! - partial RoPE: rotate only 25% of head dimensions
//!
//! ## New state
//!
//! - conv_state: [conv_dim, kernel_size-1] per GDN layer (causal conv1d sliding window)
//! - ssm_state: [n_heads, key_dim, value_dim] per GDN layer (recurrent state matrix)
//! - kv_cache: only for the 1/4 full-attention layers
//!
//! ## GGUF tensors
//!
//! GDN layers (blk.0, 1, 2, 4, 5, 6, ...):
//!   attn_norm, attn_qkv [dim, 3*dim], attn_gate [dim, dim],
//!   ssm_a [n_heads], ssm_dt.bias [n_heads], ssm_conv1d [kernel, conv_dim],
//!   ssm_alpha [dim, n_heads], ssm_beta [dim, n_heads],
//!   ssm_norm [state_size], ssm_out [dim, dim],
//!   ffn_gate/up/down, post_attention_norm
//!
//! Full attention layers (blk.3, 7, 11, ...):
//!   attn_q [dim, q_dim], attn_k [dim, kv_dim], attn_v [dim, kv_dim],
//!   attn_output [dim, dim], attn_q_norm [head_dim], attn_k_norm [head_dim]
//!   (no FFN, no norms — attention only)
