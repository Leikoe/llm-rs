use std::collections::HashMap;

use crate::backend::Backend;
use crate::gguf::metadata::MetadataValue;
use crate::kv_cache::KVCache;
use crate::tensor::DType;

/// Model hyperparameters parsed from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub architecture: String,
    pub vocab_size: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
}

impl ModelConfig {
    pub fn from_gguf_metadata(metadata: &HashMap<String, MetadataValue>) -> Self {
        let architecture = metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .unwrap_or("llama")
            .to_string();
        let arch = &architecture;

        // GGUF doesn't store vocab_size directly; derive it from the token table length.
        let vocab_size = metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .expect("missing tokenizer.ggml.tokens")
            .len();

        let dim = get_u32(metadata, &format!("{arch}.embedding_length"))
            .expect("missing embedding_length") as usize;
        let n_layers = get_u32(metadata, &format!("{arch}.block_count"))
            .expect("missing block_count") as usize;
        let n_heads = get_u32(metadata, &format!("{arch}.attention.head_count"))
            .expect("missing head_count") as usize;
        let n_kv_heads = get_u32(metadata, &format!("{arch}.attention.head_count_kv"))
            .unwrap_or(n_heads as u32) as usize;
        // Qwen3 stores head_dim explicitly (key_length); LLaMA derives it from dim/n_heads.
        let head_dim = get_u32(metadata, &format!("{arch}.attention.key_length"))
            .map(|v| v as usize)
            .unwrap_or(dim / n_heads);
        let hidden_dim = get_u32(metadata, &format!("{arch}.feed_forward_length"))
            .expect("missing feed_forward_length") as usize;
        let norm_eps = get_f32(metadata, &format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let rope_theta = get_f32(metadata, &format!("{arch}.rope.freq_base"))
            .unwrap_or(10000.0);
        // Cap context to avoid huge KV cache during development.
        let max_seq_len = (get_u32(metadata, &format!("{arch}.context_length"))
            .unwrap_or(4096) as usize)
            .min(4096);

        ModelConfig {
            architecture, vocab_size, dim, n_layers, n_heads, n_kv_heads,
            head_dim, hidden_dim, norm_eps, rope_theta, max_seq_len,
        }
    }
}

fn get_u32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Option<u32> {
    metadata.get(key).and_then(|v| match v {
        MetadataValue::U32(n) => Some(*n),
        MetadataValue::I32(n) => Some(*n as u32),
        MetadataValue::U64(n) => Some(*n as u32),
        MetadataValue::I64(n) => Some(*n as u32),
        _ => None,
    })
}

fn get_f32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Option<f32> {
    metadata.get(key).and_then(|v| match v {
        MetadataValue::F32(n) => Some(*n),
        MetadataValue::F64(n) => Some(*n as f32),
        _ => None,
    })
}

/// Token-mixing block. One variant per attention/SSM family. Each layer
/// picks a variant; `forward` matches per-layer so heterogeneous stacks
/// (e.g. Jamba: SSM + attention interleaved) drop in as new variants.
pub enum Mixer<B: Backend> {
    /// Grouped-query attention with RoPE. `q_norm`/`k_norm` are Qwen3's
    /// per-head RMSNorm on Q and K applied before RoPE — `None` for LLaMA.
    Gqa {
        wq: B::Buffer,
        wk: B::Buffer,
        wv: B::Buffer,
        wo: B::Buffer,
        q_norm: Option<B::Buffer>,
        k_norm: Option<B::Buffer>,
    },
}

/// Channel-mixing block (the "FFN").
pub enum Ffn<B: Backend> {
    /// SwiGLU: `w2 @ (silu(w1 @ x) * (w3 @ x))`.
    SwiGlu {
        w1: B::Buffer,
        w2: B::Buffer,
        w3: B::Buffer,
    },
}

pub struct Layer<B: Backend> {
    pub attn_norm: B::Buffer,
    pub mixer: Mixer<B>,
    pub ffn_norm: B::Buffer,
    pub ffn: Ffn<B>,
}

/// Immutable model: weights + config. Shareable across sessions.
pub struct Transformer<B: Backend> {
    pub config: ModelConfig,
    pub token_embedding: B::Buffer,
    pub output_norm: B::Buffer,
    pub output_weight: B::Buffer,
    pub layers: Vec<Layer<B>>,
}

/// Per-request mutable state. One session = one conversation.
pub struct Session<B: Backend> {
    pub kv_cache: KVCache<B>,
    pub pos: usize,
    logits: B::Buffer,
    decode: Scratch<B>,
}

impl<B: Backend> Session<B> {
    pub fn new(backend: &B, config: &ModelConfig) -> Self {
        Session {
            kv_cache: KVCache::new(backend, config),
            pos: 0,
            logits: backend.alloc(&[config.vocab_size as u64], DType::BF16),
            decode: Scratch::new(backend, config, 1),
        }
    }

    pub fn logits(&self) -> &B::Buffer {
        &self.logits
    }
}

impl<B: Backend> Transformer<B> {
    /// Run `tokens` starting at `session.pos` through every layer and populate the KV cache.
    /// Advances `session.pos` by `tokens.len()`. When `want_logits` is true, leaves the
    /// next-token distribution from the *last* input token in `session.logits`.
    pub fn forward(
        &self,
        backend: &B,
        session: &mut Session<B>,
        tokens: &[u32],
        want_logits: bool,
    ) {
        if tokens.is_empty() {
            return;
        }
        assert!(
            session.pos + tokens.len() <= self.config.max_seq_len,
            "context overflow: pos={} + tokens={} > max_seq_len={}",
            session.pos, tokens.len(), self.config.max_seq_len,
        );

        let start_pos = session.pos;
        let (batch, tail) = if want_logits {
            (&tokens[..tokens.len() - 1], Some(tokens[tokens.len() - 1]))
        } else {
            (tokens, None)
        };

        if !batch.is_empty() {
            let mut prefill = Scratch::new(backend, &self.config, batch.len());
            self.run_layers(backend, &mut prefill, &mut session.kv_cache, batch, start_pos);
        }

        session.pos = start_pos + tokens.len();

        let Some(tail_token) = tail else { return };
        let s = &mut session.decode;
        self.run_layers(
            backend,
            s,
            &mut session.kv_cache,
            std::slice::from_ref(&tail_token),
            start_pos + batch.len(),
        );

        backend.rms_norm(&mut s.x_norm, &s.x, &self.output_norm, self.config.norm_eps);
        backend.matmul(&mut session.logits, &self.output_weight, &s.x_norm);
    }

    fn run_layers(
        &self,
        backend: &B,
        s: &mut Scratch<B>,
        kv_cache: &mut KVCache<B>,
        tokens: &[u32],
        start_pos: usize,
    ) {
        let cfg = &self.config;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;

        backend.embed(&mut s.x, &self.token_embedding, tokens);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            backend.rms_norm(&mut s.x_norm, &s.x, &layer.attn_norm, cfg.norm_eps);

            match &layer.mixer {
                Mixer::Gqa { wq, wk, wv, wo, q_norm, k_norm } => {
                    backend.matmul(&mut s.q, wq, &s.x_norm);
                    backend.matmul(&mut s.k, wk, &s.x_norm);
                    backend.matmul(&mut s.v, wv, &s.x_norm);
                    if let (Some(qn), Some(kn)) = (q_norm, k_norm) {
                        backend.rms_norm_heads(&mut s.q, qn, cfg.norm_eps);
                        backend.rms_norm_heads(&mut s.k, kn, cfg.norm_eps);
                    }
                    backend.rope(&mut s.q, &mut s.k, start_pos, cfg.head_dim, cfg.rope_theta);

                    backend.copy_into(&mut kv_cache.k_cache[layer_idx], &s.k, start_pos * kv_dim);
                    backend.copy_into(&mut kv_cache.v_cache[layer_idx], &s.v, start_pos * kv_dim);

                    backend.attention(
                        &mut s.attn_out, &s.q,
                        &kv_cache.k_cache[layer_idx], &kv_cache.v_cache[layer_idx],
                        start_pos, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                    );
                    backend.matmul(&mut s.wo_out, wo, &s.attn_out);
                }
            }
            backend.add(&mut s.x2, &s.x, &s.wo_out);

            backend.rms_norm(&mut s.x_norm, &s.x2, &layer.ffn_norm, cfg.norm_eps);
            match &layer.ffn {
                Ffn::SwiGlu { w1, w2, w3 } => {
                    backend.matmul(&mut s.gate, w1, &s.x_norm);
                    backend.matmul(&mut s.up, w3, &s.x_norm);
                    backend.silu(&mut s.gate);
                    backend.mul(&mut s.gate_up, &s.gate, &s.up);
                    backend.matmul(&mut s.ffn_out, w2, &s.gate_up);
                }
            }
            backend.add(&mut s.x, &s.x2, &s.ffn_out);
        }
    }
}

/// Activation buffers for one `run_layers` pass, sized to a fixed seq_len.
struct Scratch<B: Backend> {
    x: B::Buffer,
    x_norm: B::Buffer,
    q: B::Buffer,
    k: B::Buffer,
    v: B::Buffer,
    attn_out: B::Buffer,
    wo_out: B::Buffer,
    x2: B::Buffer,
    gate: B::Buffer,
    up: B::Buffer,
    gate_up: B::Buffer,
    ffn_out: B::Buffer,
}

impl<B: Backend> Scratch<B> {
    fn new(backend: &B, config: &ModelConfig, seq_len: usize) -> Self {
        let dim = config.dim as u64;
        let q_dim = (config.n_heads * config.head_dim) as u64;
        let kv_dim = (config.n_kv_heads * config.head_dim) as u64;
        let hidden = config.hidden_dim as u64;
        let n = seq_len as u64;
        let shape: &[u64] = if seq_len == 1 { &[dim] } else { &[dim, n] };
        let q_shape: &[u64] = if seq_len == 1 { &[q_dim] } else { &[q_dim, n] };
        let kv_shape: &[u64] = if seq_len == 1 { &[kv_dim] } else { &[kv_dim, n] };
        let h_shape: &[u64] = if seq_len == 1 { &[hidden] } else { &[hidden, n] };
        Scratch {
            x: backend.alloc(shape, DType::BF16),
            x_norm: backend.alloc(shape, DType::BF16),
            q: backend.alloc(q_shape, DType::BF16),
            k: backend.alloc(kv_shape, DType::BF16),
            v: backend.alloc(kv_shape, DType::BF16),
            attn_out: backend.alloc(q_shape, DType::BF16),
            wo_out: backend.alloc(shape, DType::BF16),
            x2: backend.alloc(shape, DType::BF16),
            gate: backend.alloc(h_shape, DType::BF16),
            up: backend.alloc(h_shape, DType::BF16),
            gate_up: backend.alloc(h_shape, DType::BF16),
            ffn_out: backend.alloc(shape, DType::BF16),
        }
    }
}
