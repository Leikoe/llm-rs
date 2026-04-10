//! Qwen3: GQA + SwiGLU + per-head QK-norm before RoPE, split-half RoPE.

use crate::arch::{LoadError, Loader};
use crate::backend::Backend;
use crate::gguf::GgufFile;
use crate::kv_cache::KVCache;
use crate::model::Model;
use crate::tensor::{DType, RopeLayout};

struct Config {
    vocab_size: usize,
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    hidden_dim: usize,
    norm_eps: f32,
    rope_theta: f32,
    max_seq_len: usize,
}

pub struct Qwen3Model<B: Backend> {
    config: Config,
    token_embedding: B::Buffer,
    output_norm: B::Buffer,
    output_weight: B::Buffer,
    layers: Vec<Layer<B>>,
}

struct Layer<B: Backend> {
    attn_norm: B::Buffer,
    wq: B::Buffer,
    wk: B::Buffer,
    wv: B::Buffer,
    wo: B::Buffer,
    q_norm: B::Buffer,
    k_norm: B::Buffer,
    ffn_norm: B::Buffer,
    w1: B::Buffer,
    w2: B::Buffer,
    w3: B::Buffer,
}

pub struct Session<B: Backend> {
    kv_cache: KVCache<B>,
    pos: usize,
    logits: B::Buffer,
    decode: Scratch<B>,
}

struct Scratch<B: Backend> {
    x: B::Buffer, x_norm: B::Buffer,
    q: B::Buffer, k: B::Buffer, v: B::Buffer,
    attn_out: B::Buffer, wo_out: B::Buffer,
    x2: B::Buffer,
    gate: B::Buffer, up: B::Buffer, gate_up: B::Buffer,
    ffn_out: B::Buffer,
}

impl<B: Backend> Scratch<B> {
    fn new(backend: &B, c: &Config, seq_len: usize) -> Self {
        let (d, q, kv, h) = (c.dim as u64, (c.n_heads * c.head_dim) as u64,
            (c.n_kv_heads * c.head_dim) as u64, c.hidden_dim as u64);
        let n = seq_len as u64;
        let s = |base: u64| -> Vec<u64> { if seq_len == 1 { vec![base] } else { vec![base, n] } };
        Scratch {
            x: backend.alloc(&s(d), DType::BF16), x_norm: backend.alloc(&s(d), DType::BF16),
            q: backend.alloc(&s(q), DType::BF16), k: backend.alloc(&s(kv), DType::BF16),
            v: backend.alloc(&s(kv), DType::BF16), attn_out: backend.alloc(&s(q), DType::BF16),
            wo_out: backend.alloc(&s(d), DType::BF16), x2: backend.alloc(&s(d), DType::BF16),
            gate: backend.alloc(&s(h), DType::BF16), up: backend.alloc(&s(h), DType::BF16),
            gate_up: backend.alloc(&s(h), DType::BF16), ffn_out: backend.alloc(&s(d), DType::BF16),
        }
    }
}

impl<B: Backend> Qwen3Model<B> {
    pub fn load(gguf: &GgufFile, backend: &B) -> Result<Self, LoadError> {
        let config = Self::parse_config(gguf)?;
        eprintln!(
            "qwen3 model: dim={}, layers={}, heads={}, kv_heads={}, vocab={}",
            config.dim, config.n_layers, config.n_heads, config.n_kv_heads, config.vocab_size,
        );

        let mut l = Loader::new(gguf, backend);
        let token_embedding = l.req("token_embd.weight");
        let output_weight = l.opt("output.weight").unwrap_or_else(|| l.req("token_embd.weight"));
        let output_norm = l.req("output_norm.weight");

        let layers = (0..config.n_layers)
            .map(|i| Layer {
                attn_norm: l.req(&format!("blk.{i}.attn_norm.weight")),
                wq: l.req(&format!("blk.{i}.attn_q.weight")),
                wk: l.req(&format!("blk.{i}.attn_k.weight")),
                wv: l.req(&format!("blk.{i}.attn_v.weight")),
                wo: l.req(&format!("blk.{i}.attn_output.weight")),
                q_norm: l.req(&format!("blk.{i}.attn_q_norm.weight")),
                k_norm: l.req(&format!("blk.{i}.attn_k_norm.weight")),
                ffn_norm: l.req(&format!("blk.{i}.ffn_norm.weight")),
                w1: l.req(&format!("blk.{i}.ffn_gate.weight")),
                w2: l.req(&format!("blk.{i}.ffn_down.weight")),
                w3: l.req(&format!("blk.{i}.ffn_up.weight")),
            })
            .collect();

        l.print_stats();
        Ok(Qwen3Model { config, token_embedding, output_norm, output_weight, layers })
    }

    fn parse_config(gguf: &GgufFile) -> Result<Config, LoadError> {
        let m = &gguf.metadata;
        let req = |key: &str| m.get(key).and_then(|v| v.as_u32())
            .ok_or_else(|| LoadError::missing(key));
        let get_u32 = |key: &str| m.get(key).and_then(|v| v.as_u32());
        let get_f32 = |key: &str| m.get(key).and_then(|v| v.as_f32());

        let vocab_size = m.get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_string_array())
            .ok_or_else(|| LoadError::missing("tokenizer.ggml.tokens"))?.len();

        let dim = req("qwen3.embedding_length")? as usize;
        let n_layers = req("qwen3.block_count")? as usize;
        let n_heads = req("qwen3.attention.head_count")? as usize;
        let n_kv_heads = get_u32("qwen3.attention.head_count_kv").unwrap_or(n_heads as u32) as usize;
        let head_dim = get_u32("qwen3.attention.key_length").map(|v| v as usize).unwrap_or(dim / n_heads);
        let hidden_dim = req("qwen3.feed_forward_length")? as usize;
        let norm_eps = get_f32("qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let rope_theta = get_f32("qwen3.rope.freq_base").unwrap_or(10000.0);
        let model_ctx = get_u32("qwen3.context_length").unwrap_or(4096) as usize;
        let max_seq_len = model_ctx.min(4096);
        if model_ctx > max_seq_len {
            eprintln!("Note: model context {model_ctx} capped to {max_seq_len}");
        }

        Ok(Config { vocab_size, dim, n_layers, n_heads, n_kv_heads, head_dim, hidden_dim, norm_eps, rope_theta, max_seq_len })
    }
}

impl<B: Backend> Model<B> for Qwen3Model<B> {
    type Session = Session<B>;

    fn new_session(&self, backend: &B) -> Session<B> {
        let c = &self.config;
        let kv_dim = c.n_kv_heads * c.head_dim;
        Session {
            kv_cache: KVCache::new(backend, c.n_layers, kv_dim, c.max_seq_len),
            pos: 0,
            logits: backend.alloc(&[c.vocab_size as u64], DType::BF16),
            decode: Scratch::new(backend, c, 1),
        }
    }

    fn logits<'a>(&self, session: &'a Session<B>) -> &'a B::Buffer { &session.logits }

    fn forward(&self, backend: &B, session: &mut Session<B>, tokens: &[u32], want_logits: bool) {
        if tokens.is_empty() { return; }
        let c = &self.config;
        let kv_dim = c.n_kv_heads * c.head_dim;
        let start_pos = session.pos;

        let (batch, tail) = if want_logits {
            (&tokens[..tokens.len() - 1], Some(tokens[tokens.len() - 1]))
        } else {
            (tokens, None)
        };
        if !batch.is_empty() {
            let mut s = Scratch::new(backend, c, batch.len());
            self.run_layers(backend, &mut s, &mut session.kv_cache, batch, start_pos, kv_dim);
        }
        session.pos = start_pos + tokens.len();
        let Some(tail_token) = tail else { return };
        let s = &mut session.decode;
        self.run_layers(backend, s, &mut session.kv_cache,
            std::slice::from_ref(&tail_token), start_pos + batch.len(), kv_dim);
        backend.rms_norm(&mut s.x_norm, &s.x, &self.output_norm, c.norm_eps);
        backend.matmul(&mut session.logits, &self.output_weight, &s.x_norm);
    }
}

impl<B: Backend> Qwen3Model<B> {
    fn run_layers(
        &self, backend: &B, s: &mut Scratch<B>, kv_cache: &mut KVCache<B>,
        tokens: &[u32], start_pos: usize, kv_dim: usize,
    ) {
        let c = &self.config;
        backend.embed(&mut s.x, &self.token_embedding, tokens);
        for (i, layer) in self.layers.iter().enumerate() {
            backend.rms_norm(&mut s.x_norm, &s.x, &layer.attn_norm, c.norm_eps);
            backend.matmul(&mut s.q, &layer.wq, &s.x_norm);
            backend.matmul(&mut s.k, &layer.wk, &s.x_norm);
            backend.matmul(&mut s.v, &layer.wv, &s.x_norm);
            backend.rms_norm_heads(&mut s.q, &layer.q_norm, c.norm_eps);
            backend.rms_norm_heads(&mut s.k, &layer.k_norm, c.norm_eps);
            backend.rope(&mut s.q, &mut s.k, start_pos, c.head_dim, c.rope_theta, RopeLayout::SplitHalf);
            backend.copy_into(&mut kv_cache.k_cache[i], &s.k, start_pos * kv_dim);
            backend.copy_into(&mut kv_cache.v_cache[i], &s.v, start_pos * kv_dim);
            backend.attention(&mut s.attn_out, &s.q, &kv_cache.k_cache[i], &kv_cache.v_cache[i],
                start_pos, c.n_heads, c.n_kv_heads, c.head_dim);
            backend.matmul(&mut s.wo_out, &layer.wo, &s.attn_out);
            backend.add(&mut s.x2, &s.x, &s.wo_out);
            backend.rms_norm(&mut s.x_norm, &s.x2, &layer.ffn_norm, c.norm_eps);
            backend.matmul(&mut s.gate, &layer.w1, &s.x_norm);
            backend.matmul(&mut s.up, &layer.w3, &s.x_norm);
            backend.silu(&mut s.gate);
            backend.mul(&mut s.gate_up, &s.gate, &s.up);
            backend.matmul(&mut s.ffn_out, &layer.w2, &s.gate_up);
            backend.add(&mut s.x, &s.x2, &s.ffn_out);
        }
    }
}
