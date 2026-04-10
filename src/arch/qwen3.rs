//! Qwen3: GQA + SwiGLU + per-head QK-norm before RoPE, split-half RoPE.

use crate::arch::{LoadError, Loader};
use crate::backend::Backend;
use crate::batch::Batch;
use crate::gguf::GgufFile;
use crate::kv_pool::PagedKVPool;
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
}

pub struct Qwen3Model<B: Backend> {
    config: Config,
    token_embedding: B::Buffer,
    output_norm: B::Buffer,
    output_weight: B::Buffer,
    layers: Vec<Layer<B>>,
    logits_buf: B::Buffer,
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

struct Scratch<B: Backend> {
    x: B::Buffer, x_norm: B::Buffer,
    q: B::Buffer, k: B::Buffer, v: B::Buffer,
    attn_out: B::Buffer, wo_out: B::Buffer,
    x2: B::Buffer,
    gate: B::Buffer, up: B::Buffer, gate_up: B::Buffer,
    ffn_out: B::Buffer,
    last_x: B::Buffer, last_x_norm: B::Buffer,
}

impl<B: Backend> Scratch<B> {
    fn new(backend: &B, c: &Config, n: usize, num_requests: usize) -> Self {
        let (d, q, kv, h) = (c.dim as u64, (c.n_heads * c.head_dim) as u64,
            (c.n_kv_heads * c.head_dim) as u64, c.hidden_dim as u64);
        let n = n as u64;
        let s = |base: u64| -> Vec<u64> { if n == 1 { vec![base] } else { vec![base, n] } };
        let r = num_requests as u64;
        let rs = |base: u64| -> Vec<u64> { if r == 1 { vec![base] } else { vec![base, r] } };
        Scratch {
            x: backend.alloc(&s(d), DType::BF16), x_norm: backend.alloc(&s(d), DType::BF16),
            q: backend.alloc(&s(q), DType::BF16), k: backend.alloc(&s(kv), DType::BF16),
            v: backend.alloc(&s(kv), DType::BF16), attn_out: backend.alloc(&s(q), DType::BF16),
            wo_out: backend.alloc(&s(d), DType::BF16), x2: backend.alloc(&s(d), DType::BF16),
            gate: backend.alloc(&s(h), DType::BF16), up: backend.alloc(&s(h), DType::BF16),
            gate_up: backend.alloc(&s(h), DType::BF16), ffn_out: backend.alloc(&s(d), DType::BF16),
            last_x: backend.alloc(&rs(d), DType::BF16), last_x_norm: backend.alloc(&rs(d), DType::BF16),
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

        let logits_buf = backend.alloc(&[config.vocab_size as u64], DType::BF16);
        l.print_stats();
        Ok(Qwen3Model { config, token_embedding, output_norm, output_weight, layers, logits_buf })
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

        Ok(Config { vocab_size, dim, n_layers, n_heads, n_kv_heads, head_dim, hidden_dim, norm_eps, rope_theta })
    }
}

impl<B: Backend> Model<B> for Qwen3Model<B> {
    fn forward(&mut self, backend: &B, pool: &mut PagedKVPool<B>, batch: &Batch<B>) {
        let c = &self.config;
        let kv_dim = c.n_kv_heads * c.head_dim;
        let mut s = Scratch::new(backend, c, batch.num_tokens, batch.num_requests);

        backend.embed(&mut s.x, &self.token_embedding, &batch.tokens);
        for (i, layer) in self.layers.iter().enumerate() {
            backend.rms_norm(&mut s.x_norm, &s.x, &layer.attn_norm, c.norm_eps);
            backend.matmul(&mut s.q, &layer.wq, &s.x_norm);
            backend.matmul(&mut s.k, &layer.wk, &s.x_norm);
            backend.matmul(&mut s.v, &layer.wv, &s.x_norm);
            backend.rms_norm_heads(&mut s.q, &layer.q_norm, c.norm_eps);
            backend.rms_norm_heads(&mut s.k, &layer.k_norm, c.norm_eps);
            backend.rope(&mut s.q, &mut s.k, &batch.positions, c.head_dim, c.rope_theta, RopeLayout::SplitHalf);
            backend.scatter_kv(&mut pool.k[i], &s.k, &batch.slot_mapping, kv_dim, batch.num_tokens);
            backend.scatter_kv(&mut pool.v[i], &s.v, &batch.slot_mapping, kv_dim, batch.num_tokens);
            backend.attention(
                &mut s.attn_out, &s.q, &pool.k[i], &pool.v[i],
                &batch.block_tables, &batch.query_starts, &batch.seq_lens,
                c.n_heads, c.n_kv_heads, c.head_dim, pool.block_size, batch.max_blocks_per_seq,
            );
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

        backend.gather(&mut s.last_x, &s.x, &batch.logit_indices, batch.num_requests);
        backend.rms_norm(&mut s.last_x_norm, &s.last_x, &self.output_norm, c.norm_eps);
        self.logits_buf = backend.alloc(&[c.vocab_size as u64, batch.num_requests as u64], DType::BF16);
        backend.matmul(&mut self.logits_buf, &self.output_weight, &s.last_x_norm);
    }

    fn logits(&self) -> &B::Buffer { &self.logits_buf }
}
