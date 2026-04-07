use crate::backend::{Backend, DeviceBuffer};
use crate::gguf::GgufFile;
use crate::kv_cache::KVCache;
use crate::model::ModelConfig;
use crate::tensor::DType;

pub struct LlamaLayerWeights {
    pub attn_norm: DeviceBuffer,
    pub wq: DeviceBuffer,
    pub wk: DeviceBuffer,
    pub wv: DeviceBuffer,
    pub wo: DeviceBuffer,
    pub ffn_norm: DeviceBuffer,
    pub w1: DeviceBuffer, // gate
    pub w2: DeviceBuffer, // down
    pub w3: DeviceBuffer, // up
}

pub struct LlamaWeights {
    pub token_embedding: DeviceBuffer,
    pub output_norm: DeviceBuffer,
    pub output_weight: DeviceBuffer,
    pub layers: Vec<LlamaLayerWeights>,
}

/// Per-step scratch buffers, sized for one token. Reused across every forward call.
struct Activations {
    x: DeviceBuffer,
    x_norm: DeviceBuffer,
    q: DeviceBuffer,
    k: DeviceBuffer,
    v: DeviceBuffer,
    attn_out: DeviceBuffer,
    wo_out: DeviceBuffer,
    x2: DeviceBuffer,
    gate: DeviceBuffer,
    up: DeviceBuffer,
    gate_up: DeviceBuffer,
    ffn_out: DeviceBuffer,
    logits: DeviceBuffer,
}

impl Activations {
    fn new(backend: &dyn Backend, config: &ModelConfig) -> Self {
        let dim = config.dim as u64;
        let kv_dim = (config.n_kv_heads * config.head_dim) as u64;
        let hidden_dim = config.hidden_dim as u64;
        Activations {
            x: backend.alloc(&[dim], DType::BF16),
            x_norm: backend.alloc(&[dim], DType::BF16),
            q: backend.alloc(&[dim], DType::BF16),
            k: backend.alloc(&[kv_dim], DType::BF16),
            v: backend.alloc(&[kv_dim], DType::BF16),
            attn_out: backend.alloc(&[dim], DType::BF16),
            wo_out: backend.alloc(&[dim], DType::BF16),
            x2: backend.alloc(&[dim], DType::BF16),
            gate: backend.alloc(&[hidden_dim], DType::BF16),
            up: backend.alloc(&[hidden_dim], DType::BF16),
            gate_up: backend.alloc(&[hidden_dim], DType::BF16),
            ffn_out: backend.alloc(&[dim], DType::BF16),
            logits: backend.alloc(&[config.vocab_size as u64], DType::BF16),
        }
    }
}

/// Immutable model: weights + config. Shareable across sessions.
pub struct LlamaModel {
    pub config: ModelConfig,
    pub weights: LlamaWeights,
}

/// Per-request mutable state: KV cache, current position, scratch activations.
/// One session = one conversation. Independent of the model.
pub struct Session {
    pub kv_cache: KVCache,
    pub pos: usize,
    act: Activations,
}

impl Session {
    pub fn new(backend: &dyn Backend, config: &ModelConfig) -> Self {
        Session {
            kv_cache: KVCache::new(backend, config),
            pos: 0,
            act: Activations::new(backend, config),
        }
    }
}

impl LlamaModel {
    pub fn from_gguf(gguf: &GgufFile, backend: &dyn Backend) -> Self {
        let config = ModelConfig::from_gguf_metadata(&gguf.metadata);

        eprintln!(
            "{} model: dim={}, layers={}, heads={}, kv_heads={}, vocab={}",
            config.architecture,
            config.dim,
            config.n_layers,
            config.n_heads,
            config.n_kv_heads,
            config.vocab_size
        );

        let upload_start = std::time::Instant::now();
        let mut bytes_uploaded: usize = 0;
        let mut load = |name: &str| -> DeviceBuffer {
            let tv = gguf
                .tensor_view(name)
                .unwrap_or_else(|_| panic!("missing tensor: {name}"));
            bytes_uploaded += tv.data.len();
            backend.upload_tensor(&tv)
        };

        let token_embedding = load("token_embd.weight");
        let output_weight = if gguf.tensor_view("output.weight").is_ok() {
            load("output.weight")
        } else {
            load("token_embd.weight")
        };
        let output_norm = load("output_norm.weight");

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(LlamaLayerWeights {
                attn_norm: load(&format!("blk.{i}.attn_norm.weight")),
                wq: load(&format!("blk.{i}.attn_q.weight")),
                wk: load(&format!("blk.{i}.attn_k.weight")),
                wv: load(&format!("blk.{i}.attn_v.weight")),
                wo: load(&format!("blk.{i}.attn_output.weight")),
                ffn_norm: load(&format!("blk.{i}.ffn_norm.weight")),
                w1: load(&format!("blk.{i}.ffn_gate.weight")),
                w2: load(&format!("blk.{i}.ffn_down.weight")),
                w3: load(&format!("blk.{i}.ffn_up.weight")),
            });
        }

        let elapsed = upload_start.elapsed().as_secs_f64();
        let gb = bytes_uploaded as f64 / (1024.0 * 1024.0 * 1024.0);
        eprintln!("Uploaded model weights to backend: {:.2} GB in {:.3}s ({:.1} GB/s)", gb, elapsed, gb / elapsed);

        LlamaModel {
            config,
            weights: LlamaWeights {
                token_embedding,
                output_norm,
                output_weight,
                layers,
            },
        }
    }

    /// Run `tokens` starting at `session.pos` through all layers and populate the KV cache.
    /// Advances `session.pos` by `tokens.len()`. Returns logits for the last token if
    /// `want_logits` is true, otherwise `None`.
    ///
    /// A step is just `{tokens}` against a session. Decode is `tokens.len() == 1`. Prefill is
    /// `tokens.len() > 1`. When logits are requested the last token always runs through the
    /// sequential path so its hidden state lands in the persistent activation buffers.
    pub fn forward(
        &self,
        backend: &dyn Backend,
        session: &mut Session,
        tokens: &[u32],
        want_logits: bool,
    ) -> Option<Vec<f32>> {
        if tokens.is_empty() {
            return None;
        }

        let start_pos = session.pos;

        // Batched GEMM is only a win for unquantized weights with multiple tokens.
        let unquantized = matches!(
            self.weights.layers[0].wq.dtype,
            DType::F32 | DType::BF16 | DType::F16
        );

        // If logits are requested, peel the last token off so the sequential path
        // leaves its hidden state in session.act.x for the output projection.
        let (batch, tail) = if want_logits {
            (&tokens[..tokens.len() - 1], Some(tokens[tokens.len() - 1]))
        } else {
            (tokens, None)
        };

        if !batch.is_empty() {
            if unquantized && batch.len() > 1 {
                self.forward_batch_gemm(backend, session, batch, start_pos);
            } else {
                for (i, &t) in batch.iter().enumerate() {
                    self.forward_one(backend, session, t, start_pos + i);
                }
            }
        }

        session.pos = start_pos + tokens.len();

        let tail_token = tail?;
        let tail_pos = start_pos + batch.len();
        self.forward_one(backend, session, tail_token, tail_pos);

        backend.rms_norm(
            &mut session.act.x_norm,
            &session.act.x,
            &self.weights.output_norm,
            self.config.norm_eps,
        );
        backend.matvec_mul(
            &mut session.act.logits,
            &self.weights.output_weight,
            &session.act.x_norm,
        );
        Some(backend.read_to_vec_f32(&session.act.logits))
    }

    /// Run one token through all transformer layers, populating the KV cache.
    /// Leaves the post-FFN hidden state in `session.act.x`.
    fn forward_one(&self, backend: &dyn Backend, session: &mut Session, token: u32, pos: usize) {
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        // Split-borrow disjoint fields so we can pass &act.X and &mut kv_cache together.
        let act = &mut session.act;
        let kv_cache = &mut session.kv_cache;

        backend.embed(&mut act.x, &self.weights.token_embedding, token);

        for layer_idx in 0..self.config.n_layers {
            let layer = &self.weights.layers[layer_idx];

            // Attention
            backend.rms_norm(
                &mut act.x_norm,
                &act.x,
                &layer.attn_norm,
                self.config.norm_eps,
            );
            backend.matvec_mul(&mut act.q, &layer.wq, &act.x_norm);
            backend.matvec_mul(&mut act.k, &layer.wk, &act.x_norm);
            backend.matvec_mul(&mut act.v, &layer.wv, &act.x_norm);
            backend.rope(
                &mut act.q,
                &mut act.k,
                pos,
                head_dim,
                self.config.rope_theta,
            );

            kv_cache.store(backend, layer_idx, pos, &act.k, &act.v);

            backend.gqa_attention(
                &mut act.attn_out,
                &act.q,
                &kv_cache.k_cache[layer_idx],
                &kv_cache.v_cache[layer_idx],
                pos,
                n_heads,
                self.config.n_kv_heads,
                head_dim,
            );

            backend.matvec_mul(&mut act.wo_out, &layer.wo, &act.attn_out);
            backend.add(&mut act.x2, &act.x, &act.wo_out);

            // FFN
            backend.rms_norm(
                &mut act.x_norm,
                &act.x2,
                &layer.ffn_norm,
                self.config.norm_eps,
            );
            backend.matvec_mul(&mut act.gate, &layer.w1, &act.x_norm);
            backend.matvec_mul(&mut act.up, &layer.w3, &act.x_norm);
            backend.silu(&mut act.gate);
            backend.mul(&mut act.gate_up, &act.gate, &act.up);
            backend.matvec_mul(&mut act.ffn_out, &layer.w2, &act.gate_up);

            backend.add(&mut act.x, &act.x2, &act.ffn_out);
        }
    }

    /// Run a batch of tokens through all layers via batched GEMM, storing KV cache.
    /// Caller is responsible for choosing this path only when it's actually faster
    /// (unquantized weights, seq_len > 1).
    fn forward_batch_gemm(
        &self,
        backend: &dyn Backend,
        session: &mut Session,
        tokens: &[u32],
        start_pos: usize,
    ) {
        let seq_len = tokens.len();
        let dim = self.config.dim as u64;
        let kv_dim = (self.config.n_kv_heads * self.config.head_dim) as u64;
        let hidden_dim = self.config.hidden_dim as u64;
        let n = seq_len as u64;

        let mut x = backend.alloc(&[dim, n], DType::BF16);
        let mut x_norm = backend.alloc(&[dim, n], DType::BF16);
        let mut q = backend.alloc(&[dim, n], DType::BF16);
        let mut k = backend.alloc(&[kv_dim, n], DType::BF16);
        let mut v = backend.alloc(&[kv_dim, n], DType::BF16);
        let mut attn_out = backend.alloc(&[dim, n], DType::BF16);
        let mut wo_out = backend.alloc(&[dim, n], DType::BF16);
        let mut x2 = backend.alloc(&[dim, n], DType::BF16);
        let mut gate = backend.alloc(&[hidden_dim, n], DType::BF16);
        let mut up = backend.alloc(&[hidden_dim, n], DType::BF16);
        let mut gate_up = backend.alloc(&[hidden_dim, n], DType::BF16);
        let mut ffn_out = backend.alloc(&[dim, n], DType::BF16);

        backend.embed_batch(&mut x, &self.weights.token_embedding, tokens);

        for layer_idx in 0..self.config.n_layers {
            let layer = &self.weights.layers[layer_idx];

            backend.rms_norm_batch(
                &mut x_norm,
                &x,
                &layer.attn_norm,
                self.config.norm_eps,
                seq_len,
            );
            backend.matmul(&mut q, &layer.wq, &x_norm);
            backend.matmul(&mut k, &layer.wk, &x_norm);
            backend.matmul(&mut v, &layer.wv, &x_norm);
            backend.rope_batch(
                &mut q,
                &mut k,
                start_pos,
                seq_len,
                self.config.head_dim,
                self.config.rope_theta,
            );

            let kv_offset = start_pos * kv_dim as usize;
            backend.copy_into(&mut session.kv_cache.k_cache[layer_idx], &k, kv_offset);
            backend.copy_into(&mut session.kv_cache.v_cache[layer_idx], &v, kv_offset);

            backend.gqa_attention_batch(
                &mut attn_out,
                &q,
                &session.kv_cache.k_cache[layer_idx],
                &session.kv_cache.v_cache[layer_idx],
                start_pos,
                seq_len,
                self.config.n_heads,
                self.config.n_kv_heads,
                self.config.head_dim,
            );

            backend.matmul(&mut wo_out, &layer.wo, &attn_out);
            backend.add(&mut x2, &x, &wo_out);

            backend.rms_norm_batch(
                &mut x_norm,
                &x2,
                &layer.ffn_norm,
                self.config.norm_eps,
                seq_len,
            );
            backend.matmul(&mut gate, &layer.w1, &x_norm);
            backend.matmul(&mut up, &layer.w3, &x_norm);
            backend.silu(&mut gate);
            backend.mul(&mut gate_up, &gate, &up);
            backend.matmul(&mut ffn_out, &layer.w2, &gate_up);

            backend.add(&mut x, &x2, &ffn_out);
        }
    }
}
