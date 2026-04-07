use crate::backend::Backend;
use crate::gguf::GgufFile;
use crate::kv_cache::KVCache;
use crate::model::ModelConfig;
use crate::tensor::DType;

pub struct LlamaLayerWeights<B: Backend> {
    pub attn_norm: B::Buffer,
    pub wq: B::Buffer,
    pub wk: B::Buffer,
    pub wv: B::Buffer,
    pub wo: B::Buffer,
    pub ffn_norm: B::Buffer,
    pub w1: B::Buffer, // gate
    pub w2: B::Buffer, // down
    pub w3: B::Buffer, // up
}

pub struct LlamaWeights<B: Backend> {
    pub token_embedding: B::Buffer,
    pub output_norm: B::Buffer,
    pub output_weight: B::Buffer,
    pub layers: Vec<LlamaLayerWeights<B>>,
}

/// Immutable model: weights + config. Shareable across sessions.
pub struct LlamaModel<B: Backend> {
    pub config: ModelConfig,
    pub weights: LlamaWeights<B>,
}

/// Per-request mutable state. One session = one conversation.
///
/// Owns the persistent decode scratch — 12 activation buffers sized for one
/// token. Decode steps mutate them in place; prefill steps allocate a separate
/// batch-sized scratch on the stack and discard it. This is vLLM V1's
/// persistent-batch idea at our scale: never allocate on the per-token path.
pub struct Session<B: Backend> {
    pub kv_cache: KVCache<B>,
    pub pos: usize,
    /// On-device logits for the most recent `forward(want_logits=true)` call.
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

impl<B: Backend> LlamaModel<B> {
    pub fn from_gguf(gguf: &GgufFile, backend: &B) -> Self {
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
        let mut load = |name: &str| -> B::Buffer {
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

    /// Run `tokens` starting at `session.pos` through every layer and populate the KV cache.
    /// Advances `session.pos` by `tokens.len()`. When `want_logits` is true, leaves the
    /// next-token distribution from the *last* input token in `session.logits`.
    ///
    /// Decode (`tokens.len() == 1`) and prefill (`tokens.len() > 1`) run the same body —
    /// scratch is sized to the batch, the backend dispatches GEMV/GEMM and flash/causal
    /// attention from the input shape. To produce logits we run a 1-column tail step so the
    /// output projection sees a 1D hidden state.
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

        backend.rms_norm(&mut s.x_norm, &s.x, &self.weights.output_norm, self.config.norm_eps);
        backend.matmul(&mut session.logits, &self.weights.output_weight, &s.x_norm);
    }

    /// Run `tokens` through every layer using `s` as scratch. Backend ops are shape-agnostic —
    /// `tokens.len() == 1` (decode) and `> 1` (prefill) take exactly the same call.
    fn run_layers(
        &self,
        backend: &B,
        s: &mut Scratch<B>,
        kv_cache: &mut KVCache<B>,
        tokens: &[u32],
        start_pos: usize,
    ) {
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;

        backend.embed(&mut s.x, &self.weights.token_embedding, tokens);

        for layer_idx in 0..self.config.n_layers {
            let layer = &self.weights.layers[layer_idx];

            backend.rms_norm(&mut s.x_norm, &s.x, &layer.attn_norm, self.config.norm_eps);
            backend.matmul(&mut s.q, &layer.wq, &s.x_norm);
            backend.matmul(&mut s.k, &layer.wk, &s.x_norm);
            backend.matmul(&mut s.v, &layer.wv, &s.x_norm);
            backend.rope(&mut s.q, &mut s.k, start_pos, self.config.head_dim, self.config.rope_theta);

            backend.copy_into(&mut kv_cache.k_cache[layer_idx], &s.k, start_pos * kv_dim);
            backend.copy_into(&mut kv_cache.v_cache[layer_idx], &s.v, start_pos * kv_dim);

            backend.attention(
                &mut s.attn_out,
                &s.q,
                &kv_cache.k_cache[layer_idx],
                &kv_cache.v_cache[layer_idx],
                start_pos,
                self.config.n_heads,
                self.config.n_kv_heads,
                self.config.head_dim,
            );

            backend.matmul(&mut s.wo_out, &layer.wo, &s.attn_out);
            backend.add(&mut s.x2, &s.x, &s.wo_out);

            backend.rms_norm(&mut s.x_norm, &s.x2, &layer.ffn_norm, self.config.norm_eps);
            backend.matmul(&mut s.gate, &layer.w1, &s.x_norm);
            backend.matmul(&mut s.up, &layer.w3, &s.x_norm);
            backend.silu(&mut s.gate);
            backend.mul(&mut s.gate_up, &s.gate, &s.up);
            backend.matmul(&mut s.ffn_out, &layer.w2, &s.gate_up);

            backend.add(&mut s.x, &s.x2, &s.ffn_out);
        }
    }
}

/// Activation buffers for one `run_layers` pass, sized to a fixed seq_len.
/// The decode instance lives on `Session` for the lifetime of the conversation;
/// prefill instances are short-lived (one per `forward` with a batch).
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
        let kv_dim = (config.n_kv_heads * config.head_dim) as u64;
        let hidden = config.hidden_dim as u64;
        let n = seq_len as u64;
        let shape: &[u64] = if seq_len == 1 { &[dim] } else { &[dim, n] };
        let kv_shape: &[u64] = if seq_len == 1 { &[kv_dim] } else { &[kv_dim, n] };
        let h_shape: &[u64] = if seq_len == 1 { &[hidden] } else { &[hidden, n] };
        Scratch {
            x: backend.alloc(shape, DType::BF16),
            x_norm: backend.alloc(shape, DType::BF16),
            q: backend.alloc(shape, DType::BF16),
            k: backend.alloc(kv_shape, DType::BF16),
            v: backend.alloc(kv_shape, DType::BF16),
            attn_out: backend.alloc(shape, DType::BF16),
            wo_out: backend.alloc(shape, DType::BF16),
            x2: backend.alloc(shape, DType::BF16),
            gate: backend.alloc(h_shape, DType::BF16),
            up: backend.alloc(h_shape, DType::BF16),
            gate_up: backend.alloc(h_shape, DType::BF16),
            ffn_out: backend.alloc(shape, DType::BF16),
        }
    }
}
