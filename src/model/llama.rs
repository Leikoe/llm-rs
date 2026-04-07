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

    /// On-device logits buffer for the most recent forward call. Only valid
    /// when forward was called with `want_logits = true`.
    pub fn logits(&self) -> &DeviceBuffer {
        &self.act.logits
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

    /// Run `tokens` starting at `session.pos` through every layer and populate the KV cache.
    /// Advances `session.pos` by `tokens.len()`. When `want_logits` is true, populates
    /// `session.act.logits` with the next-token distribution from the *last* input token.
    ///
    /// Decode (`tokens.len() == 1`) reuses the persistent 1-column scratch in `session.act`.
    /// Prefill (`tokens.len() > 1`) allocates a wider scratch sized to the batch and runs the
    /// same layer body — the backend dispatches GEMV vs GEMM and flash vs causal attention
    /// from the input shape. To produce logits we always finish with a 1-column step using
    /// `session.act` so the output projection sees a 1D hidden state.
    pub fn forward(
        &self,
        backend: &dyn Backend,
        session: &mut Session,
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

        // If logits are requested, peel the last token off so the final layer pass
        // runs through `session.act` (1-column scratch) and leaves its hidden state
        // in `session.act.x` for the output projection.
        let (batch, tail) = if want_logits {
            (&tokens[..tokens.len() - 1], Some(tokens[tokens.len() - 1]))
        } else {
            (tokens, None)
        };

        if !batch.is_empty() {
            let mut scratch = Scratch::alloc(backend, &self.config, batch.len());
            self.run_layers(backend, &mut scratch, &mut session.kv_cache, batch, start_pos);
        }

        session.pos = start_pos + tokens.len();

        let Some(tail_token) = tail else { return };
        let tail_pos = start_pos + batch.len();
        let mut scratch = Scratch::from_act(&mut session.act);
        self.run_layers(
            backend,
            &mut scratch,
            &mut session.kv_cache,
            std::slice::from_ref(&tail_token),
            tail_pos,
        );

        backend.rms_norm(
            &mut session.act.x_norm,
            &session.act.x,
            &self.weights.output_norm,
            self.config.norm_eps,
        );
        backend.matmul(
            &mut session.act.logits,
            &self.weights.output_weight,
            &session.act.x_norm,
        );
    }

    /// Run `tokens` (length 1 for decode, >1 for prefill) through every layer using `s` as
    /// scratch. Backend ops are shape-agnostic — the same calls work in both modes.
    fn run_layers(
        &self,
        backend: &dyn Backend,
        s: &mut Scratch<'_>,
        kv_cache: &mut KVCache,
        tokens: &[u32],
        start_pos: usize,
    ) {
        let kv_dim = self.config.n_kv_heads * self.config.head_dim;

        backend.embed(s.x, &self.weights.token_embedding, tokens);

        for layer_idx in 0..self.config.n_layers {
            let layer = &self.weights.layers[layer_idx];

            // Attention
            backend.rms_norm(s.x_norm, s.x, &layer.attn_norm, self.config.norm_eps);
            backend.matmul(s.q, &layer.wq, s.x_norm);
            backend.matmul(s.k, &layer.wk, s.x_norm);
            backend.matmul(s.v, &layer.wv, s.x_norm);
            backend.rope(s.q, s.k, start_pos, self.config.head_dim, self.config.rope_theta);

            backend.copy_into(&mut kv_cache.k_cache[layer_idx], s.k, start_pos * kv_dim);
            backend.copy_into(&mut kv_cache.v_cache[layer_idx], s.v, start_pos * kv_dim);

            backend.attention(
                s.attn_out,
                s.q,
                &kv_cache.k_cache[layer_idx],
                &kv_cache.v_cache[layer_idx],
                start_pos,
                self.config.n_heads,
                self.config.n_kv_heads,
                self.config.head_dim,
            );

            backend.matmul(s.wo_out, &layer.wo, s.attn_out);
            backend.add(s.x2, s.x, s.wo_out);

            // FFN
            backend.rms_norm(s.x_norm, s.x2, &layer.ffn_norm, self.config.norm_eps);
            backend.matmul(s.gate, &layer.w1, s.x_norm);
            backend.matmul(s.up, &layer.w3, s.x_norm);
            backend.silu(s.gate);
            backend.mul(s.gate_up, s.gate, s.up);
            backend.matmul(s.ffn_out, &layer.w2, s.gate_up);

            backend.add(s.x, s.x2, s.ffn_out);
        }
    }
}

/// View over the per-step scratch buffers needed by `run_layers`. Either borrows
/// `Activations` (decode, persistent) or owns freshly-allocated 2D buffers (prefill).
struct Scratch<'a> {
    x: &'a mut DeviceBuffer,
    x_norm: &'a mut DeviceBuffer,
    q: &'a mut DeviceBuffer,
    k: &'a mut DeviceBuffer,
    v: &'a mut DeviceBuffer,
    attn_out: &'a mut DeviceBuffer,
    wo_out: &'a mut DeviceBuffer,
    x2: &'a mut DeviceBuffer,
    gate: &'a mut DeviceBuffer,
    up: &'a mut DeviceBuffer,
    gate_up: &'a mut DeviceBuffer,
    ffn_out: &'a mut DeviceBuffer,
    _own: Option<Box<OwnedScratch>>,
}

struct OwnedScratch {
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
}

impl<'a> Scratch<'a> {
    fn from_act(act: &'a mut Activations) -> Scratch<'a> {
        Scratch {
            x: &mut act.x,
            x_norm: &mut act.x_norm,
            q: &mut act.q,
            k: &mut act.k,
            v: &mut act.v,
            attn_out: &mut act.attn_out,
            wo_out: &mut act.wo_out,
            x2: &mut act.x2,
            gate: &mut act.gate,
            up: &mut act.up,
            gate_up: &mut act.gate_up,
            ffn_out: &mut act.ffn_out,
            _own: None,
        }
    }

    fn alloc(backend: &dyn Backend, config: &ModelConfig, seq_len: usize) -> Scratch<'a> {
        let dim = config.dim as u64;
        let kv_dim = (config.n_kv_heads * config.head_dim) as u64;
        let hidden = config.hidden_dim as u64;
        let n = seq_len as u64;
        let mut owned = Box::new(OwnedScratch {
            x: backend.alloc(&[dim, n], DType::BF16),
            x_norm: backend.alloc(&[dim, n], DType::BF16),
            q: backend.alloc(&[dim, n], DType::BF16),
            k: backend.alloc(&[kv_dim, n], DType::BF16),
            v: backend.alloc(&[kv_dim, n], DType::BF16),
            attn_out: backend.alloc(&[dim, n], DType::BF16),
            wo_out: backend.alloc(&[dim, n], DType::BF16),
            x2: backend.alloc(&[dim, n], DType::BF16),
            gate: backend.alloc(&[hidden, n], DType::BF16),
            up: backend.alloc(&[hidden, n], DType::BF16),
            gate_up: backend.alloc(&[hidden, n], DType::BF16),
            ffn_out: backend.alloc(&[dim, n], DType::BF16),
        });
        // Reborrow each field through the box's stable address. SAFETY: the box
        // outlives the returned Scratch (it's stored in `_own`) and the references
        // are disjoint, so this is sound.
        let p = &mut *owned as *mut OwnedScratch;
        unsafe {
            Scratch {
                x: &mut (*p).x,
                x_norm: &mut (*p).x_norm,
                q: &mut (*p).q,
                k: &mut (*p).k,
                v: &mut (*p).v,
                attn_out: &mut (*p).attn_out,
                wo_out: &mut (*p).wo_out,
                x2: &mut (*p).x2,
                gate: &mut (*p).gate,
                up: &mut (*p).up,
                gate_up: &mut (*p).gate_up,
                ffn_out: &mut (*p).ffn_out,
                _own: Some(owned),
            }
        }
    }
}
