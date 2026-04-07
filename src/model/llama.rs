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

pub struct LlamaModel {
    pub config: ModelConfig,
    pub weights: LlamaWeights,
    pub kv_cache: KVCache,
    act: Activations,
}

impl LlamaModel {
    pub fn from_gguf(gguf: &GgufFile, backend: &dyn Backend) -> Self {
        let config = ModelConfig::from_gguf_metadata(&gguf.metadata);

        eprintln!(
            "Loading {} model: dim={}, layers={}, heads={}, kv_heads={}, vocab={}",
            config.architecture,
            config.dim,
            config.n_layers,
            config.n_heads,
            config.n_kv_heads,
            config.vocab_size
        );

        let load = |name: &str| -> DeviceBuffer {
            let tv = gguf
                .tensor_view(name)
                .unwrap_or_else(|_| panic!("missing tensor: {name}"));
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

        let kv_cache = KVCache::new(backend, &config);

        let dim = config.dim as u64;
        let kv_dim = (config.n_kv_heads * config.head_dim) as u64;
        let hidden_dim = config.hidden_dim as u64;

        let act = Activations {
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
        };

        LlamaModel {
            config,
            weights: LlamaWeights {
                token_embedding,
                output_norm,
                output_weight,
                layers,
            },
            kv_cache,
            act,
        }
    }

    /// Run one token through all transformer layers, populating the KV cache.
    /// Does NOT compute logits — use this for prefill tokens where logits aren't needed.
    pub fn forward_kv_only(&mut self, backend: &dyn Backend, token: u32, pos: usize) {
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;

        backend.embed(&mut self.act.x, &self.weights.token_embedding, token);

        for layer_idx in 0..self.config.n_layers {
            let layer = &self.weights.layers[layer_idx];

            // Attention
            backend.rms_norm(
                &mut self.act.x_norm,
                &self.act.x,
                &layer.attn_norm,
                self.config.norm_eps,
            );
            backend.matvec_mul(&mut self.act.q, &layer.wq, &self.act.x_norm);
            backend.matvec_mul(&mut self.act.k, &layer.wk, &self.act.x_norm);
            backend.matvec_mul(&mut self.act.v, &layer.wv, &self.act.x_norm);
            backend.rope(
                &mut self.act.q,
                &mut self.act.k,
                pos,
                head_dim,
                self.config.rope_theta,
            );

            self.kv_cache
                .store(backend, layer_idx, pos, &self.act.k, &self.act.v);

            backend.gqa_attention(
                &mut self.act.attn_out,
                &self.act.q,
                &self.kv_cache.k_cache[layer_idx],
                &self.kv_cache.v_cache[layer_idx],
                pos,
                n_heads,
                self.config.n_kv_heads,
                head_dim,
            );

            backend.matvec_mul(&mut self.act.wo_out, &layer.wo, &self.act.attn_out);
            backend.add(&mut self.act.x2, &self.act.x, &self.act.wo_out);

            // FFN
            backend.rms_norm(
                &mut self.act.x_norm,
                &self.act.x2,
                &layer.ffn_norm,
                self.config.norm_eps,
            );
            backend.matvec_mul(&mut self.act.gate, &layer.w1, &self.act.x_norm);
            backend.matvec_mul(&mut self.act.up, &layer.w3, &self.act.x_norm);
            backend.silu(&mut self.act.gate);
            backend.mul(&mut self.act.gate_up, &self.act.gate, &self.act.up);
            backend.matvec_mul(&mut self.act.ffn_out, &layer.w2, &self.act.gate_up);

            backend.add(&mut self.act.x, &self.act.x2, &self.act.ffn_out);
        }
    }

    /// Run a batch of tokens through all layers, storing KV cache.
    /// Does NOT compute logits — use this for prefill tokens.
    /// Uses batched GEMM for unquantized weights (BF16/F16/F32) where weight reuse
    /// across sequence positions helps. Falls back to sequential forward_kv_only for
    /// quantized weights where the optimized matvec kernel is faster.
    pub fn forward_prefill(&mut self, backend: &dyn Backend, tokens: &[u32], start_pos: usize) {
        let seq_len = tokens.len();
        if seq_len == 0 {
            return;
        }

        // For quantized weights, sequential matvec is faster than GEMM
        let use_gemm = matches!(
            self.weights.layers[0].wq.dtype,
            DType::F32 | DType::BF16 | DType::F16
        );
        if !use_gemm {
            for (i, &token) in tokens.iter().enumerate() {
                self.forward_kv_only(backend, token, start_pos + i);
            }
            return;
        }

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
            backend.copy_into(&mut self.kv_cache.k_cache[layer_idx], &k, kv_offset);
            backend.copy_into(&mut self.kv_cache.v_cache[layer_idx], &v, kv_offset);

            backend.gqa_attention_batch(
                &mut attn_out,
                &q,
                &self.kv_cache.k_cache[layer_idx],
                &self.kv_cache.v_cache[layer_idx],
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

    /// Run one token through the full model and return logits.
    pub fn forward(&mut self, backend: &dyn Backend, token: u32, pos: usize) -> Vec<f32> {
        self.forward_kv_only(backend, token, pos);
        backend.rms_norm(
            &mut self.act.x_norm,
            &self.act.x,
            &self.weights.output_norm,
            self.config.norm_eps,
        );
        backend.matvec_mul(
            &mut self.act.logits,
            &self.weights.output_weight,
            &self.act.x_norm,
        );
        backend.read_to_vec_f32(&self.act.logits)
    }
}
