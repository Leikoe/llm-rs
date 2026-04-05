use crate::backend::cpu::write_f32_to_buffer;
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
            config.architecture, config.dim, config.n_layers, config.n_heads,
            config.n_kv_heads, config.vocab_size
        );

        let load = |name: &str| -> DeviceBuffer {
            let tv = gguf.tensor_view(name).unwrap_or_else(|_| panic!("missing tensor: {name}"));
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
            x: backend.alloc(&[dim], DType::F32),
            x_norm: backend.alloc(&[dim], DType::F32),
            q: backend.alloc(&[dim], DType::F32),
            k: backend.alloc(&[kv_dim], DType::F32),
            v: backend.alloc(&[kv_dim], DType::F32),
            attn_out: backend.alloc(&[dim], DType::F32),
            wo_out: backend.alloc(&[dim], DType::F32),
            x2: backend.alloc(&[dim], DType::F32),
            gate: backend.alloc(&[hidden_dim], DType::F32),
            up: backend.alloc(&[hidden_dim], DType::F32),
            gate_up: backend.alloc(&[hidden_dim], DType::F32),
            ffn_out: backend.alloc(&[dim], DType::F32),
            logits: backend.alloc(&[config.vocab_size as u64], DType::F32),
        };

        LlamaModel {
            config,
            weights: LlamaWeights { token_embedding, output_norm, output_weight, layers },
            kv_cache,
            act,
        }
    }

    pub fn forward(&mut self, backend: &dyn Backend, token: u32, pos: usize) -> Vec<f32> {
        let head_dim = self.config.head_dim;
        let n_heads = self.config.n_heads;
        let kv_dim = self.config.n_kv_heads * head_dim;
        let heads_per_kv = n_heads / self.config.n_kv_heads;

        backend.embed(&mut self.act.x, &self.weights.token_embedding, token);

        for layer_idx in 0..self.config.n_layers {
            let layer = &self.weights.layers[layer_idx];

            // Attention
            backend.rms_norm(&mut self.act.x_norm, &self.act.x, &layer.attn_norm, self.config.norm_eps);
            backend.matvec_mul(&mut self.act.q, &layer.wq, &self.act.x_norm);
            backend.matvec_mul(&mut self.act.k, &layer.wk, &self.act.x_norm);
            backend.matvec_mul(&mut self.act.v, &layer.wv, &self.act.x_norm);
            backend.rope(&mut self.act.q, &mut self.act.k, pos, head_dim, self.config.rope_theta);

            self.kv_cache.store(backend, layer_idx, pos, &self.act.k, &self.act.v);

            let attn_result = compute_attention(
                backend, &self.act.q,
                &self.kv_cache.k_cache[layer_idx],
                &self.kv_cache.v_cache[layer_idx],
                pos, n_heads, head_dim, kv_dim, heads_per_kv,
            );
            write_f32_to_buffer(&mut self.act.attn_out, &attn_result);

            backend.matvec_mul(&mut self.act.wo_out, &layer.wo, &self.act.attn_out);
            backend.add(&mut self.act.x2, &self.act.x, &self.act.wo_out);

            // FFN
            backend.rms_norm(&mut self.act.x_norm, &self.act.x2, &layer.ffn_norm, self.config.norm_eps);
            backend.matvec_mul(&mut self.act.gate, &layer.w1, &self.act.x_norm);
            backend.matvec_mul(&mut self.act.up, &layer.w3, &self.act.x_norm);
            backend.silu(&mut self.act.gate);
            backend.mul(&mut self.act.gate_up, &self.act.gate, &self.act.up);
            backend.matvec_mul(&mut self.act.ffn_out, &layer.w2, &self.act.gate_up);

            backend.add(&mut self.act.x, &self.act.x2, &self.act.ffn_out);
        }

        backend.rms_norm(&mut self.act.x_norm, &self.act.x, &self.weights.output_norm, self.config.norm_eps);
        backend.matvec_mul(&mut self.act.logits, &self.weights.output_weight, &self.act.x_norm);
        backend.read_to_vec_f32(&self.act.logits)
    }
}

fn compute_attention(
    backend: &dyn Backend,
    q_buf: &DeviceBuffer,
    k_cache_buf: &DeviceBuffer,
    v_cache_buf: &DeviceBuffer,
    pos: usize,
    n_heads: usize,
    head_dim: usize,
    kv_dim: usize,
    heads_per_kv: usize,
) -> Vec<f32> {
    let q_data = backend.read_to_vec_f32(q_buf);
    let k_cache = backend.read_to_vec_f32(k_cache_buf);
    let v_cache = backend.read_to_vec_f32(v_cache_buf);

    let dim = n_heads * head_dim;
    let mut result = vec![0.0f32; dim];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..n_heads {
        let kv_h = h / heads_per_kv;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        let mut scores = vec![0.0f32; pos + 1];
        for t in 0..=pos {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q_data[q_off + d] * k_cache[t * kv_dim + kv_off + d];
            }
            scores[t] = dot * scale;
        }

        // Softmax
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max).exp();
            sum += *s;
        }
        for s in &mut scores {
            *s /= sum;
        }

        for d in 0..head_dim {
            let mut val = 0.0f32;
            for t in 0..=pos {
                val += scores[t] * v_cache[t * kv_dim + kv_off + d];
            }
            result[q_off + d] = val;
        }
    }

    result
}
