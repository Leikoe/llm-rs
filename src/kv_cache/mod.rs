use crate::backend::Backend;
use crate::model::ModelConfig;
use crate::tensor::DType;

/// Per-layer key/value cache for autoregressive generation.
pub struct KVCache<B: Backend> {
    /// Per-layer key cache: flat [max_seq_len * kv_dim] as bf16.
    pub k_cache: Vec<B::Buffer>,
    /// Per-layer value cache: flat [max_seq_len * kv_dim] as bf16.
    pub v_cache: Vec<B::Buffer>,
    pub kv_dim: usize,
    pub max_seq_len: usize,
}

impl<B: Backend> KVCache<B> {
    pub fn new(backend: &B, config: &ModelConfig) -> Self {
        let kv_dim = config.n_kv_heads * config.head_dim;
        let mut k_cache = Vec::with_capacity(config.n_layers);
        let mut v_cache = Vec::with_capacity(config.n_layers);

        for _ in 0..config.n_layers {
            k_cache.push(backend.alloc(
                &[config.max_seq_len as u64, kv_dim as u64],
                DType::BF16,
            ));
            v_cache.push(backend.alloc(
                &[config.max_seq_len as u64, kv_dim as u64],
                DType::BF16,
            ));
        }

        KVCache {
            k_cache,
            v_cache,
            kv_dim,
            max_seq_len: config.max_seq_len,
        }
    }
}
