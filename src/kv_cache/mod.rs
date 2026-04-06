use crate::backend::{Backend, DeviceBuffer};
use crate::model::ModelConfig;
use crate::tensor::DType;

/// Per-layer key/value cache for autoregressive generation.
pub struct KVCache {
    /// Per-layer key cache: flat [max_seq_len * kv_dim] as bf16.
    pub k_cache: Vec<DeviceBuffer>,
    /// Per-layer value cache: flat [max_seq_len * kv_dim] as bf16.
    pub v_cache: Vec<DeviceBuffer>,
    pub kv_dim: usize,
    pub max_seq_len: usize,
}

impl KVCache {
    pub fn new(backend: &dyn Backend, config: &ModelConfig) -> Self {
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

    /// Store k, v vectors for a given layer at the given position.
    pub fn store(
        &mut self,
        backend: &dyn Backend,
        layer: usize,
        pos: usize,
        k: &DeviceBuffer,
        v: &DeviceBuffer,
    ) {
        let offset = pos * self.kv_dim;
        backend.copy_into(&mut self.k_cache[layer], k, offset);
        backend.copy_into(&mut self.v_cache[layer], v, offset);
    }
}
