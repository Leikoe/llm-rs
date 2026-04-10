use crate::backend::Backend;
use crate::tensor::DType;

/// Per-layer key/value cache for autoregressive generation.
pub struct KVCache<B: Backend> {
    pub k_cache: Vec<B::Buffer>,
    pub v_cache: Vec<B::Buffer>,
}

impl<B: Backend> KVCache<B> {
    pub fn new(backend: &B, n_layers: usize, kv_dim: usize, max_seq_len: usize) -> Self {
        let mut k_cache = Vec::with_capacity(n_layers);
        let mut v_cache = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            k_cache.push(backend.alloc(&[max_seq_len as u64, kv_dim as u64], DType::BF16));
            v_cache.push(backend.alloc(&[max_seq_len as u64, kv_dim as u64], DType::BF16));
        }

        KVCache { k_cache, v_cache }
    }
}
