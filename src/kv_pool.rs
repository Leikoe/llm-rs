use crate::backend::Backend;
use crate::tensor::DType;

pub const BLOCK_SIZE: usize = 256;

/// Fraction of free memory (after model weights) to use for the KV cache.
const KV_MEMORY_FRACTION: f64 = 0.80;

/// Paged KV cache pool. Single large buffer per layer, divided into fixed-size
/// blocks. A free list tracks available blocks. Sequences reference blocks via
/// a block table (logical block index → physical block ID).
pub struct PagedKVPool<B: Backend> {
    pub k: Vec<B::Buffer>,
    pub v: Vec<B::Buffer>,
    free: Vec<u32>,
    pub block_size: usize,
    pub kv_dim: usize,
}

impl<B: Backend> PagedKVPool<B> {
    /// Create a KV pool sized to ~80% of free memory (after model weights),
    /// capped at 2 GB to avoid excessive allocation time.
    pub fn new(backend: &B, n_layers: usize, kv_dim: usize, weight_bytes: usize) -> Self {
        let block_size = BLOCK_SIZE;
        // BF16: 2 bytes per element, K + V, all layers
        let bytes_per_token = n_layers * 2 * kv_dim * 2;
        let bytes_per_block = bytes_per_token * block_size;

        let total_ram = system_memory_bytes();
        let available = total_ram.saturating_sub(weight_bytes);
        let kv_budget = (available as f64 * KV_MEMORY_FRACTION) as usize;
        // Cap at 2 GB to keep allocation fast. Increase with --kv-size later.
        let kv_budget = kv_budget.min(2 * 1024 * 1024 * 1024);
        let max_blocks = (kv_budget / bytes_per_block).max(1);
        let total_rows = (max_blocks * block_size) as u64;

        let mut k = Vec::with_capacity(n_layers);
        let mut v = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            k.push(backend.alloc(&[total_rows, kv_dim as u64], DType::BF16));
            v.push(backend.alloc(&[total_rows, kv_dim as u64], DType::BF16));
        }
        let free: Vec<u32> = (0..max_blocks as u32).rev().collect();
        let max_tokens = max_blocks * block_size;
        let gb = max_blocks as f64 * bytes_per_block as f64 / 1e9;
        eprintln!("KV pool: {max_blocks} blocks × {block_size} tokens = {max_tokens} max tokens, {gb:.2} GB ({:.0}% of {:.1} GB free)",
            KV_MEMORY_FRACTION * 100.0, available as f64 / 1e9);
        PagedKVPool { k, v, free, block_size, kv_dim }
    }

    pub fn alloc_block(&mut self) -> Option<u32> {
        self.free.pop()
    }

    pub fn free_blocks(&mut self, blocks: &[u32]) {
        self.free.extend(blocks);
    }

    pub fn blocks_free(&self) -> usize {
        self.free.len()
    }
}

fn system_memory_bytes() -> usize {
    unsafe {
        let mut size: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = b"hw.memsize\0";
        libc::sysctlbyname(
            name.as_ptr() as *const _,
            &mut size as *mut u64 as *mut _,
            &mut len,
            std::ptr::null_mut(),
            0,
        );
        size as usize
    }
}

/// Try to allocate enough blocks for `needed_tokens` total. Returns true on
/// success, false if the pool doesn't have enough free blocks (no partial
/// allocation — all or nothing).
pub fn ensure_blocks<B: Backend>(
    block_table: &mut Vec<u32>,
    pool: &mut PagedKVPool<B>,
    needed_tokens: usize,
) -> bool {
    let blocks_needed = (needed_tokens + pool.block_size - 1) / pool.block_size;
    let new_blocks = blocks_needed.saturating_sub(block_table.len());
    if new_blocks > pool.blocks_free() { return false; }
    for _ in 0..new_blocks {
        block_table.push(pool.alloc_block().unwrap());
    }
    true
}
