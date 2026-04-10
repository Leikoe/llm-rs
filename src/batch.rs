use crate::backend::Backend;

/// Everything the model needs for one forward step. Built by the CLI (single
/// request) or the serving engine (multiple requests batched together).
pub struct Batch<B: Backend> {
    pub tokens: Vec<u32>,
    pub positions: B::Buffer,      // [num_tokens] u32 — absolute position per token
    pub query_starts: B::Buffer,   // [num_requests + 1] u32 — cumsum separating requests
    pub seq_lens: B::Buffer,       // [num_requests] u32 — total KV length per request
    pub block_tables: B::Buffer,   // [num_requests × max_blocks_per_seq] u32
    pub slot_mapping: B::Buffer,   // [num_tokens] u32 — KV pool slot for each token
    pub logit_indices: B::Buffer,  // [num_requests] u32 — last token of each request
    pub num_tokens: usize,
    pub num_requests: usize,
    pub max_blocks_per_seq: usize,
}

impl<B: Backend> Batch<B> {
    /// Build a batch for a single sequence (CLI path).
    pub fn single(
        backend: &B,
        tokens: &[u32],
        pos: usize,
        block_table: &[u32],
        block_size: usize,
    ) -> Self {
        let n = tokens.len();
        let positions: Vec<u32> = (0..n).map(|i| (pos + i) as u32).collect();
        let query_starts = vec![0u32, n as u32];
        let seq_lens = vec![(pos + n) as u32];
        let slot_mapping: Vec<u32> = (0..n).map(|i| {
            let p = pos + i;
            block_table[p / block_size] * block_size as u32 + (p % block_size) as u32
        }).collect();
        let logit_indices = vec![(n - 1) as u32];

        Batch {
            tokens: tokens.to_vec(),
            positions: backend.upload_u32(&positions),
            query_starts: backend.upload_u32(&query_starts),
            seq_lens: backend.upload_u32(&seq_lens),
            block_tables: backend.upload_u32(block_table),
            slot_mapping: backend.upload_u32(&slot_mapping),
            logit_indices: backend.upload_u32(&logit_indices),
            num_tokens: n,
            num_requests: 1,
            max_blocks_per_seq: block_table.len(),
        }
    }
}
