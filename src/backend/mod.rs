pub mod metal;

use crate::tensor::{DType, RopeLayout, TensorView};

/// The set of primitive ops a transformer forward pass needs.
///
/// Every op is shape-agnostic with respect to seq_len: decode (`seq_len == 1`)
/// and prefill (`seq_len > 1`) take exactly the same call. The backend reads
/// the shape from the input buffer and dispatches GEMV/GEMM and flash/causal
/// attention internally — the model layer doesn't know or care.
///
/// `Buffer` is an associated type so the model layer monomorphizes per backend
/// with no runtime indirection. There is no `dyn Backend` and no downcasting:
/// buffers from one backend cannot be passed to another by construction.
pub trait Backend {
    type Buffer;

    fn upload_tensor(&self, tv: &TensorView) -> Self::Buffer;
    fn alloc(&self, shape: &[u64], dtype: DType) -> Self::Buffer;

    /// Argmax over a 1D logit vector. Returns the index of the largest element.
    /// Stays on-device — only 4 bytes cross the bus per greedy step.
    fn argmax(&self, logits: &Self::Buffer) -> u32;

    /// Look up `tokens.len()` rows of `table` into `out`. `out` is laid out
    /// `[dim, seq_len]` with token `i`'s embedding in column `i`.
    fn embed(&self, out: &mut Self::Buffer, table: &Self::Buffer, tokens: &[u32]);

    /// `out = matmul(weight, input)`.
    /// GGML weight shape `[in_features, out_features]`. `input` is
    /// `[in_features, seq_len]` for prefill or `[in_features]` for decode.
    /// The backend reads `seq_len` from `input.shape` and dispatches
    /// GEMV or GEMM accordingly.
    fn matmul(&self, out: &mut Self::Buffer, weight: &Self::Buffer, input: &Self::Buffer);

    /// In-place RMSNorm along the contiguous (dim) axis. Operates on every
    /// column of `input` independently when `input` is 2D.
    fn rms_norm(&self, out: &mut Self::Buffer, input: &Self::Buffer, weight: &Self::Buffer, eps: f32);

    /// In-place per-head RMSNorm: independently normalize every contiguous
    /// `weight.len()`-sized chunk of `x`. Used by Qwen3's QK-norm. `weight` is broadcast
    /// across all heads.
    fn rms_norm_heads(&self, x: &mut Self::Buffer, weight: &Self::Buffer, eps: f32);

    /// Apply rotary embeddings to `q` and `k` in place. `positions` is a u32
    /// buffer giving the absolute position for each column.
    fn rope(&self, q: &mut Self::Buffer, k: &mut Self::Buffer, positions: &Self::Buffer, head_dim: usize, rope_theta: f32, layout: RopeLayout);

    /// Paged varlen grouped-query attention. Handles any mix of decode (1 token)
    /// and prefill (N tokens) requests in one dispatch.
    fn attention(
        &self,
        out: &mut Self::Buffer,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        block_table: &Self::Buffer,
        query_starts: &Self::Buffer,
        seq_lens: &Self::Buffer,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_blocks_per_seq: usize,
    );

    /// Write K/V vectors to paged pool slots via slot_mapping.
    fn scatter_kv(&self, pool: &mut Self::Buffer, src: &Self::Buffer, slot_mapping: &Self::Buffer, kv_dim: usize, num_tokens: usize);

    /// Gather columns from `src` at `indices` into `out`.
    fn gather(&self, out: &mut Self::Buffer, src: &Self::Buffer, indices: &Self::Buffer, num_indices: usize);

    /// Upload a small u32 array to a GPU buffer.
    fn upload_u32(&self, data: &[u32]) -> Self::Buffer;

    fn silu(&self, x: &mut Self::Buffer);
    fn mul(&self, out: &mut Self::Buffer, a: &Self::Buffer, b: &Self::Buffer);
    fn add(&self, out: &mut Self::Buffer, a: &Self::Buffer, b: &Self::Buffer);

    fn read_to_vec_f32(&self, buf: &Self::Buffer) -> Vec<f32>;

    /// Total GPU execution time accumulated since the last reset. Used to
    /// measure host overhead = wall_time - gpu_secs_total.
    fn gpu_secs_total(&self) -> f64;
    fn reset_gpu_secs(&self);

    /// Ensure all previously enqueued GPU work is complete.
    fn sync(&self);
}
