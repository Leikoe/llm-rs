pub mod metal;

use crate::model::RopeLayout;
use crate::tensor::{DType, TensorView};

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

    /// Apply rotary embeddings to `q` and `k` in place. `start_pos` is the
    /// absolute position of column 0; column `s` gets position `start_pos + s`.
    /// `layout` controls pair selection: `Interleaved` rotates `(x[2i], x[2i+1])`
    /// (LLaMA); `SplitHalf` rotates `(x[i], x[i+head_dim/2])` (Qwen/HF).
    fn rope(&self, q: &mut Self::Buffer, k: &mut Self::Buffer, start_pos: usize, head_dim: usize, rope_theta: f32, layout: RopeLayout);

    /// Grouped-query attention. `start_pos` is the position of `q`'s first
    /// column. KV cache must already contain entries for positions
    /// `0..start_pos + seq_len`.
    fn attention(
        &self,
        out: &mut Self::Buffer,
        q: &Self::Buffer,
        k_cache: &Self::Buffer,
        v_cache: &Self::Buffer,
        start_pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    );

    fn silu(&self, x: &mut Self::Buffer);
    fn mul(&self, out: &mut Self::Buffer, a: &Self::Buffer, b: &Self::Buffer);
    fn add(&self, out: &mut Self::Buffer, a: &Self::Buffer, b: &Self::Buffer);

    /// Copy `src` into `dst` starting at `dst_offset_elements`. Used to fold
    /// freshly-computed K/V vectors into the KV cache.
    fn copy_into(&self, dst: &mut Self::Buffer, src: &Self::Buffer, dst_offset_elements: usize);

    fn read_to_vec_f32(&self, buf: &Self::Buffer) -> Vec<f32>;

    /// Total GPU execution time accumulated since the last reset. Used to
    /// measure host overhead = wall_time - gpu_secs_total.
    fn gpu_secs_total(&self) -> f64;
    fn reset_gpu_secs(&self);

    /// Ensure all previously enqueued GPU work is complete.
    fn sync(&self);
}
