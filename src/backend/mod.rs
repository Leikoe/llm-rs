pub mod metal;

use std::any::Any;

use crate::tensor::{DType, TensorView};

pub struct DeviceBuffer {
    pub(crate) inner: Box<dyn Any>,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl DeviceBuffer {
    pub fn n_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }
}

/// The set of primitive ops a transformer forward pass needs.
///
/// Every op is shape-agnostic with respect to seq_len: decode (`seq_len == 1`)
/// and prefill (`seq_len > 1`) take exactly the same call. The backend picks
/// the right kernel internally — `matmul` dispatches GEMV for one column and
/// GEMM for many; `attention` dispatches single-query flash attention or
/// batched causal attention. The model layer doesn't know or care.
pub trait Backend {
    fn upload_tensor(&self, tv: &TensorView) -> DeviceBuffer;
    fn alloc(&self, shape: &[u64], dtype: DType) -> DeviceBuffer;

    /// Argmax over a 1D logit vector. Returns the index of the largest element.
    /// Stays on-device — only 4 bytes cross the bus per greedy step.
    fn argmax(&self, logits: &DeviceBuffer) -> u32;

    /// Look up `tokens.len()` rows of `table` into `out`. `out` is laid out
    /// `[dim, seq_len]` with token `i`'s embedding in column `i`.
    fn embed(&self, out: &mut DeviceBuffer, table: &DeviceBuffer, tokens: &[u32]);

    /// `out = matmul(weight, input)`.
    /// GGML weight shape `[in_features, out_features]`. `input` is `[in_features, seq_len]`
    /// for prefill or `[in_features]` / `[in_features, 1]` for decode. The backend
    /// reads `seq_len` from `input.shape` and dispatches GEMV or GEMM accordingly.
    fn matmul(&self, out: &mut DeviceBuffer, weight: &DeviceBuffer, input: &DeviceBuffer);

    /// In-place RMSNorm along the contiguous (dim) axis. Operates on every column
    /// of `input` independently when `input` is 2D.
    fn rms_norm(&self, out: &mut DeviceBuffer, input: &DeviceBuffer, weight: &DeviceBuffer, eps: f32);

    /// Apply rotary embeddings to `q` and `k` in place. `start_pos` is the
    /// absolute position of column 0; column `s` gets position `start_pos + s`.
    fn rope(&self, q: &mut DeviceBuffer, k: &mut DeviceBuffer, start_pos: usize, head_dim: usize, rope_theta: f32);

    /// Grouped-query attention. `start_pos` is the position of `q`'s first column;
    /// `q.shape[1]` (or 1) is the number of query columns. KV cache must already
    /// contain entries for positions `0..start_pos + seq_len`.
    fn attention(
        &self,
        out: &mut DeviceBuffer,
        q: &DeviceBuffer,
        k_cache: &DeviceBuffer,
        v_cache: &DeviceBuffer,
        start_pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    );

    fn silu(&self, x: &mut DeviceBuffer);
    fn mul(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer);
    fn add(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer);

    /// Copy `src` into `dst` starting at `dst_offset_elements`. Used to fold
    /// freshly-computed K/V vectors into the KV cache.
    fn copy_into(&self, dst: &mut DeviceBuffer, src: &DeviceBuffer, dst_offset_elements: usize);

    fn read_to_vec_f32(&self, buf: &DeviceBuffer) -> Vec<f32>;

    /// Total GPU execution time accumulated since the last reset. Default 0
    /// for backends that don't track this. Used to measure encode/sync overhead
    /// = wall_time - gpu_secs_total.
    fn gpu_secs_total(&self) -> f64 { 0.0 }
    fn reset_gpu_secs(&self) {}

    /// Ensure all previously enqueued GPU work is complete.
    fn sync(&self) {}
}
