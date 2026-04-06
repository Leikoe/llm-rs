pub mod cpu;
#[cfg(all(target_os = "macos", feature = "metal"))]
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

pub trait Backend {
    fn upload_tensor(&self, tv: &TensorView) -> DeviceBuffer;
    fn alloc(&self, shape: &[u64], dtype: DType) -> DeviceBuffer;

    /// Matrix-vector multiply: out = weight @ input.
    /// GGML shape: weight is [in_features, out_features] in memory.
    /// Weight may be quantized. The kernel handles dequantization.
    fn matvec_mul(&self, out: &mut DeviceBuffer, weight: &DeviceBuffer, input: &DeviceBuffer);
    fn matmul(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer);
    fn rms_norm(&self, out: &mut DeviceBuffer, input: &DeviceBuffer, weight: &DeviceBuffer, eps: f32);
    fn rope(&self, q: &mut DeviceBuffer, k: &mut DeviceBuffer, pos: usize, head_dim: usize, rope_theta: f32);
    fn softmax(&self, x: &mut DeviceBuffer, len: usize);
    fn silu(&self, x: &mut DeviceBuffer);
    fn mul(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer);
    fn add(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer);
    fn embed(&self, out: &mut DeviceBuffer, table: &DeviceBuffer, token_id: u32);
    fn copy_into(&self, dst: &mut DeviceBuffer, src: &DeviceBuffer, dst_offset_elements: usize);
    fn gqa_attention(
        &self,
        out: &mut DeviceBuffer,
        q: &DeviceBuffer,
        k_cache: &DeviceBuffer,
        v_cache: &DeviceBuffer,
        pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    );
    fn read_to_vec_f32(&self, buf: &DeviceBuffer) -> Vec<f32>;
    fn write_from_f32(&self, buf: &mut DeviceBuffer, data: &[f32]);

    // Batched ops for prefill
    fn embed_batch(&self, out: &mut DeviceBuffer, table: &DeviceBuffer, token_ids: &[u32]);
    fn rms_norm_batch(&self, out: &mut DeviceBuffer, input: &DeviceBuffer, weight: &DeviceBuffer, eps: f32, seq_len: usize);
    fn rope_batch(&self, q: &mut DeviceBuffer, k: &mut DeviceBuffer, start_pos: usize, seq_len: usize, head_dim: usize, rope_theta: f32);
    fn gqa_attention_batch(
        &self,
        out: &mut DeviceBuffer,
        q: &DeviceBuffer,
        k_cache: &DeviceBuffer,
        v_cache: &DeviceBuffer,
        start_pos: usize,
        seq_len: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    );

    /// Ensure all previously enqueued GPU work is complete.
    fn sync(&self) {}
}
