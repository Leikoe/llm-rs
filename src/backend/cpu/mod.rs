use half::{bf16, f16};

use crate::backend::{Backend, DeviceBuffer};
use crate::tensor::{DType, TensorView};

/// CPU buffer: either a borrowed slice (zero-copy from mmap) or owned data (activations).
pub enum CpuBuffer {
    Borrowed(*const u8, usize),
    Owned(Vec<u8>),
}

// SAFETY: Borrowed pointers come from mmap'd files that outlive the model.
unsafe impl Send for CpuBuffer {}
unsafe impl Sync for CpuBuffer {}

impl CpuBuffer {
    pub fn data(&self) -> &[u8] {
        match self {
            CpuBuffer::Borrowed(ptr, len) => unsafe { std::slice::from_raw_parts(*ptr, *len) },
            CpuBuffer::Owned(v) => v,
        }
    }

    fn data_mut(&mut self) -> &mut Vec<u8> {
        match self {
            CpuBuffer::Owned(v) => v,
            CpuBuffer::Borrowed(..) => panic!("cannot mutate borrowed buffer"),
        }
    }
}

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        CpuBackend
    }
}

impl Backend for CpuBackend {
    fn upload_tensor(&self, tv: &TensorView) -> DeviceBuffer {
        DeviceBuffer {
            inner: Box::new(CpuBuffer::Borrowed(tv.data.as_ptr(), tv.data.len())),
            dtype: tv.dtype,
            shape: tv.shape.clone(),
        }
    }

    fn alloc(&self, shape: &[u64], dtype: DType) -> DeviceBuffer {
        let n_elements: usize = shape.iter().map(|&d| d as usize).product();
        let size = dtype.storage_size(n_elements);
        DeviceBuffer {
            inner: Box::new(CpuBuffer::Owned(vec![0u8; size])),
            dtype,
            shape: shape.to_vec(),
        }
    }

    fn matvec_mul(&self, out: &mut DeviceBuffer, weight: &DeviceBuffer, input: &DeviceBuffer) {
        // GGML shape: [in_features, out_features] — out_features rows of in_features elements.
        let in_features = weight.shape[0] as usize;
        let out_features = weight.shape[1] as usize;

        // Read input as f32
        let input_f32 = self.read_to_vec_f32(input);
        assert!(input_f32.len() >= in_features);

        let mut result = vec![0.0f32; out_features];

        match weight.dtype {
            DType::F32 => {
                let w_data = get_cpu_data(weight);
                for row in 0..out_features {
                    let mut sum = 0.0f32;
                    let row_offset = row * in_features * 4;
                    for col in 0..in_features {
                        let w = f32::from_le_bytes([
                            w_data[row_offset + col * 4],
                            w_data[row_offset + col * 4 + 1],
                            w_data[row_offset + col * 4 + 2],
                            w_data[row_offset + col * 4 + 3],
                        ]);
                        sum += w * input_f32[col];
                    }
                    result[row] = sum;
                }
            }
            DType::F16 => {
                let w_data = get_cpu_data(weight);
                for row in 0..out_features {
                    let mut sum = 0.0f32;
                    let row_offset = row * in_features * 2;
                    for col in 0..in_features {
                        let w = f16::from_le_bytes([
                            w_data[row_offset + col * 2],
                            w_data[row_offset + col * 2 + 1],
                        ]);
                        sum += w.to_f32() * input_f32[col];
                    }
                    result[row] = sum;
                }
            }
            DType::BF16 => {
                let w_data = get_cpu_data(weight);
                for row in 0..out_features {
                    let mut sum = 0.0f32;
                    let row_offset = row * in_features * 2;
                    for col in 0..in_features {
                        let w = bf16::from_le_bytes([
                            w_data[row_offset + col * 2],
                            w_data[row_offset + col * 2 + 1],
                        ]);
                        sum += w.to_f32() * input_f32[col];
                    }
                    result[row] = sum;
                }
            }
            DType::Q4_0 => {
                let w_data = get_cpu_data(weight);
                let blocks_per_row = in_features / 32;
                for row in 0..out_features {
                    let mut sum = 0.0f32;
                    let row_block_offset = row * blocks_per_row * 18; // Q4_0 block = 18 bytes
                    for b in 0..blocks_per_row {
                        let block_offset = row_block_offset + b * 18;
                        let scale = f16::from_le_bytes([
                            w_data[block_offset],
                            w_data[block_offset + 1],
                        ])
                        .to_f32();
                        for j in 0..16 {
                            let packed = w_data[block_offset + 2 + j];
                            let lo = (packed & 0x0F) as i32 - 8;
                            let hi = (packed >> 4) as i32 - 8;
                            sum += scale * lo as f32 * input_f32[b * 32 + j * 2];
                            sum += scale * hi as f32 * input_f32[b * 32 + j * 2 + 1];
                        }
                    }
                    result[row] = sum;
                }
            }
            DType::Q8_0 => {
                let w_data = get_cpu_data(weight);
                let blocks_per_row = in_features / 32;
                for row in 0..out_features {
                    let mut sum = 0.0f32;
                    let row_block_offset = row * blocks_per_row * 34; // Q8_0 block = 34 bytes
                    for b in 0..blocks_per_row {
                        let block_offset = row_block_offset + b * 34;
                        let scale = f16::from_le_bytes([
                            w_data[block_offset],
                            w_data[block_offset + 1],
                        ])
                        .to_f32();
                        for j in 0..32 {
                            let q = w_data[block_offset + 2 + j] as i8;
                            sum += scale * q as f32 * input_f32[b * 32 + j];
                        }
                    }
                    result[row] = sum;
                }
            }
            _ => unimplemented!("matvec_mul not implemented for {:?}", weight.dtype),
        }

        write_f32_to_buffer(out, &result);
    }

    fn matmul(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer) {
        // a: [M, K], b: [K, N], out: [M, N]
        let m = a.shape[0] as usize;
        let k = a.shape[1] as usize;
        let n = b.shape[1] as usize;

        let a_f32 = self.read_to_vec_f32(a);
        let b_f32 = self.read_to_vec_f32(b);
        let mut result = vec![0.0f32; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_f32[i * k + p] * b_f32[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        write_f32_to_buffer(out, &result);
    }

    fn rms_norm(
        &self,
        out: &mut DeviceBuffer,
        input: &DeviceBuffer,
        weight: &DeviceBuffer,
        eps: f32,
    ) {
        let x = self.read_to_vec_f32(input);
        let w = self.read_to_vec_f32(weight);
        let dim = x.len();

        let mut sum_sq = 0.0f32;
        for &v in &x {
            sum_sq += v * v;
        }
        let rms = (sum_sq / dim as f32 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        let mut result = vec![0.0f32; dim];
        for i in 0..dim {
            result[i] = x[i] * inv_rms * w[i];
        }

        write_f32_to_buffer(out, &result);
    }

    fn rope(
        &self,
        q: &mut DeviceBuffer,
        k: &mut DeviceBuffer,
        pos: usize,
        head_dim: usize,
        rope_theta: f32,
    ) {
        rope_inplace(q, pos, head_dim, rope_theta);
        rope_inplace(k, pos, head_dim, rope_theta);
    }

    fn softmax(&self, x: &mut DeviceBuffer, len: usize) {
        let mut data = self.read_to_vec_f32(x);
        let slice = &mut data[..len];

        let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in slice.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in slice.iter_mut() {
            *v /= sum;
        }

        write_f32_to_buffer(x, &data);
    }

    fn silu(&self, x: &mut DeviceBuffer) {
        let mut data = self.read_to_vec_f32(x);
        for v in data.iter_mut() {
            *v = *v * (1.0 / (1.0 + (-*v).exp()));
        }
        write_f32_to_buffer(x, &data);
    }

    fn mul(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer) {
        let a_data = self.read_to_vec_f32(a);
        let b_data = self.read_to_vec_f32(b);
        let result: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).collect();
        write_f32_to_buffer(out, &result);
    }

    fn add(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer) {
        let a_data = self.read_to_vec_f32(a);
        let b_data = self.read_to_vec_f32(b);
        let result: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a + b).collect();
        write_f32_to_buffer(out, &result);
    }

    fn embed(&self, out: &mut DeviceBuffer, table: &DeviceBuffer, token_id: u32) {
        // GGML shape convention: [contiguous_dim, outer_dim] = [dim, vocab_size].
        // Memory layout: vocab_size rows of dim elements each.
        // Row token_id starts at offset token_id * dim.
        let dim = table.shape[0] as usize;
        let table_f32 = self.read_to_vec_f32(table);
        let start = token_id as usize * dim;
        let result = table_f32[start..start + dim].to_vec();
        write_f32_to_buffer(out, &result);
    }

    fn copy_into(&self, dst: &mut DeviceBuffer, src: &DeviceBuffer, dst_offset_elements: usize) {
        let src_data = get_cpu_data(src);
        let dst_buf = dst.inner.downcast_mut::<CpuBuffer>().unwrap();
        let offset_bytes = dst_offset_elements * 4; // f32
        let len = src_data.len();
        dst_buf.data_mut()[offset_bytes..offset_bytes + len].copy_from_slice(src_data);
    }

    fn read_to_vec_f32(&self, buf: &DeviceBuffer) -> Vec<f32> {
        let data = get_cpu_data(buf);
        match buf.dtype {
            DType::F32 => {
                data.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            DType::F16 => {
                data.chunks_exact(2)
                    .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect()
            }
            DType::BF16 => {
                data.chunks_exact(2)
                    .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
                    .collect()
            }
            _ => {
                // For quantized types, dequantize to f32
                dequantize_to_f32(data, buf.dtype, buf.n_elements())
            }
        }
    }
}

fn get_cpu_data(buf: &DeviceBuffer) -> &[u8] {
    buf.inner.downcast_ref::<CpuBuffer>().unwrap().data()
}

pub fn write_f32_to_buffer(buf: &mut DeviceBuffer, data: &[f32]) {
    let cpu_buf = buf.inner.downcast_mut::<CpuBuffer>().unwrap();
    let vec = cpu_buf.data_mut();
    vec.clear();
    vec.reserve(data.len() * 4);
    for &v in data {
        vec.extend_from_slice(&v.to_le_bytes());
    }
    buf.dtype = DType::F32;
}

fn rope_inplace(buf: &mut DeviceBuffer, pos: usize, head_dim: usize, theta: f32) {
    let mut data = {
        let cpu_buf = buf.inner.downcast_ref::<CpuBuffer>().unwrap();
        cpu_buf
            .data()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect::<Vec<f32>>()
    };

    let n_elements = data.len();
    let n_heads = n_elements / head_dim;

    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..head_dim / 2 {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            let cos = angle.cos();
            let sin = angle.sin();

            // LLaMA uses interleaved pairs: (x[2i], x[2i+1])
            let x0 = data[base + 2 * i];
            let x1 = data[base + 2 * i + 1];
            data[base + 2 * i] = x0 * cos - x1 * sin;
            data[base + 2 * i + 1] = x0 * sin + x1 * cos;
        }
    }

    write_f32_to_buffer(buf, &data);
}

fn dequantize_to_f32(data: &[u8], dtype: DType, n_elements: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; n_elements];
    match dtype {
        DType::Q4_0 => {
            let n_blocks = n_elements / 32;
            for b in 0..n_blocks {
                let block_offset = b * 18;
                let scale = f16::from_le_bytes([
                    data[block_offset],
                    data[block_offset + 1],
                ])
                .to_f32();
                for j in 0..16 {
                    let packed = data[block_offset + 2 + j];
                    let lo = (packed & 0x0F) as i32 - 8;
                    let hi = (packed >> 4) as i32 - 8;
                    result[b * 32 + j * 2] = scale * lo as f32;
                    result[b * 32 + j * 2 + 1] = scale * hi as f32;
                }
            }
        }
        DType::Q8_0 => {
            let n_blocks = n_elements / 32;
            for b in 0..n_blocks {
                let block_offset = b * 34;
                let scale = f16::from_le_bytes([
                    data[block_offset],
                    data[block_offset + 1],
                ])
                .to_f32();
                for j in 0..32 {
                    let q = data[block_offset + 2 + j] as i8;
                    result[b * 32 + j] = scale * q as f32;
                }
            }
        }
        _ => unimplemented!("dequantize_to_f32 not implemented for {:?}", dtype),
    }
    result
}
