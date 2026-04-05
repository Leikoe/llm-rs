use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;

use crate::backend::{Backend, DeviceBuffer};
use crate::tensor::{DType, TensorView};

struct MetalBuffer(Retained<ProtocolObject<dyn MTLBuffer>>);

struct CmdState {
    cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

pub struct MetalBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    cmd: RefCell<Option<CmdState>>,
}

impl MetalBackend {
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let queue = device.newCommandQueue().expect("No command queue");

        let source = NSString::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/ops.metal"
        )));
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .expect("Failed to compile Metal shaders");

        let names: &[&str] = &[
            "embed_f32",
            "embed_bf16",
            "embed_f16",
            "embed_q4k",
            "embed_q6k",
            "matvec_f32",
            "matvec_bf16",
            "matvec_f16",
            "matvec_q4_0",
            "matvec_q8_0",
            "matvec_q4k",
            "matvec_q6k",
            "rms_norm",
            "rope",
            "softmax_kernel",
            "silu_inplace",
            "add_vecs",
            "mul_vecs",
            "copy_offset",
            "matvec_f32_simd",
            "matvec_bf16_simd",
            "matvec_f16_simd",
            "matvec_q4_0_simd",
            "matvec_q8_0_simd",
            "matvec_q4k_simd",
            "matvec_q6k_simd",
            "gqa_attention",
        ];

        let mut pipelines = HashMap::new();
        for &name in names {
            let ns = NSString::from_str(name);
            let func = library
                .newFunctionWithName(&ns)
                .unwrap_or_else(|| panic!("Function not found: {name}"));
            let pso = device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|e| panic!("Pipeline failed for {name}: {e}"));
            pipelines.insert(name, pso);
        }

        eprintln!("Metal backend initialized");

        MetalBackend {
            device,
            queue,
            pipelines,
            cmd: RefCell::new(None),
        }
    }

    fn flush(&self) {
        let mut state = self.cmd.borrow_mut();
        if let Some(s) = state.take() {
            s.encoder.endEncoding();
            s.cmd_buf.commit();
            s.cmd_buf.waitUntilCompleted();
        }
    }

    fn encode(
        &self,
        pipeline: &'static str,
        grid: MTLSize,
        tg: MTLSize,
        tg_mem: usize,
        setup: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) {
        let mut state = self.cmd.borrow_mut();
        let s = state.get_or_insert_with(|| {
            let cb = self.queue.commandBuffer().unwrap();
            let enc = cb.computeCommandEncoder().unwrap();
            CmdState {
                cmd_buf: cb,
                encoder: enc,
            }
        });
        let pso = &self.pipelines[pipeline];
        unsafe {
            s.encoder.setComputePipelineState(pso);
            setup(&s.encoder);
            if tg_mem > 0 {
                s.encoder
                    .setThreadgroupMemoryLength_atIndex(tg_mem as _, 0);
            }
            s.encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
            s.encoder
                .memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
    }

    fn encode_groups(
        &self,
        pipeline: &'static str,
        groups: MTLSize,
        tg: MTLSize,
        tg_mem: usize,
        setup: impl FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) {
        let mut state = self.cmd.borrow_mut();
        let s = state.get_or_insert_with(|| {
            let cb = self.queue.commandBuffer().unwrap();
            let enc = cb.computeCommandEncoder().unwrap();
            CmdState {
                cmd_buf: cb,
                encoder: enc,
            }
        });
        let pso = &self.pipelines[pipeline];
        unsafe {
            s.encoder.setComputePipelineState(pso);
            setup(&s.encoder);
            if tg_mem > 0 {
                s.encoder
                    .setThreadgroupMemoryLength_atIndex(tg_mem as _, 0);
            }
            s.encoder
                .dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
            s.encoder
                .memoryBarrierWithScope(MTLBarrierScope::Buffers);
        }
    }
}

fn sz(w: usize, h: usize, d: usize) -> MTLSize {
    MTLSize {
        width: w,
        height: h,
        depth: d,
    }
}

unsafe fn set_buf(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buf: &DeviceBuffer,
    idx: usize,
) {
    let m = &buf.inner.downcast_ref::<MetalBuffer>().unwrap().0;
    unsafe { enc.setBuffer_offset_atIndex(Some(m), 0, idx) };
}

unsafe fn set_u32(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, val: u32, idx: usize) {
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::new_unchecked(&val as *const u32 as *mut _),
            4,
            idx,
        )
    };
}

unsafe fn set_f32(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, val: f32, idx: usize) {
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::new_unchecked(&val as *const f32 as *mut _),
            4,
            idx,
        )
    };
}

impl Backend for MetalBackend {
    fn upload_tensor(&self, tv: &TensorView) -> DeviceBuffer {
        let buf = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(tv.data.as_ptr() as *mut _),
                tv.data.len(),
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("Failed to create buffer");
        DeviceBuffer {
            inner: Box::new(MetalBuffer(buf)),
            dtype: tv.dtype,
            shape: tv.shape.clone(),
        }
    }

    fn alloc(&self, shape: &[u64], dtype: DType) -> DeviceBuffer {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let size = dtype.storage_size(n);
        let buf = self
            .device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .expect("Failed to alloc buffer");
        DeviceBuffer {
            inner: Box::new(MetalBuffer(buf)),
            dtype,
            shape: shape.to_vec(),
        }
    }

    fn matvec_mul(&self, out: &mut DeviceBuffer, weight: &DeviceBuffer, input: &DeviceBuffer) {
        let kernel = match weight.dtype {
            DType::F32 => "matvec_f32_simd",
            DType::BF16 => "matvec_bf16_simd",
            DType::F16 => "matvec_f16_simd",
            DType::Q4_0 => "matvec_q4_0_simd",
            DType::Q8_0 => "matvec_q8_0_simd",
            DType::Q4K => "matvec_q4k_simd",
            DType::Q6K => "matvec_q6k_simd",
            _ => unimplemented!("matvec for {:?}", weight.dtype),
        };
        let in_f = weight.shape[0] as u32;
        let out_f = weight.shape[1] as u32;
        let rows_per_tg: usize = 8;
        let n_tg = (out_f as usize + rows_per_tg - 1) / rows_per_tg;
        self.encode_groups(
            kernel,
            sz(n_tg, 1, 1),
            sz(rows_per_tg * 32, 1, 1),
            0,
            |enc| unsafe {
                set_buf(enc, weight, 0);
                set_buf(enc, input, 1);
                set_buf(enc, out, 2);
                set_u32(enc, in_f, 3);
                set_u32(enc, out_f, 4);
            },
        );
    }

    fn matmul(&self, _out: &mut DeviceBuffer, _a: &DeviceBuffer, _b: &DeviceBuffer) {
        unimplemented!("Metal matmul not yet implemented");
    }

    fn rms_norm(
        &self,
        out: &mut DeviceBuffer,
        input: &DeviceBuffer,
        weight: &DeviceBuffer,
        eps: f32,
    ) {
        let dim = input.n_elements() as u32;
        let tg = 1024usize.min(dim as usize);
        let shared = (((tg as usize) + 31) / 32) * 4;
        self.encode_groups("rms_norm", sz(1, 1, 1), sz(tg, 1, 1), shared, |enc| unsafe {
            set_buf(enc, input, 0);
            set_buf(enc, weight, 1);
            set_buf(enc, out, 2);
            set_u32(enc, dim, 3);
            set_f32(enc, eps, 4);
        });
    }

    fn rope(
        &self,
        q: &mut DeviceBuffer,
        k: &mut DeviceBuffer,
        pos: usize,
        head_dim: usize,
        rope_theta: f32,
    ) {
        let pos = pos as u32;
        let hd = head_dim as u32;

        let q_pairs = (q.n_elements() / 2) as u32;
        self.encode("rope", sz(q_pairs as usize, 1, 1), sz(256, 1, 1), 0, |enc| unsafe {
            set_buf(enc, q, 0);
            set_u32(enc, pos, 1);
            set_u32(enc, hd, 2);
            set_f32(enc, rope_theta, 3);
            set_u32(enc, q_pairs, 4);
        });

        let k_pairs = (k.n_elements() / 2) as u32;
        self.encode("rope", sz(k_pairs as usize, 1, 1), sz(256, 1, 1), 0, |enc| unsafe {
            set_buf(enc, k, 0);
            set_u32(enc, pos, 1);
            set_u32(enc, hd, 2);
            set_f32(enc, rope_theta, 3);
            set_u32(enc, k_pairs, 4);
        });
    }

    fn softmax(&self, x: &mut DeviceBuffer, len: usize) {
        let len_u32 = len as u32;
        let tg = 1024usize.min(len);
        let shared = (((tg as usize) + 31) / 32) * 4;
        self.encode_groups(
            "softmax_kernel",
            sz(1, 1, 1),
            sz(tg, 1, 1),
            shared,
            |enc| unsafe {
                set_buf(enc, x, 0);
                set_u32(enc, len_u32, 1);
            },
        );
    }

    fn silu(&self, x: &mut DeviceBuffer) {
        let n = x.n_elements() as u32;
        self.encode("silu_inplace", sz(n as usize, 1, 1), sz(256, 1, 1), 0, |enc| unsafe {
            set_buf(enc, x, 0);
            set_u32(enc, n, 1);
        });
    }

    fn mul(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer) {
        let n = a.n_elements() as u32;
        self.encode("mul_vecs", sz(n as usize, 1, 1), sz(256, 1, 1), 0, |enc| unsafe {
            set_buf(enc, a, 0);
            set_buf(enc, b, 1);
            set_buf(enc, out, 2);
            set_u32(enc, n, 3);
        });
    }

    fn add(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer) {
        let n = a.n_elements() as u32;
        self.encode("add_vecs", sz(n as usize, 1, 1), sz(256, 1, 1), 0, |enc| unsafe {
            set_buf(enc, a, 0);
            set_buf(enc, b, 1);
            set_buf(enc, out, 2);
            set_u32(enc, n, 3);
        });
    }

    fn embed(&self, out: &mut DeviceBuffer, table: &DeviceBuffer, token_id: u32) {
        let kernel = match table.dtype {
            DType::F32 => "embed_f32",
            DType::BF16 => "embed_bf16",
            DType::F16 => "embed_f16",
            DType::Q4K => "embed_q4k",
            DType::Q6K => "embed_q6k",
            _ => unimplemented!("embed for {:?}", table.dtype),
        };
        let dim = table.shape[0] as u32;
        self.encode(kernel, sz(dim as usize, 1, 1), sz(256, 1, 1), 0, |enc| unsafe {
            set_buf(enc, table, 0);
            set_buf(enc, out, 1);
            set_u32(enc, token_id, 2);
            set_u32(enc, dim, 3);
        });
    }

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
    ) {
        let kv_dim = (n_kv_heads * head_dim) as u32;
        let shared_mem = (pos + 1 + 8) * 4; // scores + scratch
        self.encode_groups(
            "gqa_attention",
            sz(n_heads, 1, 1),
            sz(head_dim, 1, 1),
            shared_mem,
            |enc| unsafe {
                set_buf(enc, q, 0);
                set_buf(enc, k_cache, 1);
                set_buf(enc, v_cache, 2);
                set_buf(enc, out, 3);
                set_u32(enc, pos as u32, 4);
                set_u32(enc, n_heads as u32, 5);
                set_u32(enc, n_kv_heads as u32, 6);
                set_u32(enc, head_dim as u32, 7);
                set_u32(enc, kv_dim, 8);
            },
        );
    }

    fn copy_into(&self, dst: &mut DeviceBuffer, src: &DeviceBuffer, dst_offset_elements: usize) {
        let count = src.n_elements() as u32;
        let offset = dst_offset_elements as u32;
        self.encode(
            "copy_offset",
            sz(count as usize, 1, 1),
            sz(256, 1, 1),
            0,
            |enc| unsafe {
                set_buf(enc, src, 0);
                set_buf(enc, dst, 1);
                set_u32(enc, offset, 2);
                set_u32(enc, count, 3);
            },
        );
    }

    fn read_to_vec_f32(&self, buf: &DeviceBuffer) -> Vec<f32> {
        self.flush();
        let m = &buf.inner.downcast_ref::<MetalBuffer>().unwrap().0;
        let n = buf.n_elements();
        let ptr = m.contents().as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
    }

    fn write_from_f32(&self, buf: &mut DeviceBuffer, data: &[f32]) {
        self.flush();
        let m = &buf.inner.downcast_ref::<MetalBuffer>().unwrap().0;
        let ptr = m.contents().as_ptr() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        buf.dtype = DType::F32;
    }
}
