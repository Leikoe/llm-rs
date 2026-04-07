use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::*;

use crate::backend::{Backend, DeviceBuffer};
use crate::tensor::{DType, TensorView};

struct MetalBuffer {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Byte offset of the logical data within `buf`. Non-zero only for zero-copy
    /// weight uploads where the underlying GGUF tensor isn't page-aligned and we
    /// wrapped a page-rounded-down region.
    offset: usize,
}

const PAGE_SIZE: usize = 16384;

/// Load the pre-compiled Metal library (.metallib) embedded at build time.
/// Falls back to runtime source compilation if the Metal toolchain wasn't available.
fn load_metal_library(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Retained<ProtocolObject<dyn MTLLibrary>> {
    #[cfg(metal_precompiled)]
    {
        let metallib_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/default.metallib"));
        let path = std::env::temp_dir().join("llm_rs_ops.metallib");
        std::fs::write(&path, metallib_bytes).expect("Failed to write metallib");
        let url = NSURL::fileURLWithPath(&NSString::from_str(path.to_str().unwrap()));
        device
            .newLibraryWithURL_error(&url)
            .expect("Failed to load pre-compiled Metal library")
    }
    #[cfg(not(metal_precompiled))]
    {
        eprintln!("Warning: Metal shaders not pre-compiled, JIT compiling from source...");
        let source = NSString::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/ops.metal"
        )));
        device
            .newLibraryWithSource_options_error(&source, None)
            .expect("Failed to compile Metal shaders")
    }
}

struct CmdState {
    cmd_buf: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
}

// M1 Pro 14-core GPU specs (adjust for your hardware)
const GPU_MEM_BW_GB_S: f64 = 200.0;
const GPU_FP32_TFLOPS: f64 = 4.5;
const GPU_FP16_TFLOPS: f64 = 9.0;

#[derive(Default)]
struct KernelStats {
    gpu_secs: f64,
    bytes: usize,
    flops: usize,
    count: usize,
}

pub struct MetalBackend {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: HashMap<&'static str, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    cmd: RefCell<Option<CmdState>>,
    perf: bool,
    perf_stats: RefCell<HashMap<&'static str, KernelStats>>,
    /// Total GPU execution time (seconds) accumulated across all flushed command buffers.
    /// Used to compute encode/sync overhead = wall_time - gpu_time.
    gpu_secs_total: RefCell<f64>,
}


impl MetalBackend {
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let queue = device.newCommandQueue().expect("No command queue");

        let library = load_metal_library(&device);

        let names: &[&str] = &[
            "embed_f32",
            "embed_bf16",
            "embed_f16",
            "embed_q4k",
            "embed_q6k",
            "matvec_f32_simd",
            "matvec_bf16_simd",
            "matvec_f16_simd",
            "matvec_q4_0_simd",
            "matvec_q8_0_simd",
            "matvec_q4k_simd",
            "matvec_q6k_simd",
            "rms_norm",
            "rope",
            "softmax_kernel",
            "silu_inplace",
            "add_vecs",
            "mul_vecs",
            "copy_offset",
            "flash_attention",
            "embed_batch_f32",
            "embed_batch_bf16",
            "embed_batch_f16",
            "embed_batch_q4k",
            "embed_batch_q6k",
            "rms_norm_batch",
            "rope_batch",
            "gemm_f32",
            "gemm_bf16",
            "gemm_f16",
            "gemm_q4_0",
            "gemm_q8_0",
            "gemm_q4k",
            "gemm_q6k",
            "causal_attention",
            "argmax_bf16",
        ];

        let t0 = std::time::Instant::now();
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
        let pipeline_ms = t0.elapsed().as_millis();

        let perf = std::env::var("LLM_PERF").is_ok();
        if perf {
            eprintln!("Metal backend initialized in {pipeline_ms}ms (LLM_PERF=1: per-kernel GPU timing enabled)");
        } else {
            eprintln!("Metal backend initialized in {pipeline_ms}ms");
        }

        MetalBackend {
            device,
            queue,
            pipelines,
            cmd: RefCell::new(None),
            perf,
            perf_stats: RefCell::new(HashMap::new()),
            gpu_secs_total: RefCell::new(0.0),
        }
    }

    fn flush(&self) {
        let mut state = self.cmd.borrow_mut();
        if let Some(s) = state.take() {
            s.encoder.endEncoding();
            s.cmd_buf.commit();
            s.cmd_buf.waitUntilCompleted();
            *self.gpu_secs_total.borrow_mut() += s.cmd_buf.GPUEndTime() - s.cmd_buf.GPUStartTime();
        }
    }

    /// Flush and return GPU execution time in seconds (from GPU timestamps).
    fn flush_gpu_timed(&self) -> f64 {
        let mut state = self.cmd.borrow_mut();
        if let Some(s) = state.take() {
            s.encoder.endEncoding();
            s.cmd_buf.commit();
            s.cmd_buf.waitUntilCompleted();
            s.cmd_buf.GPUEndTime() - s.cmd_buf.GPUStartTime()
        } else {
            0.0
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

    /// Record kernel perf stats. When perf mode is on, flushes after each kernel
    /// to get accurate GPU timestamps, then accumulates stats.
    fn perf_log(&self, kernel: &'static str, bytes: usize, flops: usize) {
        if !self.perf {
            return;
        }
        let gpu_secs = self.flush_gpu_timed();
        let mut stats = self.perf_stats.borrow_mut();
        let entry = stats.entry(kernel).or_default();
        entry.gpu_secs += gpu_secs;
        entry.bytes += bytes;
        entry.flops += flops;
        entry.count += 1;
    }

    /// Print accumulated perf stats and reset.
    fn perf_dump(&self) {
        if !self.perf {
            return;
        }
        let mut stats = self.perf_stats.borrow_mut();
        if stats.is_empty() {
            return;
        }

        let mut entries: Vec<_> = stats.drain().collect();
        entries.sort_by(|a, b| b.1.gpu_secs.partial_cmp(&a.1.gpu_secs).unwrap());

        let total_gpu: f64 = entries.iter().map(|(_, s)| s.gpu_secs).sum();
        let total_bytes: usize = entries.iter().map(|(_, s)| s.bytes).sum();

        eprintln!("  {:<24} {:>5} {:>8}  {:>8}  {:>12}  {:>12}", "kernel", "calls", "GPU time", "% total", "BW (% SOL)", "GFLOP/s (%)");
        eprintln!("  {}", "-".repeat(80));

        for (kernel, s) in &entries {
            let time_str = if s.gpu_secs >= 0.001 {
                format!("{:.2}ms", s.gpu_secs * 1000.0)
            } else {
                format!("{:.0}us", s.gpu_secs * 1_000_000.0)
            };
            let pct_total = s.gpu_secs / total_gpu * 100.0;

            let bw_gb_s = s.bytes as f64 / s.gpu_secs / 1e9;
            let bw_pct = bw_gb_s / GPU_MEM_BW_GB_S * 100.0;
            let bw_str = format!("{:.0}GB/s ({:.0}%)", bw_gb_s, bw_pct);

            let flops_str = if s.flops > 0 {
                let peak_tflops = if kernel.contains("f16") || kernel.contains("bf16") {
                    GPU_FP16_TFLOPS
                } else {
                    GPU_FP32_TFLOPS
                };
                let gflops = s.flops as f64 / s.gpu_secs / 1e9;
                let flops_pct = gflops / (peak_tflops * 1000.0) * 100.0;
                format!("{:.0} ({:.1}%)", gflops, flops_pct)
            } else {
                "-".to_string()
            };

            eprintln!(
                "  {:<24} {:>5} {:>8}  {:>7.1}%  {:>12}  {:>12}",
                kernel, s.count, time_str, pct_total, bw_str, flops_str
            );
        }

        let total_bw = total_bytes as f64 / total_gpu / 1e9;
        let total_bw_pct = total_bw / GPU_MEM_BW_GB_S * 100.0;
        let total_str = if total_gpu >= 0.001 {
            format!("{:.2}ms", total_gpu * 1000.0)
        } else {
            format!("{:.0}us", total_gpu * 1_000_000.0)
        };
        eprintln!("  {}", "-".repeat(80));
        eprintln!(
            "  {:<24} {:>5} {:>8}  {:>7}   {:>12}",
            "TOTAL", "", total_str, "100.0%",
            format!("{:.0}GB/s ({:.0}%)", total_bw, total_bw_pct)
        );
        eprintln!();
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
    let m = buf.inner.downcast_ref::<MetalBuffer>().unwrap();
    unsafe { enc.setBuffer_offset_atIndex(Some(&m.buf), m.offset, idx) };
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
    fn gpu_secs_total(&self) -> f64 { *self.gpu_secs_total.borrow() }
    fn reset_gpu_secs(&self) { *self.gpu_secs_total.borrow_mut() = 0.0; }

    fn upload_tensor(&self, tv: &TensorView) -> DeviceBuffer {
        // Zero-copy upload: wrap the GGUF buffer region directly. UMA on Apple Silicon
        // means the GPU sees the same memory the CPU loaded the file into.
        // Metal requires the pointer to be page-aligned and the length to be a page
        // multiple, so we round the start down to the nearest page and pass an
        // in-page byte offset on every binding. The trailing pages of the GGUF
        // allocation are valid (page-padded by AlignedBuf) so the rounded-up length
        // never reads past the end.
        let ptr = tv.data.as_ptr() as usize;
        let aligned = ptr & !(PAGE_SIZE - 1);
        let offset = ptr - aligned;
        let length = (offset + tv.data.len() + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        let buf = unsafe {
            self.device.newBufferWithBytesNoCopy_length_options_deallocator(
                NonNull::new_unchecked(aligned as *mut _),
                length,
                MTLResourceOptions::StorageModeShared,
                None, // GGUF buffer outlives the model — Metal must not free it.
            )
        }
        .expect("Failed to create no-copy buffer");
        DeviceBuffer {
            inner: Box::new(MetalBuffer { buf, offset }),
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
            inner: Box::new(MetalBuffer { buf, offset: 0 }),
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
        // Optimized kernels: NR rows per simdgroup, NSG simdgroups per threadgroup
        let (nr, nsg) = match weight.dtype {
            DType::Q4K | DType::Q6K => (2, 4),
            DType::BF16 | DType::F16 => (2, 4),
            _ => (1, 8), // original: 1 row per simdgroup, 8 simdgroups
        };
        let rows_per_tg = nr * nsg;
        let n_tg = (out_f as usize + rows_per_tg - 1) / rows_per_tg;
        self.encode_groups(
            kernel,
            sz(n_tg, 1, 1),
            sz(nsg * 32, 1, 1),
            0,
            |enc| unsafe {
                set_buf(enc, weight, 0);
                set_buf(enc, input, 1);
                set_buf(enc, out, 2);
                set_u32(enc, in_f, 3);
                set_u32(enc, out_f, 4);
            },
        );

        let weight_bytes = weight.dtype.storage_size(in_f as usize * out_f as usize);
        let input_bytes = input.dtype.storage_size(in_f as usize);
        let output_bytes = out.dtype.storage_size(out_f as usize);
        self.perf_log(kernel, weight_bytes + input_bytes + output_bytes, 2 * in_f as usize * out_f as usize);
    }

    fn matmul(&self, out: &mut DeviceBuffer, weight: &DeviceBuffer, input: &DeviceBuffer) {
        let in_f = weight.shape[0] as u32;
        let out_f = weight.shape[1] as u32;
        let seq_len = input.shape[1] as u32;

        // For quantized types, GEMM is slower than sequential matvec because
        // quantized weights are tiny (~0.5 bytes/elem for Q4K) so F32 input reads
        // dominate and weight reuse provides no benefit. Dispatch the optimized
        // matvec kernel for each sequence position instead.
        let use_gemm = matches!(weight.dtype, DType::F32 | DType::BF16 | DType::F16);

        if use_gemm {
            let kernel = match weight.dtype {
                DType::F32 => "gemm_f32",
                DType::BF16 => "gemm_bf16",
                DType::F16 => "gemm_f16",
                _ => unreachable!(),
            };

            // BF16 uses MMA kernel (simdgroup_matrix, 32×8 tiles), others scalar (4×4 tiles)
            let (nsg, n_tg_x, n_tg_y): (usize, usize, usize) = match weight.dtype {
                DType::BF16 => {
                    let nsg = 4usize;
                    let nr = 2usize; // MMA_NR: row tiles per simdgroup
                    let tm = nsg * nr * 8; // MMA_TM: 64 rows per TG
                    (nsg,
                     (out_f as usize + tm - 1) / tm,
                     (seq_len as usize + 7) / 8)
                }
                _ => {
                    let nsg = 4usize; // GEMM_NSG
                    let tile_s = 4usize; // GEMM_TILE_S
                    (nsg,
                     (out_f as usize + nsg - 1) / nsg,
                     (seq_len as usize + tile_s - 1) / tile_s)
                }
            };

            self.encode_groups(
                kernel,
                sz(n_tg_x, n_tg_y, 1),
                sz(nsg * 32, 1, 1),
                0,  // shader uses static threadgroup arrays
                |enc| unsafe {
                    set_buf(enc, weight, 0);
                    set_buf(enc, input, 1);
                    set_buf(enc, out, 2);
                    set_u32(enc, in_f, 3);
                    set_u32(enc, out_f, 4);
                    set_u32(enc, seq_len, 5);
                },
            );

            let kernel_label = kernel;
            let weight_bytes = weight.dtype.storage_size(in_f as usize * out_f as usize);
            let input_bytes = input.dtype.storage_size(in_f as usize * seq_len as usize);
            let output_bytes = out.dtype.storage_size(out_f as usize * seq_len as usize);
            self.perf_log(kernel_label, weight_bytes + input_bytes + output_bytes,
                2 * in_f as usize * out_f as usize * seq_len as usize);
        } else {
            // Sequential matvec: dispatch one matvec per sequence position.
            // All positions are independent (same weights, different input columns),
            // so we skip barriers between dispatches and add one at the end.
            let kernel = match weight.dtype {
                DType::Q4_0 => "matvec_q4_0_simd",
                DType::Q8_0 => "matvec_q8_0_simd",
                DType::Q4K => "matvec_q4k_simd",
                DType::Q6K => "matvec_q6k_simd",
                _ => unimplemented!("matmul for {:?}", weight.dtype),
            };
            let (nr, nsg) = match weight.dtype {
                DType::Q4K | DType::Q6K => (2, 4),
                _ => (1, 8),
            };
            let rows_per_tg = nr * nsg;
            let n_tg = (out_f as usize + rows_per_tg - 1) / rows_per_tg;
            let in_stride = input.dtype.storage_size(in_f as usize);
            let out_stride = out.dtype.storage_size(out_f as usize);
            let groups = sz(n_tg, 1, 1);
            let tg = sz(nsg * 32, 1, 1);

            let in_m = input.inner.downcast_ref::<MetalBuffer>().unwrap();
            let out_m = out.inner.downcast_ref::<MetalBuffer>().unwrap();
            let w_m = weight.inner.downcast_ref::<MetalBuffer>().unwrap();

            let mut state = self.cmd.borrow_mut();
            let s = state.get_or_insert_with(|| {
                let cb = self.queue.commandBuffer().unwrap();
                let enc = cb.computeCommandEncoder().unwrap();
                CmdState { cmd_buf: cb, encoder: enc }
            });
            let pso = &self.pipelines[kernel];

            unsafe {
                s.encoder.setComputePipelineState(pso);
                s.encoder.setBuffer_offset_atIndex(Some(&w_m.buf), w_m.offset, 0);
                set_u32(&s.encoder, in_f, 3);
                set_u32(&s.encoder, out_f, 4);

                for pos in 0..seq_len as usize {
                    s.encoder.setBuffer_offset_atIndex(Some(&in_m.buf), in_m.offset + pos * in_stride, 1);
                    s.encoder.setBuffer_offset_atIndex(Some(&out_m.buf), out_m.offset + pos * out_stride, 2);
                    s.encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tg);
                }
                s.encoder.memoryBarrierWithScope(MTLBarrierScope::Buffers);
            }
            drop(state);

            let weight_bytes = weight.dtype.storage_size(in_f as usize * out_f as usize);
            let input_bytes = input.dtype.storage_size(in_f as usize);
            let output_bytes = out.dtype.storage_size(out_f as usize);
            let total_bytes = weight_bytes * seq_len as usize + (input_bytes + output_bytes) * seq_len as usize;
            self.perf_log(kernel, total_bytes,
                2 * in_f as usize * out_f as usize * seq_len as usize);
        }
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

        let d = dim as usize;
        let act_bytes = input.dtype.storage_size(d);
        self.perf_log("rms_norm", act_bytes * 2 + weight.dtype.storage_size(d), 5 * d);
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

        let total_n = q.n_elements() + k.n_elements();
        let elem_size = q.dtype.storage_size(1);
        self.perf_log("rope", total_n * elem_size * 2, total_n / 2 * 10);
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

        self.perf_log("softmax", x.dtype.storage_size(len) * 2, 5 * len);
    }

    fn silu(&self, x: &mut DeviceBuffer) {
        let n = x.n_elements() as u32;
        self.encode("silu_inplace", sz(n as usize, 1, 1), sz(256, 1, 1), 0, |enc| unsafe {
            set_buf(enc, x, 0);
            set_u32(enc, n, 1);
        });

        self.perf_log("silu", x.dtype.storage_size(n as usize) * 2, 4 * n as usize);
    }

    fn mul(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer) {
        let n = a.n_elements() as u32;
        self.encode("mul_vecs", sz(n as usize, 1, 1), sz(256, 1, 1), 0, |enc| unsafe {
            set_buf(enc, a, 0);
            set_buf(enc, b, 1);
            set_buf(enc, out, 2);
            set_u32(enc, n, 3);
        });

        self.perf_log("mul", a.dtype.storage_size(n as usize) * 3, n as usize);
    }

    fn add(&self, out: &mut DeviceBuffer, a: &DeviceBuffer, b: &DeviceBuffer) {
        let n = a.n_elements() as u32;
        self.encode("add_vecs", sz(n as usize, 1, 1), sz(256, 1, 1), 0, |enc| unsafe {
            set_buf(enc, a, 0);
            set_buf(enc, b, 1);
            set_buf(enc, out, 2);
            set_u32(enc, n, 3);
        });

        self.perf_log("add", a.dtype.storage_size(n as usize) * 3, n as usize);
    }

    fn argmax(&self, logits: &DeviceBuffer) -> u32 {
        assert!(matches!(logits.dtype, DType::BF16), "argmax expects BF16 logits");
        let n = logits.n_elements() as u32;
        let out = self.alloc(&[1], DType::I32);
        self.encode_groups(
            "argmax_bf16",
            sz(1, 1, 1),
            sz(256, 1, 1),
            0,
            |enc| unsafe {
                set_buf(enc, logits, 0);
                set_buf(enc, &out, 1);
                set_u32(enc, n, 2);
            },
        );
        self.flush();
        let m = out.inner.downcast_ref::<MetalBuffer>().unwrap();
        let ptr = unsafe { (m.buf.contents().as_ptr() as *const u8).add(m.offset) as *const u32 };
        unsafe { *ptr }
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

        let row_bytes = table.dtype.storage_size(dim as usize);
        self.perf_log(kernel, row_bytes + out.dtype.storage_size(dim as usize), 0);
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
        let fa_block = 32;
        let n_simdgroups = (head_dim + 31) / 32;
        let shared_mem = (fa_block + n_simdgroups) * 4;
        self.encode_groups(
            "flash_attention",
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

        let seq_len = pos + 1;
        let elem = q.dtype.storage_size(1);
        let bytes = n_heads * head_dim * elem
            + seq_len * n_kv_heads * head_dim * elem * 2
            + n_heads * head_dim * elem;
        let flops = n_heads * seq_len * head_dim * 4;
        self.perf_log("flash_attention", bytes, flops);
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

        self.perf_log("copy_offset", src.dtype.storage_size(count as usize) * 2, 0);
    }

    fn read_to_vec_f32(&self, buf: &DeviceBuffer) -> Vec<f32> {
        if self.perf {
            // In perf mode, all kernels already flushed individually.
            // Dump the accumulated stats.
            self.perf_dump();
        } else {
            self.flush();
        }
        let m = buf.inner.downcast_ref::<MetalBuffer>().unwrap();
        let n = buf.n_elements();
        let base = unsafe { (m.buf.contents().as_ptr() as *const u8).add(m.offset) };
        match buf.dtype {
            DType::BF16 => {
                let ptr = base as *const u16;
                let raw = unsafe { std::slice::from_raw_parts(ptr, n) };
                raw.iter().map(|&bits| f32::from_bits((bits as u32) << 16)).collect()
            }
            _ => {
                let ptr = base as *const f32;
                unsafe { std::slice::from_raw_parts(ptr, n) }.to_vec()
            }
        }
    }

    fn embed_batch(&self, out: &mut DeviceBuffer, table: &DeviceBuffer, token_ids: &[u32]) {
        let seq_len = token_ids.len();
        let token_buf = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(token_ids.as_ptr() as *mut _),
                token_ids.len() * 4,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("Failed to create token_ids buffer");

        let kernel = match table.dtype {
            DType::F32 => "embed_batch_f32",
            DType::BF16 => "embed_batch_bf16",
            DType::F16 => "embed_batch_f16",
            DType::Q4K => "embed_batch_q4k",
            DType::Q6K => "embed_batch_q6k",
            _ => unimplemented!("embed_batch for {:?}", table.dtype),
        };
        let dim = table.shape[0] as u32;
        let seq_u32 = seq_len as u32;

        self.encode(
            kernel,
            sz(dim as usize, seq_len, 1),
            sz(256, 1, 1),
            0,
            |enc| unsafe {
                set_buf(enc, table, 0);
                enc.setBuffer_offset_atIndex(Some(&token_buf), 0, 1);
                set_buf(enc, out, 2);
                set_u32(enc, dim, 3);
                set_u32(enc, seq_u32, 4);
            },
        );
    }

    fn rms_norm_batch(
        &self,
        out: &mut DeviceBuffer,
        input: &DeviceBuffer,
        weight: &DeviceBuffer,
        eps: f32,
        seq_len: usize,
    ) {
        let dim = weight.n_elements() as u32;
        let tg = 1024usize.min(dim as usize);
        let shared = (((tg) + 31) / 32) * 4;
        self.encode_groups(
            "rms_norm_batch",
            sz(seq_len, 1, 1),
            sz(tg, 1, 1),
            shared,
            |enc| unsafe {
                set_buf(enc, input, 0);
                set_buf(enc, weight, 1);
                set_buf(enc, out, 2);
                set_u32(enc, dim, 3);
                set_f32(enc, eps, 4);
            },
        );
    }

    fn rope_batch(
        &self,
        q: &mut DeviceBuffer,
        k: &mut DeviceBuffer,
        start_pos: usize,
        seq_len: usize,
        head_dim: usize,
        rope_theta: f32,
    ) {
        let q_total = q.n_elements();
        let q_pairs_per_row = q_total / seq_len / 2;
        let q_row_stride = q_total / seq_len;
        self.encode(
            "rope_batch",
            sz(q_pairs_per_row, seq_len, 1),
            sz(256, 1, 1),
            0,
            |enc| unsafe {
                set_buf(enc, q, 0);
                set_u32(enc, start_pos as u32, 1);
                set_u32(enc, head_dim as u32, 2);
                set_f32(enc, rope_theta, 3);
                set_u32(enc, q_pairs_per_row as u32, 4);
                set_u32(enc, q_row_stride as u32, 5);
            },
        );

        let k_total = k.n_elements();
        let k_pairs_per_row = k_total / seq_len / 2;
        let k_row_stride = k_total / seq_len;
        self.encode(
            "rope_batch",
            sz(k_pairs_per_row, seq_len, 1),
            sz(256, 1, 1),
            0,
            |enc| unsafe {
                set_buf(enc, k, 0);
                set_u32(enc, start_pos as u32, 1);
                set_u32(enc, head_dim as u32, 2);
                set_f32(enc, rope_theta, 3);
                set_u32(enc, k_pairs_per_row as u32, 4);
                set_u32(enc, k_row_stride as u32, 5);
            },
        );
    }

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
    ) {
        let kv_dim = (n_kv_heads * head_dim) as u32;
        let fa_block = 32;
        let n_simdgroups = (head_dim + 31) / 32;
        let shared_mem = (fa_block + n_simdgroups) * 4;
        let n_tg = n_heads * seq_len;
        self.encode_groups(
            "causal_attention",
            sz(n_tg, 1, 1),
            sz(head_dim, 1, 1),
            shared_mem,
            |enc| unsafe {
                set_buf(enc, q, 0);
                set_buf(enc, k_cache, 1);
                set_buf(enc, v_cache, 2);
                set_buf(enc, out, 3);
                set_u32(enc, start_pos as u32, 4);
                set_u32(enc, seq_len as u32, 5);
                set_u32(enc, n_heads as u32, 6);
                set_u32(enc, n_kv_heads as u32, 7);
                set_u32(enc, head_dim as u32, 8);
                set_u32(enc, kv_dim, 9);
                set_u32(enc, n_heads as u32, 10);
            },
        );
    }

    fn sync(&self) {
        self.flush();
    }

    fn write_from_f32(&self, buf: &mut DeviceBuffer, data: &[f32]) {
        self.flush();
        let m = buf.inner.downcast_ref::<MetalBuffer>().unwrap();
        let base = unsafe { (m.buf.contents().as_ptr() as *mut u8).add(m.offset) };
        match buf.dtype {
            DType::BF16 => {
                let ptr = base as *mut u16;
                for (i, &val) in data.iter().enumerate() {
                    unsafe { *ptr.add(i) = (val.to_bits() >> 16) as u16; }
                }
            }
            _ => {
                let ptr = base as *mut f32;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                }
            }
        }
    }
}

