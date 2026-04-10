use std::cell::RefCell;
use std::collections::HashMap;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSURL};
use objc2_metal::*;

use crate::backend::Backend;
use crate::tensor::{DType, RopeLayout, TensorView};

pub struct MetalBuffer {
    buf: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Byte offset of the logical data within `buf`. Non-zero only for zero-copy
    /// weight uploads where the underlying GGUF tensor isn't page-aligned and we
    /// wrapped a page-rounded-down region.
    offset: usize,
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl MetalBuffer {
    pub fn n_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Number of columns. 1D shapes (decode) → 1, 2D shapes (prefill) → shape[1].
    /// This is the only thing the backend needs to tell decode and prefill apart.
    pub fn seq_len(&self) -> usize {
        if self.shape.len() > 1 { self.shape[1] as usize } else { 1 }
    }

}

const PAGE_SIZE: usize = 16384;

/// Repack GGUF Q4K (144 B/block) into the device layout (160 B/block):
/// - 8 half d_sc[k] = d * sc[k], with odd k pre-divided by 16 so the kernel
///   can use `byte & 0xF0` directly for the high nibble (no shift).
/// - 8 half d_m [k] = dmin * m[k]
/// - 32 u32 qs_t[j]: transposed nibble plane. qs_t[j] packs the 8 nibbles
///   at lane `j` across all 8 sub-blocks into one u32:
///     byte[p] of qs_t[j] == qs[p*32 + j]   for p in 0..4
///   i.e. the low nibble of byte[p] is sub-block (2p)'s nibble at lane j,
///   and the high nibble of byte[p] is sub-block (2p+1)'s nibble at lane j.
fn repack_q4k(src: &[u8]) -> Vec<u8> {
    assert!(src.len() % 144 == 0, "Q4K source must be a multiple of 144 B");
    let n_blocks = src.len() / 144;
    let mut out = vec![0u8; n_blocks * 160];
    for (blk_i, blk) in src.chunks_exact(144).enumerate() {
        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
        let scales = &blk[4..16];
        let qs = &blk[16..144];
        let dst = &mut out[blk_i * 160..(blk_i + 1) * 160];
        for k in 0..8 {
            let (sc, m);
            if k < 4 {
                sc = (scales[k] & 63) as u32;
                m  = (scales[k + 4] & 63) as u32;
            } else {
                sc = ((scales[k + 4] & 0x0F) as u32) | (((scales[k - 4] >> 6) as u32) << 4);
                m  = ((scales[k + 4] >> 4) as u32)   | (((scales[k] >> 6) as u32)     << 4);
            }
            let odd_fold = if k & 1 == 1 { 1.0f32 / 16.0 } else { 1.0 };
            let d_sc = half::f16::from_f32(d * sc as f32 * odd_fold).to_le_bytes();
            let d_m  = half::f16::from_f32(dmin * m as f32).to_le_bytes();
            dst[k * 2..k * 2 + 2].copy_from_slice(&d_sc);
            dst[16 + k * 2..16 + k * 2 + 2].copy_from_slice(&d_m);
        }
        for j in 0..32 {
            let v: u32 = (qs[j] as u32)
                | ((qs[32 + j] as u32) << 8)
                | ((qs[64 + j] as u32) << 16)
                | ((qs[96 + j] as u32) << 24);
            dst[32 + j * 4..32 + j * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// Start an Xcode GPU capture to `path`, finalized on `stop_capture`.
/// Requires `METAL_CAPTURE_ENABLED=1` in the environment at launch so the
/// Metal framework enables the capture machinery for a non-Xcode binary.
fn start_capture(device: &ProtocolObject<dyn MTLDevice>, path: &str) {
    // Clear any stale file so the capture manager doesn't refuse to overwrite.
    let _ = std::fs::remove_file(path);
    let url = NSURL::fileURLWithPath(&NSString::from_str(path));
    let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
    if !manager.supportsDestination(MTLCaptureDestination::GPUTraceDocument) {
        eprintln!(
            "LLM_CAPTURE: GPUTraceDocument destination unsupported. \
             Launch with METAL_CAPTURE_ENABLED=1."
        );
        return;
    }
    let desc = MTLCaptureDescriptor::new();
    let device_obj: &objc2::runtime::AnyObject = device.as_ref();
    unsafe {
        desc.setCaptureObject(Some(device_obj));
        desc.setDestination(MTLCaptureDestination::GPUTraceDocument);
        desc.setOutputURL(Some(&url));
    }
    if let Err(e) = manager.startCaptureWithDescriptor_error(&desc) {
        eprintln!("LLM_CAPTURE: startCapture failed: {e}");
        return;
    }
    eprintln!("LLM_CAPTURE: recording → {path}");
}

fn stop_capture() {
    let manager = unsafe { MTLCaptureManager::sharedCaptureManager() };
    manager.stopCapture();
}

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
    /// Set iff `LLM_CAPTURE=<path>` was set at init. On Drop we stop the
    /// capture so the `.gputrace` file is finalized and ready to open in Xcode.
    capturing: bool,
}


enum Dispatch {
    Threads(MTLSize),
    Groups(MTLSize),
}

impl MetalBackend {
    pub fn new() -> Self {
        let device = MTLCreateSystemDefaultDevice().expect("No Metal device");
        let queue = device.newCommandQueue().expect("No command queue");

        let library = load_metal_library(&device);

        let names: &[&str] = &[
            "matvec_f32_simd",
            "matvec_bf16_simd",
            "matvec_f16_simd",
            "matvec_q4_0_simd",
            "matvec_q8_0_simd",
            "matvec_q4k_simd",
            "matvec_q6k_simd",
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
            "gemm_q4k",
            "gemm_q4k_pipe",
            "gemm_q6k",
            "causal_attention",
            "argmax",
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

        // Optional GPU capture for Xcode's shader profiler. Set
        //   LLM_CAPTURE=/tmp/foo.gputrace METAL_CAPTURE_ENABLED=1
        // and every dispatch issued until this backend is dropped is recorded
        // into the trace file. Open the result in Xcode to get per-line ALU /
        // memory cost for each kernel.
        let capturing = match std::env::var("LLM_CAPTURE") {
            Ok(path) => { start_capture(&device, &path); true }
            Err(_) => false,
        };

        MetalBackend {
            device,
            queue,
            pipelines,
            cmd: RefCell::new(None),
            perf,
            perf_stats: RefCell::new(HashMap::new()),
            gpu_secs_total: RefCell::new(0.0),
            capturing,
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

    fn dispatch(
        &self,
        pipeline: &'static str,
        mode: Dispatch,
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
            match mode {
                Dispatch::Threads(grid) => s.encoder.dispatchThreads_threadsPerThreadgroup(grid, tg),
                Dispatch::Groups(groups) => s.encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, tg),
            }
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

unsafe fn bind_buffer(
    enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buf: &MetalBuffer,
    idx: usize,
) {
    unsafe { enc.setBuffer_offset_atIndex(Some(&buf.buf), buf.offset, idx) };
}

unsafe fn bind_u32(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, val: u32, idx: usize) {
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::new_unchecked(&val as *const u32 as *mut _),
            4,
            idx,
        )
    };
}

unsafe fn bind_f32(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, val: f32, idx: usize) {
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::new_unchecked(&val as *const f32 as *mut _),
            4,
            idx,
        )
    };
}

impl Backend for MetalBackend {
    type Buffer = MetalBuffer;

    fn gpu_secs_total(&self) -> f64 { *self.gpu_secs_total.borrow() }
    fn reset_gpu_secs(&self) { *self.gpu_secs_total.borrow_mut() = 0.0; }

    fn upload_tensor(&self, tv: &TensorView) -> MetalBuffer {
        // Q4K: repack into the device layout (160 B/block) with scales and
        // mins pre-multiplied by d/dmin, and the nibble plane transposed so
        // one u32 load gives each lane its 8 sub-block nibbles. Costs 11%
        // more VRAM than the on-disk 144 B/block layout; gains 2× kernel BW.
        if tv.dtype == DType::Q4K {
            let repacked = repack_q4k(tv.data);
            let buf = self
                .device
                .newBufferWithLength_options(repacked.len(), MTLResourceOptions::StorageModeShared)
                .expect("Failed to alloc Q4K repack buffer");
            unsafe {
                let dst = buf.contents().as_ptr() as *mut u8;
                std::ptr::copy_nonoverlapping(repacked.as_ptr(), dst, repacked.len());
            }
            return MetalBuffer { buf, offset: 0, dtype: tv.dtype, shape: tv.shape.to_vec() };
        }

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
        MetalBuffer { buf, offset, dtype: tv.dtype, shape: tv.shape.to_vec() }
    }

    fn alloc(&self, shape: &[u64], dtype: DType) -> MetalBuffer {
        let n: usize = shape.iter().map(|&d| d as usize).product();
        let size = dtype.storage_size(n);
        let buf = self
            .device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .expect("Failed to alloc buffer");
        MetalBuffer { buf, offset: 0, dtype, shape: shape.to_vec() }
    }

    fn matmul(&self, out: &mut MetalBuffer, weight: &MetalBuffer, input: &MetalBuffer) {
        let in_f = weight.shape[0] as u32;
        let out_f = weight.shape[1] as u32;
        let seq_len = input.seq_len() as u32;

        // Decode (single column): always dispatch the optimized SIMD matvec.
        if seq_len == 1 {
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
            let (nr, nsg) = match weight.dtype {
                DType::Q4K => (4, 8),
                DType::Q6K => (2, 4),
                DType::BF16 | DType::F16 => (2, 4),
                _ => (1, 8),
            };
            let rows_per_tg = nr * nsg;
            let n_tg = (out_f as usize + rows_per_tg - 1) / rows_per_tg;
            self.dispatch(
                kernel,
                Dispatch::Groups(sz(n_tg, 1, 1)),
                sz(nsg * 32, 1, 1),
                0,
                |enc| unsafe {
                    bind_buffer(enc, weight, 0);
                    bind_buffer(enc, input, 1);
                    bind_buffer(enc, out, 2);
                    bind_u32(enc, in_f, 3);
                    bind_u32(enc, out_f, 4);
                },
            );

            let weight_bytes = weight.dtype.storage_size(in_f as usize * out_f as usize);
            let input_bytes = input.dtype.storage_size(in_f as usize);
            let output_bytes = out.dtype.storage_size(out_f as usize);
            self.perf_log(kernel, weight_bytes + input_bytes + output_bytes, 2 * in_f as usize * out_f as usize);
            return;
        }

        let use_gemm = matches!(
            weight.dtype,
            DType::F32 | DType::BF16 | DType::F16 | DType::Q4K | DType::Q6K
        );

        if use_gemm {
            // Q4K has two GEMM kernels with different sweet spots:
            //  - gemm_q4k: register-tiled, low overhead, best for small seq.
            //  - gemm_q4k_pipe: simdgroup_matrix-based, larger TG, higher
            //    fixed overhead but ~15% faster for big prefill (seq≥32).
            // Crossover empirically lives around seq=16–32 on M1 Pro.
            let q4k_use_pipe = matches!(weight.dtype, DType::Q4K) && seq_len >= 32;
            let kernel = match weight.dtype {
                DType::F32 => "gemm_f32",
                DType::BF16 => "gemm_bf16",
                DType::F16 => "gemm_f16",
                DType::Q4K if q4k_use_pipe => "gemm_q4k_pipe",
                DType::Q4K => "gemm_q4k",
                DType::Q6K => "gemm_q6k",
                _ => unreachable!(),
            };

            // BF16 uses MMA kernel (simdgroup_matrix, 32×8 tiles). Q4K/Q6K reuse
            // matvec tiling (NR*NSG rows × TILE_S seq columns per TG). F32/F16
            // use the scalar GEMM (4×4 tiles).
            let (nsg, n_tg_x, n_tg_y): (usize, usize, usize) = match weight.dtype {
                DType::BF16 => {
                    let nsg = 4usize;
                    let nr = 2usize; // MMA_NR: row tiles per simdgroup
                    let tm = nsg * nr * 8; // MMA_TM: 64 rows per TG
                    (nsg,
                     (out_f as usize + tm - 1) / tm,
                     (seq_len as usize + 7) / 8)
                }
                DType::Q4K if q4k_use_pipe => {
                    // gemm_q4k_pipe: Q4KP_NSG=4, Q4KP_TM=32, Q4KP_TN=16.
                    let nsg = 4usize;
                    let tm = 32usize;
                    let tn = 16usize;
                    (nsg,
                     (out_f as usize + tm - 1) / tm,
                     (seq_len as usize + tn - 1) / tn)
                }
                DType::Q4K => {
                    // gemm_q4k: NR=4, NSG=8, TILE_S=4 (small-batch path).
                    let nsg = 8usize;
                    let nr  = 4usize;
                    let tile_s = 4usize;
                    let rows_per_tg = nr * nsg;
                    (nsg,
                     (out_f as usize + rows_per_tg - 1) / rows_per_tg,
                     (seq_len as usize + tile_s - 1) / tile_s)
                }
                DType::Q6K => {
                    let nsg = 4usize;
                    let nr  = 2usize;
                    let tile_s = 4usize;
                    let rows_per_tg = nr * nsg;
                    (nsg,
                     (out_f as usize + rows_per_tg - 1) / rows_per_tg,
                     (seq_len as usize + tile_s - 1) / tile_s)
                }
                _ => {
                    let nsg = 4usize; // GEMM_NSG
                    let tile_s = 4usize; // GEMM_TILE_S
                    (nsg,
                     (out_f as usize + nsg - 1) / nsg,
                     (seq_len as usize + tile_s - 1) / tile_s)
                }
            };

            self.dispatch(
                kernel,
                Dispatch::Groups(sz(n_tg_x, n_tg_y, 1)),
                sz(nsg * 32, 1, 1),
                0,  // shader uses static threadgroup arrays
                |enc| unsafe {
                    bind_buffer(enc, weight, 0);
                    bind_buffer(enc, input, 1);
                    bind_buffer(enc, out, 2);
                    bind_u32(enc, in_f, 3);
                    bind_u32(enc, out_f, 4);
                    bind_u32(enc, seq_len, 5);
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
                DType::Q4K => (4, 8),
                DType::Q6K => (2, 4),
                _ => (1, 8),
            };
            let rows_per_tg = nr * nsg;
            let n_tg = (out_f as usize + rows_per_tg - 1) / rows_per_tg;
            let in_stride = input.dtype.storage_size(in_f as usize);
            let out_stride = out.dtype.storage_size(out_f as usize);
            let groups = sz(n_tg, 1, 1);
            let tg = sz(nsg * 32, 1, 1);

            let mut state = self.cmd.borrow_mut();
            let s = state.get_or_insert_with(|| {
                let cb = self.queue.commandBuffer().unwrap();
                let enc = cb.computeCommandEncoder().unwrap();
                CmdState { cmd_buf: cb, encoder: enc }
            });
            let pso = &self.pipelines[kernel];

            unsafe {
                s.encoder.setComputePipelineState(pso);
                s.encoder.setBuffer_offset_atIndex(Some(&weight.buf), weight.offset, 0);
                bind_u32(&s.encoder, in_f, 3);
                bind_u32(&s.encoder, out_f, 4);

                for pos in 0..seq_len as usize {
                    s.encoder.setBuffer_offset_atIndex(Some(&input.buf), input.offset + pos * in_stride, 1);
                    s.encoder.setBuffer_offset_atIndex(Some(&out.buf), out.offset + pos * out_stride, 2);
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
        out: &mut MetalBuffer,
        input: &MetalBuffer,
        weight: &MetalBuffer,
        eps: f32,
    ) {
        let dim = weight.n_elements() as u32;
        let seq_len = MetalBuffer::seq_len(input);
        let tg = 1024usize.min(dim as usize);
        let shared = (((tg) + 31) / 32) * 4;
        self.dispatch(
            "rms_norm_batch",
            Dispatch::Groups(sz(seq_len, 1, 1)),
            sz(tg, 1, 1),
            shared,
            |enc| unsafe {
                bind_buffer(enc, input, 0);
                bind_buffer(enc, weight, 1);
                bind_buffer(enc, out, 2);
                bind_u32(enc, dim, 3);
                bind_f32(enc, eps, 4);
            },
        );

        let d = dim as usize;
        let act_bytes = input.dtype.storage_size(d * seq_len);
        self.perf_log("rms_norm", act_bytes * 2 + weight.dtype.storage_size(d), 5 * d * seq_len);
    }

    fn rms_norm_heads(&self, x: &mut MetalBuffer, weight: &MetalBuffer, eps: f32) {
        // Same shader as rms_norm; in-place, one group per head_dim chunk.
        let dim = weight.n_elements() as u32;
        let n_groups = x.n_elements() / dim as usize;
        let tg = 1024usize.min(dim as usize);
        let shared = (((tg) + 31) / 32) * 4;
        self.dispatch(
            "rms_norm_batch",
            Dispatch::Groups(sz(n_groups, 1, 1)),
            sz(tg, 1, 1),
            shared,
            |enc| unsafe {
                bind_buffer(enc, x, 0);
                bind_buffer(enc, weight, 1);
                bind_buffer(enc, x, 2);
                bind_u32(enc, dim, 3);
                bind_f32(enc, eps, 4);
            },
        );

        let d = dim as usize;
        let act_bytes = x.dtype.storage_size(d * n_groups);
        self.perf_log("rms_norm_heads", act_bytes * 2 + weight.dtype.storage_size(d), 5 * d * n_groups);
    }

    fn rope(
        &self,
        q: &mut MetalBuffer,
        k: &mut MetalBuffer,
        start_pos: usize,
        head_dim: usize,
        rope_theta: f32,
        layout: RopeLayout,
    ) {
        let seq_len = MetalBuffer::seq_len(q);
        let neox_flag: u32 = match layout {
            RopeLayout::SplitHalf => 1,
            RopeLayout::Interleaved => 0,
        };

        let dispatch = |buf: &mut MetalBuffer| {
            let total = buf.n_elements();
            let row_stride = total / seq_len;
            let pairs_per_row = row_stride / 2;
            self.dispatch(
                "rope_batch",
                Dispatch::Threads(sz(pairs_per_row, seq_len, 1)),
                sz(256, 1, 1),
                0,
                |enc| unsafe {
                    bind_buffer(enc, buf, 0);
                    bind_u32(enc, start_pos as u32, 1);
                    bind_u32(enc, head_dim as u32, 2);
                    bind_f32(enc, rope_theta, 3);
                    bind_u32(enc, pairs_per_row as u32, 4);
                    bind_u32(enc, row_stride as u32, 5);
                    bind_u32(enc, neox_flag, 6);
                },
            );
        };
        dispatch(q);
        dispatch(k);

        let total_n = q.n_elements() + k.n_elements();
        let elem_size = q.dtype.storage_size(1);
        self.perf_log("rope", total_n * elem_size * 2, total_n / 2 * 10);
    }

    fn silu(&self, x: &mut MetalBuffer) {
        let n = x.n_elements() as u32;
        self.dispatch("silu_inplace", Dispatch::Threads(sz(n as usize, 1, 1)), sz(256, 1, 1), 0, |enc| unsafe {
            bind_buffer(enc, x, 0);
            bind_u32(enc, n, 1);
        });

        self.perf_log("silu", x.dtype.storage_size(n as usize) * 2, 4 * n as usize);
    }

    fn mul(&self, out: &mut MetalBuffer, a: &MetalBuffer, b: &MetalBuffer) {
        let n = a.n_elements() as u32;
        self.dispatch("mul_vecs", Dispatch::Threads(sz(n as usize, 1, 1)), sz(256, 1, 1), 0, |enc| unsafe {
            bind_buffer(enc, a, 0);
            bind_buffer(enc, b, 1);
            bind_buffer(enc, out, 2);
            bind_u32(enc, n, 3);
        });

        self.perf_log("mul", a.dtype.storage_size(n as usize) * 3, n as usize);
    }

    fn add(&self, out: &mut MetalBuffer, a: &MetalBuffer, b: &MetalBuffer) {
        let n = a.n_elements() as u32;
        self.dispatch("add_vecs", Dispatch::Threads(sz(n as usize, 1, 1)), sz(256, 1, 1), 0, |enc| unsafe {
            bind_buffer(enc, a, 0);
            bind_buffer(enc, b, 1);
            bind_buffer(enc, out, 2);
            bind_u32(enc, n, 3);
        });

        self.perf_log("add", a.dtype.storage_size(n as usize) * 3, n as usize);
    }

    fn argmax(&self, logits: &MetalBuffer) -> u32 {
        assert!(matches!(logits.dtype, DType::BF16), "argmax expects BF16 logits");
        let n = logits.n_elements() as u32;
        let out = self.alloc(&[1], DType::I32);
        self.dispatch(
            "argmax",
            Dispatch::Groups(sz(1, 1, 1)),
            sz(256, 1, 1),
            0,
            |enc| unsafe {
                bind_buffer(enc, logits, 0);
                bind_buffer(enc, &out, 1);
                bind_u32(enc, n, 2);
            },
        );
        self.flush();
        let ptr = unsafe { (out.buf.contents().as_ptr() as *const u8).add(out.offset) as *const u32 };
        unsafe { *ptr }
    }

    fn embed(&self, out: &mut MetalBuffer, table: &MetalBuffer, tokens: &[u32]) {
        let seq_len = tokens.len();
        let token_buf = unsafe {
            self.device.newBufferWithBytes_length_options(
                NonNull::new_unchecked(tokens.as_ptr() as *mut _),
                tokens.len() * 4,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("Failed to create tokens buffer");

        let kernel = match table.dtype {
            DType::F32 => "embed_batch_f32",
            DType::BF16 => "embed_batch_bf16",
            DType::F16 => "embed_batch_f16",
            DType::Q4K => "embed_batch_q4k",
            DType::Q6K => "embed_batch_q6k",
            _ => unimplemented!("embed for {:?}", table.dtype),
        };
        let dim = table.shape[0] as u32;

        self.dispatch(
            kernel,
            Dispatch::Threads(sz(dim as usize, seq_len, 1)),
            sz(256, 1, 1),
            0,
            |enc| unsafe {
                bind_buffer(enc, table, 0);
                enc.setBuffer_offset_atIndex(Some(&token_buf), 0, 1);
                bind_buffer(enc, out, 2);
                bind_u32(enc, dim, 3);
                bind_u32(enc, seq_len as u32, 4);
            },
        );

        let row_bytes = table.dtype.storage_size(dim as usize);
        self.perf_log(kernel, (row_bytes + out.dtype.storage_size(dim as usize)) * seq_len, 0);
    }

    fn attention(
        &self,
        out: &mut MetalBuffer,
        q: &MetalBuffer,
        k_cache: &MetalBuffer,
        v_cache: &MetalBuffer,
        start_pos: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        let seq_len = MetalBuffer::seq_len(q);
        let kv_dim = (n_kv_heads * head_dim) as u32;
        let fa_block = 32;
        let n_simdgroups = (head_dim + 31) / 32;
        let shared_mem = (fa_block + n_simdgroups) * 4;

        if seq_len == 1 {
            // Decode: single-query flash attention.
            self.dispatch(
                "flash_attention",
                Dispatch::Groups(sz(n_heads, 1, 1)),
                sz(head_dim, 1, 1),
                shared_mem,
                |enc| unsafe {
                    bind_buffer(enc, q, 0);
                    bind_buffer(enc, k_cache, 1);
                    bind_buffer(enc, v_cache, 2);
                    bind_buffer(enc, out, 3);
                    bind_u32(enc, start_pos as u32, 4);
                    bind_u32(enc, n_heads as u32, 5);
                    bind_u32(enc, n_kv_heads as u32, 6);
                    bind_u32(enc, head_dim as u32, 7);
                    bind_u32(enc, kv_dim, 8);
                },
            );
            let kv_len = start_pos + 1;
            let elem = q.dtype.storage_size(1);
            let bytes = n_heads * head_dim * elem * 2 + kv_len * n_kv_heads * head_dim * elem * 2;
            self.perf_log("flash_attention", bytes, n_heads * kv_len * head_dim * 4);
        } else {
            // Prefill: causal attention over `seq_len` query columns.
            let n_tg = n_heads * seq_len;
            self.dispatch(
                "causal_attention",
                Dispatch::Groups(sz(n_tg, 1, 1)),
                sz(head_dim, 1, 1),
                shared_mem,
                |enc| unsafe {
                    bind_buffer(enc, q, 0);
                    bind_buffer(enc, k_cache, 1);
                    bind_buffer(enc, v_cache, 2);
                    bind_buffer(enc, out, 3);
                    bind_u32(enc, start_pos as u32, 4);
                    bind_u32(enc, seq_len as u32, 5);
                    bind_u32(enc, n_heads as u32, 6);
                    bind_u32(enc, n_kv_heads as u32, 7);
                    bind_u32(enc, head_dim as u32, 8);
                    bind_u32(enc, kv_dim, 9);
                    bind_u32(enc, n_heads as u32, 10);
                },
            );
            let kv_len = start_pos + seq_len;
            let elem = q.dtype.storage_size(1);
            let bytes = n_heads * seq_len * head_dim * elem * 2
                + kv_len * n_kv_heads * head_dim * elem * 2;
            self.perf_log("causal_attention", bytes, n_heads * seq_len * kv_len * head_dim * 4);
        }
    }

    fn copy_into(&self, dst: &mut MetalBuffer, src: &MetalBuffer, dst_offset_elements: usize) {
        let count = src.n_elements() as u32;
        let offset = dst_offset_elements as u32;
        self.dispatch(
            "copy_offset",
            Dispatch::Threads(sz(count as usize, 1, 1)),
            sz(256, 1, 1),
            0,
            |enc| unsafe {
                bind_buffer(enc, src, 0);
                bind_buffer(enc, dst, 1);
                bind_u32(enc, offset, 2);
                bind_u32(enc, count, 3);
            },
        );

        self.perf_log("copy_offset", src.dtype.storage_size(count as usize) * 2, 0);
    }

    fn read_to_vec_f32(&self, buf: &MetalBuffer) -> Vec<f32> {
        self.flush();
        let n = buf.n_elements();
        let base = unsafe { (buf.buf.contents().as_ptr() as *const u8).add(buf.offset) };
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

    fn sync(&self) {
        if self.perf {
            self.perf_dump();
        } else {
            self.flush();
        }
    }
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        if self.capturing {
            self.flush();
            stop_capture();
            eprintln!("LLM_CAPTURE: finalized");
        }
    }
}


