#include <metal_stdlib>
using namespace metal;

// BF16 -> float: BF16 is upper 16 bits of float32.
// M1 lacks native bfloat, so we do bit manipulation.
inline float bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16);
}

// ===================== Q4_K helpers =====================
inline float2 unpack_q4k_scale(device const uchar* scales, uint sb) {
    uint sc, m;
    if (sb < 4) {
        sc = scales[sb] & 63;
        m  = scales[sb + 4] & 63;
    } else {
        sc = (scales[sb + 4] & 0x0F) | ((scales[sb - 4] >> 6) << 4);
        m  = (scales[sb + 4] >> 4)   | ((scales[sb] >> 6)     << 4);
    }
    return float2(float(sc), float(m));
}

// ===================== Q6_K helpers =====================
// Q6_K block: 210 bytes for 256 elements
// [0..128] ql (lower 4 bits), [128..192] qh (upper 2 bits),
// [192..208] scales (int8 x 16), [208..210] d (f16)

inline uchar dequant_q6k_elem(device const uchar* ql, device const uchar* qh, uint i) {
    uint half_idx = i / 128;
    uint j = i % 128;

    uchar q_lo, q_hi;
    if (j < 64) q_lo = ql[half_idx * 64 + j] & 0x0F;
    else        q_lo = ql[half_idx * 64 + j - 64] >> 4;

    if      (j < 32) q_hi = qh[half_idx * 32 + j] & 3;
    else if (j < 64) q_hi = (qh[half_idx * 32 + j - 32] >> 2) & 3;
    else if (j < 96) q_hi = (qh[half_idx * 32 + j - 64] >> 4) & 3;
    else              q_hi = (qh[half_idx * 32 + j - 96] >> 6) & 3;

    return q_lo | (q_hi << 4);
}

// ===================== SIMD Matvec =====================
// out[row] = dot(weight[row,:], input[:])
// GGML weight layout: [in_features, out_features] = out_features rows.
// One SIMD group (32 threads) per row, NSG groups per threadgroup.
// Each thread strides through the dot product; simd_sum reduces.

constant constexpr uint ROWS_PER_TG = 8;

kernel void matvec_f32_simd(
    device const float* weight  [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid * ROWS_PER_TG + sgid;
    if (row >= out_features) return;
    uint base = row * in_features;
    float sum = 0.0;
    for (uint i = lane; i < in_features; i += 32)
        sum += weight[base + i] * float(input[i]);
    sum = simd_sum(sum);
    if (lane == 0) output[row] = bfloat(sum);
}

// BF16: 2 rows per simdgroup, float4 wide loads.
constant constexpr uint BF16_NR = 2;
constant constexpr uint BF16_NSG = 4;

kernel void matvec_bf16_simd(
    device const ushort* weight [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint row_base = tgid * (BF16_NR * BF16_NSG) + sgid * BF16_NR;
    if (row_base >= out_features) return;

    float sumf[BF16_NR] = {};

    // 32 lanes * 4 elements = 128 elements per iteration
    for (uint i = lane * 4; i < in_features; i += 128) {
        float inp0 = float(input[i]);
        float inp1 = float(input[i + 1]);
        float inp2 = float(input[i + 2]);
        float inp3 = float(input[i + 3]);

        for (uint r = 0; r < BF16_NR; r++) {
            uint row = row_base + r;
            if (row >= out_features) break;
            device const ushort* wp = weight + row * in_features + i;
            sumf[r] += bf16_to_f32(wp[0]) * inp0
                     + bf16_to_f32(wp[1]) * inp1
                     + bf16_to_f32(wp[2]) * inp2
                     + bf16_to_f32(wp[3]) * inp3;
        }
    }

    for (uint r = 0; r < BF16_NR; r++) {
        uint row = row_base + r;
        if (row >= out_features) break;
        float s = simd_sum(sumf[r]);
        if (lane == 0) output[row] = bfloat(s);
    }
}

// F16: 2 rows per simdgroup, float4 wide loads.
constant constexpr uint F16_NR = 2;
constant constexpr uint F16_NSG = 4;

kernel void matvec_f16_simd(
    device const half* weight   [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint row_base = tgid * (F16_NR * F16_NSG) + sgid * F16_NR;
    if (row_base >= out_features) return;

    float sumf[F16_NR] = {};

    for (uint i = lane * 4; i < in_features; i += 128) {
        float inp0 = float(input[i]);
        float inp1 = float(input[i + 1]);
        float inp2 = float(input[i + 2]);
        float inp3 = float(input[i + 3]);

        for (uint r = 0; r < F16_NR; r++) {
            uint row = row_base + r;
            if (row >= out_features) break;
            device const half* wp = weight + row * in_features + i;
            sumf[r] += float(wp[0]) * inp0
                     + float(wp[1]) * inp1
                     + float(wp[2]) * inp2
                     + float(wp[3]) * inp3;
        }
    }

    for (uint r = 0; r < F16_NR; r++) {
        uint row = row_base + r;
        if (row >= out_features) break;
        float s = simd_sum(sumf[r]);
        if (lane == 0) output[row] = bfloat(s);
    }
}

kernel void matvec_q4_0_simd(
    device const uchar* weight  [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid * ROWS_PER_TG + sgid;
    if (row >= out_features) return;

    uint blocks_per_row = in_features / 32;
    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint bo = (row * blocks_per_row + b) * 18;
        ushort scale_bits = ushort(weight[bo]) | (ushort(weight[bo + 1]) << 8);
        float scale = float(as_type<half>(scale_bits));

        uint byte_idx = lane / 2;
        uchar packed = weight[bo + 2 + byte_idx];
        int q = (lane % 2 == 0) ? (int(packed & 0x0F) - 8) : (int(packed >> 4) - 8);
        sum += scale * float(q) * float(input[b * 32 + lane]);
    }

    sum = simd_sum(sum);
    if (lane == 0) output[row] = bfloat(sum);
}

kernel void matvec_q8_0_simd(
    device const uchar* weight  [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid * ROWS_PER_TG + sgid;
    if (row >= out_features) return;

    uint blocks_per_row = in_features / 32;
    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint bo = (row * blocks_per_row + b) * 34;
        ushort scale_bits = ushort(weight[bo]) | (ushort(weight[bo + 1]) << 8);
        float scale = float(as_type<half>(scale_bits));
        float q = float(as_type<char>(weight[bo + 2 + lane]));
        sum += scale * q * float(input[b * 32 + lane]);
    }

    sum = simd_sum(sum);
    if (lane == 0) output[row] = bfloat(sum);
}

// Q4K: 2 rows per simdgroup, 4 simdgroups per threadgroup = 8 rows per tg.
// Input cached in registers, paired nibble processing.
constant constexpr uint Q4K_NR = 2;
constant constexpr uint Q4K_NSG = 4;

kernel void matvec_q4k_simd(
    device const uchar* weight  [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint row_base = tgid * (Q4K_NR * Q4K_NSG) + sgid * Q4K_NR;
    if (row_base >= out_features) return;

    uint blocks_per_row = in_features / 256;
    float sumf[Q4K_NR] = {};

    for (uint b = 0; b < blocks_per_row; b++) {
        // Cache input in registers: 8 values per thread (one per sub-block)
        float yl[8];
        uint input_base = b * 256;
        for (uint k = 0; k < 8; k++)
            yl[k] = float(input[input_base + k * 32 + lane]);

        for (uint r = 0; r < Q4K_NR; r++) {
            uint row = row_base + r;
            if (row >= out_features) break;

            uint bo = (row * blocks_per_row + b) * 144;
            ushort d_bits = ushort(weight[bo]) | (ushort(weight[bo + 1]) << 8);
            ushort dmin_bits = ushort(weight[bo + 2]) | (ushort(weight[bo + 3]) << 8);
            float d = float(as_type<half>(d_bits));
            float dmin = float(as_type<half>(dmin_bits));
            device const uchar* scales = weight + bo + 4;
            device const uchar* qs = weight + bo + 16;

            // Process 4 byte pairs: each byte holds 2 sub-blocks (lo/hi nibble)
            for (uint pair = 0; pair < 4; pair++) {
                uchar packed = qs[pair * 32 + lane];

                float2 sm_lo = unpack_q4k_scale(scales, pair * 2);
                float2 sm_hi = unpack_q4k_scale(scales, pair * 2 + 1);

                sumf[r] += (d * sm_lo.x * float(packed & 0x0F) - dmin * sm_lo.y) * yl[pair * 2];
                sumf[r] += (d * sm_hi.x * float(packed >> 4)   - dmin * sm_hi.y) * yl[pair * 2 + 1];
            }
        }
    }

    for (uint r = 0; r < Q4K_NR; r++) {
        uint row = row_base + r;
        if (row >= out_features) break;
        float s = simd_sum(sumf[r]);
        if (lane == 0) output[row] = bfloat(s);
    }
}

// Q6K: 2 rows per simdgroup, 4 simdgroups per threadgroup.
constant constexpr uint Q6K_NR = 2;
constant constexpr uint Q6K_NSG = 4;

kernel void matvec_q6k_simd(
    device const uchar* weight  [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint row_base = tgid * (Q6K_NR * Q6K_NSG) + sgid * Q6K_NR;
    if (row_base >= out_features) return;

    uint blocks_per_row = in_features / 256;
    float sumf[Q6K_NR] = {};

    for (uint b = 0; b < blocks_per_row; b++) {
        // Cache input in registers
        float yl[8];
        uint input_base = b * 256;
        for (uint k = 0; k < 8; k++)
            yl[k] = float(input[input_base + k * 32 + lane]);

        for (uint r = 0; r < Q6K_NR; r++) {
            uint row = row_base + r;
            if (row >= out_features) break;

            uint bo = (row * blocks_per_row + b) * 210;
            device const uchar* ql = weight + bo;
            device const uchar* qh = weight + bo + 128;
            device const uchar* scales = weight + bo + 192;
            ushort d_bits = ushort(weight[bo + 208]) | (ushort(weight[bo + 209]) << 8);
            float d = float(as_type<half>(d_bits));

            for (uint k = 0; k < 8; k++) {
                uint elem = k * 32 + lane;
                float sc = float(as_type<char>(scales[elem / 16]));
                uchar q = dequant_q6k_elem(ql, qh, elem);
                sumf[r] += d * sc * (float(q) - 32.0) * yl[k];
            }
        }
    }

    for (uint r = 0; r < Q6K_NR; r++) {
        uint row = row_base + r;
        if (row >= out_features) break;
        float s = simd_sum(sumf[r]);
        if (lane == 0) output[row] = bfloat(s);
    }
}

// ===================== Flash Attention 2 =====================
// One threadgroup per query head. Tiles over KV in blocks of FA_BLOCK positions.
// Online softmax: maintains running max (m_i) and sum-of-exp (l_i) across tiles.
// Shared memory: FA_BLOCK scores + scratch for reductions. O(1) in seq_len.
// Each thread owns one output dimension and accumulates the weighted value sum.
//
// Algorithm (per head):
//   o = 0, m = -inf, l = 0
//   for each KV tile [t_start..t_end]:
//     s[j] = dot(q, k[t_start+j]) * scale   (cooperative across threads)
//     m_new = max(m, max(s))
//     rescale = exp(m - m_new)
//     o = o * rescale + sum(exp(s[j] - m_new) * v[t_start+j])
//     l = l * rescale + sum(exp(s[j] - m_new))
//     m = m_new
//   output = o / l

constant constexpr uint FA_BLOCK = 32;

kernel void flash_attention(
    device const bfloat* q       [[buffer(0)]],
    device const bfloat* k_cache [[buffer(1)]],
    device const bfloat* v_cache [[buffer(2)]],
    device bfloat* output        [[buffer(3)]],
    constant uint& pos          [[buffer(4)]],
    constant uint& n_heads      [[buffer(5)]],
    constant uint& n_kv_heads   [[buffer(6)]],
    constant uint& head_dim_c   [[buffer(7)]],
    constant uint& kv_dim       [[buffer(8)]],
    uint head    [[threadgroup_position_in_grid]],
    uint d       [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg      [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared   [[threadgroup(0)]])
{
    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint q_off  = head * head_dim_c;
    uint kv_off = kv_head * head_dim_c;
    float scale = rsqrt(float(head_dim_c));
    uint seq_len = pos + 1;
    uint n_sg = (tg_size + 31) / 32;

    // Shared memory: [FA_BLOCK scores] [n_sg scratch]
    threadgroup float* scores  = shared;
    threadgroup float* scratch = shared + FA_BLOCK;

    // Per-thread accumulators (each thread owns dimension d < head_dim)
    float o_acc = 0.0;
    float m_i = -INFINITY;
    float l_i = 0.0;

    for (uint t_start = 0; t_start < seq_len; t_start += FA_BLOCK) {
        uint t_end = min(t_start + FA_BLOCK, seq_len);
        uint tile_len = t_end - t_start;

        // Phase 1: cooperatively compute dot(q, k) for this tile
        for (uint j = d; j < tile_len; j += tg_size) {
            float dot = 0.0;
            for (uint i = 0; i < head_dim_c; i++)
                dot += float(q[q_off + i]) * float(k_cache[(t_start + j) * kv_dim + kv_off + i]);
            scores[j] = dot * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Phase 2: find tile max
        float tile_max = -INFINITY;
        for (uint j = d; j < tile_len; j += tg_size)
            tile_max = max(tile_max, scores[j]);
        tile_max = simd_max(tile_max);
        if (lane == 0) scratch[sg] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float v = (lane < n_sg) ? scratch[lane] : -INFINITY;
            v = simd_max(v);
            if (lane == 0) scratch[0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max = scratch[0];

        // Phase 3: online softmax rescale
        float m_new = max(m_i, tile_max);
        float rescale = exp(m_i - m_new);
        o_acc *= rescale;
        l_i *= rescale;

        // Compute exp(score - m_new) and tile sum
        float tile_sum = 0.0;
        for (uint j = d; j < tile_len; j += tg_size) {
            float e = exp(scores[j] - m_new);
            scores[j] = e;
            tile_sum += e;
        }
        tile_sum = simd_sum(tile_sum);
        if (lane == 0) scratch[sg] = tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float v = (lane < n_sg) ? scratch[lane] : 0.0;
            v = simd_sum(v);
            if (lane == 0) scratch[0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        l_i += scratch[0];
        m_i = m_new;

        // Phase 4: accumulate weighted values
        if (d < head_dim_c) {
            for (uint j = 0; j < tile_len; j++)
                o_acc += scores[j] * float(v_cache[(t_start + j) * kv_dim + kv_off + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (d < head_dim_c)
        output[q_off + d] = bfloat(o_acc / l_i);
}

// ===================== SiLU =====================

kernel void silu_inplace(
    device bfloat* x   [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < len) {
        float v = float(x[tid]);
        x[tid] = bfloat(v / (1.0 + exp(-v)));
    }
}

// ===================== Elementwise =====================

kernel void add_vecs(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out     [[buffer(2)]],
    constant uint& len     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < len) out[tid] = bfloat(float(a[tid]) + float(b[tid]));
}

kernel void mul_vecs(
    device const bfloat* a [[buffer(0)]],
    device const bfloat* b [[buffer(1)]],
    device bfloat* out     [[buffer(2)]],
    constant uint& len     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < len) out[tid] = bfloat(float(a[tid]) * float(b[tid]));
}

// ===================== Argmax =====================
//
// Reduce a logit vector to the index of its maximum value. Single threadgroup,
// 256 threads, tree reduction in static threadgroup memory. Result is one u32.
//
// Replaces a 128k×4 byte CPU readback with a 4-byte readback for greedy
// sampling. The full forward+sample stays GPU-resident.

kernel void argmax(
    device const bfloat* logits [[buffer(0)]],
    device uint* out            [[buffer(1)]],
    constant uint& n            [[buffer(2)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup float shared_val[256];
    threadgroup uint  shared_idx[256];

    float best_v = -INFINITY;
    uint  best_i = 0;
    for (uint i = tid; i < n; i += tg_size) {
        float v = float(logits[i]);
        if (v > best_v) { best_v = v; best_i = i; }
    }
    shared_val[tid] = best_v;
    shared_idx[tid] = best_i;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other_v = shared_val[tid + stride];
            if (other_v > shared_val[tid]) {
                shared_val[tid] = other_v;
                shared_idx[tid] = shared_idx[tid + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) out[0] = shared_idx[0];
}

// ===================== Copy with offset =====================

kernel void copy_offset(
    device const bfloat* src [[buffer(0)]],
    device bfloat* dst       [[buffer(1)]],
    constant uint& dst_off   [[buffer(2)]],
    constant uint& count     [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count) dst[dst_off + tid] = src[tid];
}

// ===================== Batched Embedding =====================
// 2D grid: (dim, seq_len). Each thread embeds one element of one token.

kernel void embed_batch_f32(
    device const float* table   [[buffer(0)]],
    device const uint* tok_ids  [[buffer(1)]],
    device bfloat* out          [[buffer(2)]],
    constant uint& dim          [[buffer(3)]],
    constant uint& seq_len      [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint d = pos.x, s = pos.y;
    if (d >= dim || s >= seq_len) return;
    out[s * dim + d] = bfloat(table[tok_ids[s] * dim + d]);
}

kernel void embed_batch_bf16(
    device const ushort* table  [[buffer(0)]],
    device const uint* tok_ids  [[buffer(1)]],
    device bfloat* out          [[buffer(2)]],
    constant uint& dim          [[buffer(3)]],
    constant uint& seq_len      [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint d = pos.x, s = pos.y;
    if (d >= dim || s >= seq_len) return;
    out[s * dim + d] = bfloat(bf16_to_f32(table[tok_ids[s] * dim + d]));
}

kernel void embed_batch_f16(
    device const half* table    [[buffer(0)]],
    device const uint* tok_ids  [[buffer(1)]],
    device bfloat* out          [[buffer(2)]],
    constant uint& dim          [[buffer(3)]],
    constant uint& seq_len      [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint d = pos.x, s = pos.y;
    if (d >= dim || s >= seq_len) return;
    out[s * dim + d] = bfloat(float(table[tok_ids[s] * dim + d]));
}

kernel void embed_batch_q4k(
    device const uchar* table   [[buffer(0)]],
    device const uint* tok_ids  [[buffer(1)]],
    device bfloat* out          [[buffer(2)]],
    constant uint& dim          [[buffer(3)]],
    constant uint& seq_len      [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint tid = pos.x, s = pos.y;
    if (tid >= dim || s >= seq_len) return;
    uint token_id = tok_ids[s];
    uint block_idx = tid / 256;
    uint elem_in_block = tid % 256;
    uint sb = elem_in_block / 32;
    uint blocks_per_row = dim / 256;
    uint bo = (token_id * blocks_per_row + block_idx) * 144;
    ushort d_bits = ushort(table[bo]) | (ushort(table[bo + 1]) << 8);
    ushort dmin_bits = ushort(table[bo + 2]) | (ushort(table[bo + 3]) << 8);
    float d = float(as_type<half>(d_bits));
    float dmin = float(as_type<half>(dmin_bits));
    device const uchar* scales = table + bo + 4;
    device const uchar* qs = table + bo + 16;
    float2 sm = unpack_q4k_scale(scales, sb);
    uint group = elem_in_block / 64;
    uint gj = elem_in_block % 64;
    float q;
    if (gj < 32) q = float(qs[group * 32 + gj] & 0x0F);
    else         q = float(qs[group * 32 + gj - 32] >> 4);
    out[s * dim + tid] = bfloat(d * sm.x * q - dmin * sm.y);
}

kernel void embed_batch_q6k(
    device const uchar* table   [[buffer(0)]],
    device const uint* tok_ids  [[buffer(1)]],
    device bfloat* out          [[buffer(2)]],
    constant uint& dim          [[buffer(3)]],
    constant uint& seq_len      [[buffer(4)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint tid = pos.x, s = pos.y;
    if (tid >= dim || s >= seq_len) return;
    uint token_id = tok_ids[s];
    uint block_idx = tid / 256;
    uint elem_in_block = tid % 256;
    uint blocks_per_row = dim / 256;
    uint bo = (token_id * blocks_per_row + block_idx) * 210;
    device const uchar* ql = table + bo;
    device const uchar* qh = table + bo + 128;
    device const uchar* scales = table + bo + 192;
    ushort d_bits = ushort(table[bo + 208]) | (ushort(table[bo + 209]) << 8);
    float d = float(as_type<half>(d_bits));
    float sc = float(as_type<char>(scales[elem_in_block / 16]));
    uchar q = dequant_q6k_elem(ql, qh, elem_in_block);
    out[s * dim + tid] = bfloat(d * sc * (float(q) - 32.0));
}

// ===================== Batched RMS Norm =====================
// One threadgroup per row. seq_len rows of dim elements.

kernel void rms_norm_batch(
    device const bfloat* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device bfloat* output      [[buffer(2)]],
    constant uint& dim         [[buffer(3)]],
    constant float& eps        [[buffer(4)]],
    uint row     [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared  [[threadgroup(0)]])
{
    uint off = row * dim;
    float sum_sq = 0.0;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = float(input[off + i]);
        sum_sq += v * v;
    }

    sum_sq = simd_sum(sum_sq);
    uint lane = tid % 32;
    uint sg   = tid / 32;
    uint n_sg = (tg_size + 31) / 32;
    if (lane == 0) shared[sg] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (lane < n_sg) ? shared[lane] : 0.0;
        v = simd_sum(v);
        if (lane == 0) shared[0] = rsqrt(v / float(dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv_rms = shared[0];
    for (uint i = tid; i < dim; i += tg_size)
        output[off + i] = bfloat(float(input[off + i]) * inv_rms * weight[i]);
}

// ===================== Batched RoPE =====================
// 2D grid: (n_pairs_per_row, seq_len). Each thread handles one rotation pair.

// LLaMA-style rope: rotate interleaved pairs (x[2i], x[2i+1]) within each head.
// Qwen3/HF-style (NeoX) rope: rotate split pairs (x[i], x[i+head_dim/2]).
// `neox == 0` picks interleaved, `neox == 1` picks split.
kernel void rope_batch(
    device bfloat* x           [[buffer(0)]],
    constant uint& start_pos   [[buffer(1)]],
    constant uint& head_dim    [[buffer(2)]],
    constant float& theta      [[buffer(3)]],
    constant uint& n_pairs     [[buffer(4)]],
    constant uint& row_stride  [[buffer(5)]],
    constant uint& neox        [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]])
{
    uint pair = pos.x, s = pos.y;
    if (pair >= n_pairs) return;
    uint p = start_pos + s;

    uint half_hd = head_dim / 2;
    uint head = pair / half_hd;
    uint i    = pair % half_hd;
    uint head_base = s * row_stride + head * head_dim;
    uint a, b;
    if (neox != 0) { a = head_base + i;         b = head_base + i + half_hd; }
    else           { a = head_base + 2 * i;     b = head_base + 2 * i + 1;   }

    float freq  = 1.0 / pow(theta, 2.0 * float(i) / float(head_dim));
    float angle = float(p) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float x0 = float(x[a]);
    float x1 = float(x[b]);
    x[a] = bfloat(x0 * cos_a - x1 * sin_a);
    x[b] = bfloat(x0 * sin_a + x1 * cos_a);
}

// ===================== GEMM =====================
// out[s, r] = sum_k weight[r, k] * input[s, k]
// 2D grid: (ceil(out_features / GEMM_NSG), ceil(seq_len / GEMM_TILE_S)).
// Each simdgroup handles 1 row × TILE_S seq positions.
// Weight loaded once per tile, reused across TILE_S positions.

constant constexpr uint GEMM_NSG = 4;
constant constexpr uint GEMM_TILE_S = 4;

kernel void gemm_f32(
    device const float* weight  [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    constant uint& seq_len      [[buffer(5)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid.x * GEMM_NSG + sgid;
    uint s_start = tgid.y * GEMM_TILE_S;
    if (row >= out_features || s_start >= seq_len) return;
    uint tile = min(GEMM_TILE_S, seq_len - s_start);

    float sumf[GEMM_TILE_S] = {};
    for (uint i = lane; i < in_features; i += 32) {
        float w = weight[row * in_features + i];
        for (uint ds = 0; ds < tile; ds++)
            sumf[ds] += w * float(input[(s_start + ds) * in_features + i]);
    }
    for (uint ds = 0; ds < tile; ds++) {
        float t = simd_sum(sumf[ds]);
        if (lane == 0) output[(s_start + ds) * out_features + row] = bfloat(t);
    }
}

// MMA-based GEMM for BF16 using simdgroup_bfloat8x8.
// Each simdgroup: NR*8 out rows x 8 seq positions (NR row tiles, reusing input).
// 2D grid: (ceil(out_features / (NSG*NR*8)), ceil(seq_len / 8)).
// Input is now bfloat -- load directly from device memory with transpose, no threadgroup staging needed.
constant constexpr uint MMA_NSG = 4;  // simdgroups per TG
constant constexpr uint MMA_NR = 2;   // row tiles per simdgroup
constant constexpr uint MMA_TM = MMA_NSG * MMA_NR * 8;  // 64 rows per TG
constant constexpr uint MMA_TN = 8;   // seq positions per TG

kernel void gemm_bf16(
    device const bfloat* weight  [[buffer(0)]],
    device const bfloat* input   [[buffer(1)]],
    device bfloat* output        [[buffer(2)]],
    constant uint& in_features   [[buffer(3)]],
    constant uint& out_features  [[buffer(4)]],
    constant uint& seq_len       [[buffer(5)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint m_base = tgid.x * MMA_TM + sgid * MMA_NR * 8;
    uint n_start = tgid.y * MMA_TN;
    if (m_base >= out_features || n_start >= seq_len) return;

    // NR accumulators per simdgroup (each 8x8)
    simdgroup_bfloat8x8 acc[MMA_NR];
    for (uint r = 0; r < MMA_NR; r++)
        acc[r] = make_filled_simdgroup_matrix<bfloat, 8, 8>(0.0bf);

    uint n_threads = MMA_NSG * 32;

    for (uint k = 0; k < in_features; k += 8) {
        // Load input directly from device memory with transpose.
        // Input layout: input[s * in_features + k] (s = row, k = col)
        // We want B[ki][sj] = input[(n_start + sj) * in_features + k + ki]
        // With transpose flag: B[ki][sj] = src[sj * stride + ki]
        // base = input + n_start * in_features + k, stride = in_features
        simdgroup_bfloat8x8 B;
        simdgroup_load(B, input + n_start * in_features + k, in_features, ulong2(0), true);

        for (uint r = 0; r < MMA_NR; r++) {
            simdgroup_bfloat8x8 A;
            simdgroup_load(A, weight + (m_base + r * 8) * in_features + k, in_features);
            simdgroup_multiply_accumulate(acc[r], A, B, acc[r]);
        }
    }

    // Store results via threadgroup staging for bounds checking
    threadgroup bfloat tg_acc[MMA_TM * MMA_TN];
    for (uint r = 0; r < MMA_NR; r++)
        simdgroup_store(acc[r], tg_acc + (sgid * MMA_NR + r) * 8 * MMA_TN, MMA_TN);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid_in_tg; idx < MMA_TM * MMA_TN; idx += n_threads) {
        uint r = idx / MMA_TN;
        uint s = idx % MMA_TN;
        uint gr = tgid.x * MMA_TM + r;
        uint gs = n_start + s;
        if (gr < out_features && gs < seq_len)
            output[gs * out_features + gr] = tg_acc[idx];
    }
}

kernel void gemm_f16(
    device const half* weight   [[buffer(0)]],
    device const bfloat* input  [[buffer(1)]],
    device bfloat* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    constant uint& seq_len      [[buffer(5)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid.x * GEMM_NSG + sgid;
    uint s_start = tgid.y * GEMM_TILE_S;
    if (row >= out_features || s_start >= seq_len) return;
    uint tile = min(GEMM_TILE_S, seq_len - s_start);

    float sumf[GEMM_TILE_S] = {};
    for (uint i = lane * 4; i < in_features; i += 128) {
        device const half* wp = weight + row * in_features + i;
        float w0 = float(wp[0]), w1 = float(wp[1]);
        float w2 = float(wp[2]), w3 = float(wp[3]);
        for (uint ds = 0; ds < tile; ds++) {
            uint base = (s_start + ds) * in_features + i;
            float inp0 = float(input[base]);
            float inp1 = float(input[base + 1]);
            float inp2 = float(input[base + 2]);
            float inp3 = float(input[base + 3]);
            sumf[ds] += w0 * inp0 + w1 * inp1 + w2 * inp2 + w3 * inp3;
        }
    }
    for (uint ds = 0; ds < tile; ds++) {
        float t = simd_sum(sumf[ds]);
        if (lane == 0) output[(s_start + ds) * out_features + row] = bfloat(t);
    }
}

// ===================== Causal Attention =====================
// 2D grid: (n_heads, batch_seq_len). One threadgroup per (head, query_position).
// Each query at batch index q_idx attends to KV at positions [0..start_pos+q_idx].

kernel void causal_attention(
    device const bfloat* q       [[buffer(0)]],
    device const bfloat* k_cache [[buffer(1)]],
    device const bfloat* v_cache [[buffer(2)]],
    device bfloat* output        [[buffer(3)]],
    constant uint& start_pos    [[buffer(4)]],
    constant uint& batch_seq_len [[buffer(5)]],
    constant uint& n_heads      [[buffer(6)]],
    constant uint& n_kv_heads   [[buffer(7)]],
    constant uint& head_dim_c   [[buffer(8)]],
    constant uint& kv_dim       [[buffer(9)]],
    constant uint& n_heads_grid [[buffer(10)]],
    uint tgid_flat [[threadgroup_position_in_grid]],
    uint d       [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]],
    uint sg      [[simdgroup_index_in_threadgroup]],
    threadgroup float* shared   [[threadgroup(0)]])
{
    uint head  = tgid_flat % n_heads_grid;
    uint q_idx = tgid_flat / n_heads_grid;
    uint abs_pos = start_pos + q_idx;
    uint seq_len = abs_pos + 1;

    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint dim_stride = n_heads * head_dim_c;
    uint q_off  = q_idx * dim_stride + head * head_dim_c;
    uint kv_off = kv_head * head_dim_c;
    float scale = rsqrt(float(head_dim_c));
    uint n_sg = (tg_size + 31) / 32;

    threadgroup float* scores  = shared;
    threadgroup float* scratch = shared + FA_BLOCK;

    float o_acc = 0.0;
    float m_i = -INFINITY;
    float l_i = 0.0;

    for (uint t_start = 0; t_start < seq_len; t_start += FA_BLOCK) {
        uint t_end = min(t_start + FA_BLOCK, seq_len);
        uint tile_len = t_end - t_start;

        for (uint j = d; j < tile_len; j += tg_size) {
            float dot = 0.0;
            for (uint i = 0; i < head_dim_c; i++)
                dot += float(q[q_off + i]) * float(k_cache[(t_start + j) * kv_dim + kv_off + i]);
            scores[j] = dot * scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float tile_max = -INFINITY;
        for (uint j = d; j < tile_len; j += tg_size)
            tile_max = max(tile_max, scores[j]);
        tile_max = simd_max(tile_max);
        if (lane == 0) scratch[sg] = tile_max;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float v = (lane < n_sg) ? scratch[lane] : -INFINITY;
            v = simd_max(v);
            if (lane == 0) scratch[0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_max = scratch[0];

        float m_new = max(m_i, tile_max);
        float rescale = exp(m_i - m_new);
        o_acc *= rescale;
        l_i *= rescale;

        float tile_sum = 0.0;
        for (uint j = d; j < tile_len; j += tg_size) {
            float e = exp(scores[j] - m_new);
            scores[j] = e;
            tile_sum += e;
        }
        tile_sum = simd_sum(tile_sum);
        if (lane == 0) scratch[sg] = tile_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            float v = (lane < n_sg) ? scratch[lane] : 0.0;
            v = simd_sum(v);
            if (lane == 0) scratch[0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        l_i += scratch[0];
        m_i = m_new;

        if (d < head_dim_c) {
            for (uint j = 0; j < tile_len; j++)
                o_acc += scores[j] * float(v_cache[(t_start + j) * kv_dim + kv_off + d]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (d < head_dim_c)
        output[q_off + d] = bfloat(o_acc / l_i);
}
