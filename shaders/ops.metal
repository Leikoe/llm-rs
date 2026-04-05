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

// ===================== Embedding =====================

kernel void embed_f32(
    device const float* table [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant uint& token_id   [[buffer(2)]],
    constant uint& dim        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < dim) out[tid] = table[token_id * dim + tid];
}

kernel void embed_bf16(
    device const ushort* table [[buffer(0)]],
    device float* out          [[buffer(1)]],
    constant uint& token_id    [[buffer(2)]],
    constant uint& dim         [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < dim) out[tid] = bf16_to_f32(table[token_id * dim + tid]);
}

kernel void embed_f16(
    device const half* table [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant uint& token_id  [[buffer(2)]],
    constant uint& dim       [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < dim) out[tid] = float(table[token_id * dim + tid]);
}

kernel void embed_q4k(
    device const uchar* table [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant uint& token_id   [[buffer(2)]],
    constant uint& dim        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= dim) return;
    // Q4K: 256 elements per 144-byte block
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

    out[tid] = d * sm.x * q - dmin * sm.y;
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

kernel void embed_q6k(
    device const uchar* table [[buffer(0)]],
    device float* out         [[buffer(1)]],
    constant uint& token_id   [[buffer(2)]],
    constant uint& dim        [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= dim) return;
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
    out[tid] = d * sc * (float(q) - 32.0);
}

// ===================== Matrix-Vector Multiply =====================
// out[row] = dot(weight[row,:], input[:])
// GGML layout: weight is [in_features, out_features] = out_features rows of in_features.
// One thread per output row.

kernel void matvec_f32(
    device const float* weight [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= out_features) return;
    float sum = 0.0;
    uint base = row * in_features;
    for (uint i = 0; i < in_features; i++) {
        sum += weight[base + i] * input[i];
    }
    output[row] = sum;
}

kernel void matvec_bf16(
    device const ushort* weight [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= out_features) return;
    float sum = 0.0;
    uint base = row * in_features;
    for (uint i = 0; i < in_features; i++) {
        sum += bf16_to_f32(weight[base + i]) * input[i];
    }
    output[row] = sum;
}

kernel void matvec_f16(
    device const half* weight  [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= out_features) return;
    float sum = 0.0;
    uint base = row * in_features;
    for (uint i = 0; i < in_features; i++) {
        sum += float(weight[base + i]) * input[i];
    }
    output[row] = sum;
}

kernel void matvec_q4_0(
    device const uchar* weight [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= out_features) return;

    uint blocks_per_row = in_features / 32;
    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint bo = (row * blocks_per_row + b) * 18; // Q4_0 block = 18 bytes
        ushort scale_bits = ushort(weight[bo]) | (ushort(weight[bo + 1]) << 8);
        float scale = float(as_type<half>(scale_bits));

        for (uint j = 0; j < 16; j++) {
            uchar packed = weight[bo + 2 + j];
            int lo = int(packed & 0x0F) - 8;
            int hi = int(packed >> 4) - 8;
            uint idx = b * 32 + j * 2;
            sum += scale * (float(lo) * input[idx] + float(hi) * input[idx + 1]);
        }
    }

    output[row] = sum;
}

kernel void matvec_q8_0(
    device const uchar* weight [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= out_features) return;

    uint blocks_per_row = in_features / 32;
    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint bo = (row * blocks_per_row + b) * 34; // Q8_0 block = 34 bytes
        ushort scale_bits = ushort(weight[bo]) | (ushort(weight[bo + 1]) << 8);
        float scale = float(as_type<half>(scale_bits));

        for (uint j = 0; j < 32; j++) {
            float q = float(as_type<char>(weight[bo + 2 + j]));
            sum += scale * q * input[b * 32 + j];
        }
    }

    output[row] = sum;
}

kernel void matvec_q4k(
    device const uchar* weight [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= out_features) return;

    uint blocks_per_row = in_features / 256;
    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint bo = (row * blocks_per_row + b) * 144;
        ushort d_bits = ushort(weight[bo]) | (ushort(weight[bo + 1]) << 8);
        ushort dmin_bits = ushort(weight[bo + 2]) | (ushort(weight[bo + 3]) << 8);
        float d = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));
        device const uchar* scales = weight + bo + 4;
        device const uchar* qs = weight + bo + 16;

        for (uint i = 0; i < 256; i++) {
            uint sb = i / 32;
            float2 sm = unpack_q4k_scale(scales, sb);
            uint group = i / 64;
            uint gj = i % 64;
            float q;
            if (gj < 32) q = float(qs[group * 32 + gj] & 0x0F);
            else         q = float(qs[group * 32 + gj - 32] >> 4);
            sum += (d * sm.x * q - dmin * sm.y) * input[b * 256 + i];
        }
    }

    output[row] = sum;
}

kernel void matvec_q6k(
    device const uchar* weight [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint row [[thread_position_in_grid]])
{
    if (row >= out_features) return;

    uint blocks_per_row = in_features / 256;
    float sum = 0.0;

    for (uint b = 0; b < blocks_per_row; b++) {
        uint bo = (row * blocks_per_row + b) * 210;
        device const uchar* ql = weight + bo;
        device const uchar* qh = weight + bo + 128;
        device const uchar* scales = weight + bo + 192;
        ushort d_bits = ushort(weight[bo + 208]) | (ushort(weight[bo + 209]) << 8);
        float d = float(as_type<half>(d_bits));

        for (uint sb = 0; sb < 16; sb++) {
            float sc = float(as_type<char>(scales[sb]));
            float d_sc = d * sc;
            for (uint j = 0; j < 16; j++) {
                uint elem = sb * 16 + j;
                uchar q = dequant_q6k_elem(ql, qh, elem);
                sum += d_sc * (float(q) - 32.0) * input[b * 256 + elem];
            }
        }
    }

    output[row] = sum;
}

// ===================== SIMD Matvec =====================
// One SIMD group (32 threads) per row, 8 rows per threadgroup.
// Each thread handles a stride of the dot product, reduced with simd_sum.

constant constexpr uint ROWS_PER_TG = 8;

kernel void matvec_f32_simd(
    device const float* weight [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
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
        sum += weight[base + i] * input[i];
    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum;
}

kernel void matvec_bf16_simd(
    device const ushort* weight [[buffer(0)]],
    device const float* input   [[buffer(1)]],
    device float* output        [[buffer(2)]],
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
        sum += bf16_to_f32(weight[base + i]) * input[i];
    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum;
}

kernel void matvec_f16_simd(
    device const half* weight  [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
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
        sum += float(weight[base + i]) * input[i];
    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum;
}

kernel void matvec_q4_0_simd(
    device const uchar* weight [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
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
        sum += scale * float(q) * input[b * 32 + lane];
    }

    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum;
}

kernel void matvec_q8_0_simd(
    device const uchar* weight [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
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
        sum += scale * q * input[b * 32 + lane];
    }

    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum;
}

kernel void matvec_q4k_simd(
    device const uchar* weight [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid * ROWS_PER_TG + sgid;
    if (row >= out_features) return;

    uint blocks_per_row = in_features / 256;
    float sum = 0.0;

    // Each SIMD group (32 lanes) processes one row.
    // For each Q4K block (256 elements = 8 sub-blocks of 32):
    // each lane handles one element per sub-block, iterating over all 8 sub-blocks.
    for (uint b = 0; b < blocks_per_row; b++) {
        uint bo = (row * blocks_per_row + b) * 144;
        ushort d_bits = ushort(weight[bo]) | (ushort(weight[bo + 1]) << 8);
        ushort dmin_bits = ushort(weight[bo + 2]) | (ushort(weight[bo + 3]) << 8);
        float d = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));
        device const uchar* scales = weight + bo + 4;
        device const uchar* qs = weight + bo + 16;

        for (uint sb = 0; sb < 8; sb++) {
            float2 sm = unpack_q4k_scale(scales, sb);
            float sc_val = d * sm.x;
            float m_val = dmin * sm.y;

            uint group = sb / 2;
            float q;
            if (sb % 2 == 0) q = float(qs[group * 32 + lane] & 0x0F);
            else             q = float(qs[group * 32 + lane] >> 4);

            uint idx = b * 256 + sb * 32 + lane;
            sum += (sc_val * q - m_val) * input[idx];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum;
}

kernel void matvec_q6k_simd(
    device const uchar* weight [[buffer(0)]],
    device const float* input  [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& in_features  [[buffer(3)]],
    constant uint& out_features [[buffer(4)]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    uint row = tgid * ROWS_PER_TG + sgid;
    if (row >= out_features) return;

    uint blocks_per_row = in_features / 256;
    float sum = 0.0;

    // 32 lanes process 256 elements: each lane handles 8 elements per block
    // (one element from each of the 8 pairs of sub-blocks)
    for (uint b = 0; b < blocks_per_row; b++) {
        uint bo = (row * blocks_per_row + b) * 210;
        device const uchar* ql = weight + bo;
        device const uchar* qh = weight + bo + 128;
        device const uchar* scales = weight + bo + 192;
        ushort d_bits = ushort(weight[bo + 208]) | (ushort(weight[bo + 209]) << 8);
        float d = float(as_type<half>(d_bits));

        // Each lane processes elements lane, lane+32, lane+64, ..., lane+224
        for (uint k = 0; k < 8; k++) {
            uint elem = k * 32 + lane;
            float sc = float(as_type<char>(scales[elem / 16]));
            uchar q = dequant_q6k_elem(ql, qh, elem);
            sum += d * sc * (float(q) - 32.0) * input[b * 256 + elem];
        }
    }

    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum;
}

// ===================== GQA Attention =====================
// One threadgroup per query head. Threadgroup size = head_dim.
// Each thread handles one output dimension + a stripe of positions.
// Shared memory: scores[seq_len] + scratch[n_simdgroups].

kernel void gqa_attention(
    device const float* q       [[buffer(0)]],
    device const float* k_cache [[buffer(1)]],
    device const float* v_cache [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant uint& pos          [[buffer(4)]],
    constant uint& n_heads      [[buffer(5)]],
    constant uint& n_kv_heads   [[buffer(6)]],
    constant uint& head_dim_c   [[buffer(7)]],
    constant uint& kv_dim       [[buffer(8)]],
    uint head    [[threadgroup_position_in_grid]],
    uint d       [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared   [[threadgroup(0)]])
{
    uint heads_per_kv = n_heads / n_kv_heads;
    uint kv_head = head / heads_per_kv;
    uint q_off  = head * head_dim_c;
    uint kv_off = kv_head * head_dim_c;
    float scale = rsqrt(float(head_dim_c));
    uint seq_len = pos + 1;

    // Pointers into shared memory
    threadgroup float* scores  = shared;
    threadgroup float* scratch = shared + seq_len;

    uint lane = d % 32;
    uint sg   = d / 32;
    uint n_sg = (tg_size + 31) / 32;

    // Phase 1: compute attention scores
    for (uint t = d; t < seq_len; t += tg_size) {
        float dot = 0.0;
        for (uint i = 0; i < head_dim_c; i++)
            dot += q[q_off + i] * k_cache[t * kv_dim + kv_off + i];
        scores[t] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: softmax -- find max
    float m = -INFINITY;
    for (uint t = d; t < seq_len; t += tg_size) m = max(m, scores[t]);
    m = simd_max(m);
    if (lane == 0) scratch[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (lane < n_sg) ? scratch[lane] : -INFINITY;
        v = simd_max(v);
        if (lane == 0) scratch[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = scratch[0];

    // exp + sum
    float s = 0.0;
    for (uint t = d; t < seq_len; t += tg_size) {
        float v = exp(scores[t] - max_val);
        scores[t] = v;
        s += v;
    }
    s = simd_sum(s);
    if (lane == 0) scratch[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (lane < n_sg) ? scratch[lane] : 0.0;
        v = simd_sum(v);
        if (lane == 0) scratch[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = scratch[0];

    // normalize
    for (uint t = d; t < seq_len; t += tg_size) scores[t] /= total;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: weighted value sum -- one thread per dimension
    if (d < head_dim_c) {
        float acc = 0.0;
        for (uint t = 0; t < seq_len; t++)
            acc += scores[t] * v_cache[t * kv_dim + kv_off + d];
        output[q_off + d] = acc;
    }
}

// ===================== RMS Norm =====================
// Single threadgroup: parallel reduction for sum-of-squares, then normalize.

kernel void rms_norm(
    device const float* input  [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output       [[buffer(2)]],
    constant uint& dim         [[buffer(3)]],
    constant float& eps        [[buffer(4)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared  [[threadgroup(0)]])
{
    float sum_sq = 0.0;
    for (uint i = tid; i < dim; i += tg_size) {
        float v = input[i];
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
    for (uint i = tid; i < dim; i += tg_size) {
        output[i] = input[i] * inv_rms * weight[i];
    }
}

// ===================== RoPE =====================
// One thread per (x[2i], x[2i+1]) pair.

kernel void rope(
    device float* x        [[buffer(0)]],
    constant uint& pos     [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant float& theta  [[buffer(3)]],
    constant uint& n_pairs [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= n_pairs) return;

    uint half_hd = head_dim / 2;
    uint head = tid / half_hd;
    uint i    = tid % half_hd;
    uint base = head * head_dim + 2 * i;

    float freq  = 1.0 / pow(theta, 2.0 * float(i) / float(head_dim));
    float angle = float(pos) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float x0 = x[base];
    float x1 = x[base + 1];
    x[base]     = x0 * cos_a - x1 * sin_a;
    x[base + 1] = x0 * sin_a + x1 * cos_a;
}

// ===================== Softmax =====================
// Single threadgroup: find max, exp, sum, normalize.

kernel void softmax_kernel(
    device float* x        [[buffer(0)]],
    constant uint& len     [[buffer(1)]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    threadgroup float* shared [[threadgroup(0)]])
{
    uint lane = tid % 32;
    uint sg   = tid / 32;
    uint n_sg = (tg_size + 31) / 32;

    // Phase 1: max
    float m = -INFINITY;
    for (uint i = tid; i < len; i += tg_size) m = max(m, x[i]);
    m = simd_max(m);
    if (lane == 0) shared[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (lane < n_sg) ? shared[lane] : -INFINITY;
        v = simd_max(v);
        if (lane == 0) shared[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float max_val = shared[0];

    // Phase 2: exp + sum
    float s = 0.0;
    for (uint i = tid; i < len; i += tg_size) {
        float v = exp(x[i] - max_val);
        x[i] = v;
        s += v;
    }
    s = simd_sum(s);
    if (lane == 0) shared[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = (lane < n_sg) ? shared[lane] : 0.0;
        v = simd_sum(v);
        if (lane == 0) shared[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = shared[0];

    // Phase 3: normalize
    for (uint i = tid; i < len; i += tg_size) x[i] /= sum;
}

// ===================== SiLU =====================

kernel void silu_inplace(
    device float* x    [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < len) {
        float v = x[tid];
        x[tid] = v / (1.0 + exp(-v));
    }
}

// ===================== Elementwise =====================

kernel void add_vecs(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant uint& len    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < len) out[tid] = a[tid] + b[tid];
}

kernel void mul_vecs(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out     [[buffer(2)]],
    constant uint& len    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < len) out[tid] = a[tid] * b[tid];
}

// ===================== Copy with offset =====================

kernel void copy_offset(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant uint& dst_off  [[buffer(2)]],
    constant uint& count    [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < count) dst[dst_off + tid] = src[tid];
}
