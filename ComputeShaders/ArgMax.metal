//
//  ArgMax.metal
//  AdityaIntelligenceProMax
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

// Every kernel starts from (lowest, 0) and takes an element only when it is strictly larger, or
// equal with a smaller index. That is the CPU loop's answer exactly: the first index of the max,
// NaNs never win, and 0 when nothing beats lowest.
template <typename T> inline T argmax_lowest() { return numeric_limits<T>::lowest(); }
template <> inline bool argmax_lowest<bool>() { return false; }

// simd_shuffle_down has no bool, bfloat or 64-bit overloads: those move their bits through one that does.
template <typename T> inline T argmax_shuffle_down(T v, ushort delta) { return simd_shuffle_down(v, delta); }
template <> inline bool argmax_shuffle_down<bool>(bool v, ushort delta) { return bool(simd_shuffle_down(ushort(v), delta)); }
template <> inline bfloat argmax_shuffle_down<bfloat>(bfloat v, ushort delta) { return as_type<bfloat>(simd_shuffle_down(as_type<ushort>(v), delta)); }
template <> inline long argmax_shuffle_down<long>(long v, ushort delta) { return as_type<long>(simd_shuffle_down(as_type<uint2>(v), delta)); }
template <> inline ulong argmax_shuffle_down<ulong>(ulong v, ushort delta) { return as_type<ulong>(simd_shuffle_down(as_type<uint2>(v), delta)); }

template <typename T>
inline void argmax_take(thread T& best, thread uint& best_idx, T v, uint idx) {
    if (v > best || (v == best && idx < best_idx)) { best = v; best_idx = idx; }
}

// One thread walks the whole reduced axis.
template <typename T>
inline int argmax_row(device const T* src, size_m src_idx, size_m reduce_axis_stride, size_m noOfOpp) {
    T best = argmax_lowest<T>();
    uint best_idx = 0;
    for (size_m i = 0; i < noOfOpp; i++) {
        T val = src[src_idx + i * reduce_axis_stride];
        if (val > best) { best = val; best_idx = i; }
    }
    return int(best_idx);
}

// (value, index) pairs folded down a 32-lane simdgroup: lane 0 ends with the simdgroup's answer.
template <typename T>
inline void argmax_simd(thread T& best, thread uint& best_idx) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        T v = argmax_shuffle_down(best, offset);
        uint idx = simd_shuffle_down(best_idx, offset);
        argmax_take(best, best_idx, v, idx);
    }
}

// The whole threadgroup reduces one output: each thread strides the axis, then the simdgroups fold
// their pairs, then simdgroup 0 folds the per-simdgroup pairs. The result is valid in thread 0.
template <typename T>
inline int argmax_threadgroup(device const T* src, size_m src_idx, size_m reduce_axis_stride, size_m noOfOpp,
                              uint tid, uint threads_per_tg, uint simd_lane_id, uint simd_group_id, uint num_simdgroups,
                              threadgroup T* shared_val, threadgroup uint* shared_idx) {
    T best = argmax_lowest<T>();
    uint best_idx = 0;
    for (size_m i = tid; i < noOfOpp; i += threads_per_tg) {
        T val = src[src_idx + i * reduce_axis_stride];
        if (val > best) { best = val; best_idx = i; }
    }
    argmax_simd(best, best_idx);

    if (simd_lane_id == 0) {
        shared_val[simd_group_id] = best;
        shared_idx[simd_group_id] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        best = argmax_lowest<T>();
        best_idx = 0;
        if (simd_lane_id < num_simdgroups) {
            best = shared_val[simd_lane_id];
            best_idx = shared_idx[simd_lane_id];
        }
        argmax_simd(best, best_idx);
    }
    return int(best_idx);
}

template <typename T>
[[kernel]] void argmax_gg_nd1(device int* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m& dst_stride [[buffer(4)]], constant const size_m& src_stride [[buffer(5)]], uint index [[thread_position_in_grid]]) {
    dst[index * dst_stride] = argmax_row(src, index * src_stride, reduce_axis_stride, noOfOpp);
}

template <typename T>
[[kernel]] void argmax_gg_nd2(device int* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]], uint2 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[1] + index.y * src_strides[0];
    size_m dst_idx = index.x * dst_strides[1] + index.y * dst_strides[0];
    dst[dst_idx] = argmax_row(src, src_idx, reduce_axis_stride, noOfOpp);
}

template <typename T>
[[kernel]] void argmax_gg_nd3(device int* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[2] + index.y * src_strides[1] + index.z * src_strides[0];
    size_m dst_idx = index.x * dst_strides[2] + index.y * dst_strides[1] + index.z * dst_strides[0];
    dst[dst_idx] = argmax_row(src, src_idx, reduce_axis_stride, noOfOpp);
}

template <typename T>
[[kernel]] void argmax_gg(device int* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]], constant const size_m* shape [[buffer(6)]], constant const int& ndim [[buffer(7)]], uint3 index [[thread_position_in_grid]]) {
    size_m dst_idx = index.x * dst_strides[ndim - 1] + index.y * dst_strides[ndim - 2];
    size_m src_idx = index.x * src_strides[ndim - 1] + index.y * src_strides[ndim - 2];
    size_m remaining = index.z;
    for (int i = ndim - 3; i >= 0; i--) {
        size_m coord = remaining % shape[i];
        dst_idx += coord * dst_strides[i];
        src_idx += coord * src_strides[i];
        remaining /= shape[i];
    }
    dst[dst_idx] = argmax_row(src, src_idx, reduce_axis_stride, noOfOpp);
}

template <typename T>
[[kernel]] void argmax_gg_tgr_nd1(device int* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m& dst_stride [[buffer(4)]], constant const size_m& src_stride [[buffer(5)]],
                              uint tgid                   [[threadgroup_position_in_grid]],
                              uint tid                    [[thread_position_in_threadgroup]],
                              uint threads_per_tg         [[threads_per_threadgroup]],
                              uint simd_lane_id           [[thread_index_in_simdgroup]],
                              uint simd_group_id          [[simdgroup_index_in_threadgroup]],
                              uint num_simdgroups         [[simdgroups_per_threadgroup]]) {
    threadgroup T shared_val[32];
    threadgroup uint shared_idx[32];
    int idx = argmax_threadgroup(src, tgid * src_stride, reduce_axis_stride, noOfOpp, tid, threads_per_tg,
                                 simd_lane_id, simd_group_id, num_simdgroups, shared_val, shared_idx);
    if (tid == 0) dst[tgid * dst_stride] = idx;
}

template <typename T>
[[kernel]] void argmax_gg_tgr_nd2(device int* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]],
                              uint2 tgid                  [[threadgroup_position_in_grid]],
                              uint2 tid_vec               [[thread_position_in_threadgroup]],
                              uint2 threads_per_tg_vec    [[threads_per_threadgroup]],
                              uint simd_lane_id           [[thread_index_in_simdgroup]],
                              uint simd_group_id          [[simdgroup_index_in_threadgroup]],
                              uint num_simdgroups         [[simdgroups_per_threadgroup]]) {
    threadgroup T shared_val[32];
    threadgroup uint shared_idx[32];
    size_m src_idx = tgid.x * src_strides[1] + tgid.y * src_strides[0];
    int idx = argmax_threadgroup(src, src_idx, reduce_axis_stride, noOfOpp, tid_vec.x, threads_per_tg_vec.x,
                                 simd_lane_id, simd_group_id, num_simdgroups, shared_val, shared_idx);
    if (tid_vec.x == 0) dst[tgid.x * dst_strides[1] + tgid.y * dst_strides[0]] = idx;
}

template <typename T>
[[kernel]] void argmax_gg_tgr_nd3(device int* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]],
                              uint3 tgid                  [[threadgroup_position_in_grid]],
                              uint3 tid_vec               [[thread_position_in_threadgroup]],
                              uint3 threads_per_tg_vec    [[threads_per_threadgroup]],
                              uint simd_lane_id           [[thread_index_in_simdgroup]],
                              uint simd_group_id          [[simdgroup_index_in_threadgroup]],
                              uint num_simdgroups         [[simdgroups_per_threadgroup]]) {
    threadgroup T shared_val[32];
    threadgroup uint shared_idx[32];
    size_m src_idx = tgid.x * src_strides[2] + tgid.y * src_strides[1] + tgid.z * src_strides[0];
    int idx = argmax_threadgroup(src, src_idx, reduce_axis_stride, noOfOpp, tid_vec.x, threads_per_tg_vec.x,
                                 simd_lane_id, simd_group_id, num_simdgroups, shared_val, shared_idx);
    if (tid_vec.x == 0) dst[tgid.x * dst_strides[2] + tgid.y * dst_strides[1] + tgid.z * dst_strides[0]] = idx;
}

template <typename T>
[[kernel]] void argmax_gg_tgr_nd(device int* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]], constant const size_m* shape [[buffer(6)]], constant const int& ndim [[buffer(7)]],
                             uint3 tgid                   [[threadgroup_position_in_grid]],
                             uint3 tid_vec                [[thread_position_in_threadgroup]],
                             uint3 threads_per_tg_vec     [[threads_per_threadgroup]],
                             uint simd_lane_id            [[thread_index_in_simdgroup]],
                             uint simd_group_id           [[simdgroup_index_in_threadgroup]],
                             uint num_simdgroups          [[simdgroups_per_threadgroup]]) {
    threadgroup T shared_val[32];
    threadgroup uint shared_idx[32];
    size_m dst_idx = tgid.x * dst_strides[ndim - 1] + tgid.y * dst_strides[ndim - 2];
    size_m src_idx = tgid.x * src_strides[ndim - 1] + tgid.y * src_strides[ndim - 2];
    size_m remaining = tgid.z;
    for (int i = ndim - 3; i >= 0; i--) {
        size_m coord = remaining % shape[i];
        dst_idx += coord * dst_strides[i];
        src_idx += coord * src_strides[i];
        remaining /= shape[i];
    }
    int idx = argmax_threadgroup(src, src_idx, reduce_axis_stride, noOfOpp, tid_vec.x, threads_per_tg_vec.x,
                                 simd_lane_id, simd_group_id, num_simdgroups, shared_val, shared_idx);
    if (tid_vec.x == 0) dst[dst_idx] = idx;
}

#define INSTANTIATE_ARGMAX(type_code, type) \
    instantiate_kernel("ArgMaxGPU_" #type_code "_0", argmax_gg_nd1, type); \
    instantiate_kernel("ArgMaxGPU_" #type_code "_1", argmax_gg_nd2, type); \
    instantiate_kernel("ArgMaxGPU_" #type_code "_2", argmax_gg_nd3, type); \
    instantiate_kernel("ArgMaxGPU_" #type_code "_3", argmax_gg, type); \
    instantiate_kernel("ArgMaxGPU_" #type_code "_4", argmax_gg_tgr_nd1, type); \
    instantiate_kernel("ArgMaxGPU_" #type_code "_5", argmax_gg_tgr_nd2, type); \
    instantiate_kernel("ArgMaxGPU_" #type_code "_6", argmax_gg_tgr_nd3, type); \
    instantiate_kernel("ArgMaxGPU_" #type_code "_7", argmax_gg_tgr_nd, type);

INSTANTIATE_ARGMAX(0, float)
INSTANTIATE_ARGMAX(1, half)
INSTANTIATE_ARGMAX(2, uint8_t)
INSTANTIATE_ARGMAX(3, int)
INSTANTIATE_ARGMAX(4, int16_t)
INSTANTIATE_ARGMAX(5, uint32_t)
INSTANTIATE_ARGMAX(6, uint16_t)
INSTANTIATE_ARGMAX(7, bool)
INSTANTIATE_ARGMAX(8, int8_t)
INSTANTIATE_ARGMAX(10, bfloat)
INSTANTIATE_ARGMAX(11, int64_t)
INSTANTIATE_ARGMAX(12, uint64_t)
