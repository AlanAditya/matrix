//
//  Min.metal
//  AdityaIntelligenceProMax
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

template <typename T>
[[kernel]] void min_gg_nd1(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m& dst_stride [[buffer(4)]], constant const size_m& src_stride [[buffer(5)]], uint index [[thread_position_in_grid]]) {
    size_m src_idx = index * src_stride;
    T current_min = src[src_idx];
    for (size_m i = 1; i < noOfOpp; i++) {
        T val = src[src_idx + i * reduce_axis_stride];
        if (val < current_min) current_min = val;
    }
    dst[index * dst_stride] = current_min;
}

template <typename T>
[[kernel]] void min_gg_nd2(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]], uint2 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[1] + index.y * src_strides[0];
    size_m dst_idx = index.x * dst_strides[1] + index.y * dst_strides[0];
    T current_min = src[src_idx];
    for (size_m i = 1; i < noOfOpp; i++) {
        T val = src[src_idx + i * reduce_axis_stride];
        if (val < current_min) current_min = val;
    }
    dst[dst_idx] = current_min;
}

template <typename T>
[[kernel]] void min_gg_nd3(device T* dst [[buffer(0)]], const device T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[2] + index.y * src_strides[1] + index.z * src_strides[0];
    size_m dst_idx = index.x * dst_strides[2] + index.y * dst_strides[1] + index.z * dst_strides[0];
    T current_min = src[src_idx];
    for (size_m i = 1; i < noOfOpp; i++) {
        T val = src[src_idx + i * reduce_axis_stride];
        if (val < current_min) current_min = val;
    }
    dst[dst_idx] = current_min;
}

template <typename T>
[[kernel]] void min_gg(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]], constant const size_m* shape [[buffer(6)]], constant const int& ndim [[buffer(7)]], uint3 index [[thread_position_in_grid]]) {
    size_m dst_idx = index.x * dst_strides[ndim - 1] + index.y * dst_strides[ndim - 2];
    size_m src_idx = index.x * src_strides[ndim - 1] + index.y * src_strides[ndim - 2];
    size_m remaining = index.z;
    for (int i = ndim - 3; i >= 0; i--) {
        size_m coord = remaining % shape[i];
        dst_idx += coord * dst_strides[i];
        src_idx += coord * src_strides[i];
        remaining /= shape[i];
    }
    T current_min = src[src_idx];
    for (size_m i = 1; i < noOfOpp; i++) {
        T val = src[src_idx + i * reduce_axis_stride];
        if (val < current_min) current_min = val;
    }
    dst[dst_idx] = current_min;
}


template <typename T>
[[kernel]] void min_gg_tgr_nd1(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m& dst_stride [[buffer(4)]], constant const size_m& src_stride [[buffer(5)]],
                           uint tgid                   [[threadgroup_position_in_grid]],
                           uint tid                     [[thread_position_in_threadgroup]],
                           uint threads_per_tg          [[threads_per_threadgroup]],
                           uint simd_lane_id            [[thread_index_in_simdgroup]],
                           uint simd_group_id           [[simdgroup_index_in_threadgroup]],
                           uint num_simdgroups          [[simdgroups_per_threadgroup]]) {
    
    size_m reduced_src_idx = tgid * src_stride;
    T current_min = numeric_limits<T>::max();
    for (size_m i = tid; i < noOfOpp; i+= threads_per_tg) {
        current_min = min(current_min, src[reduced_src_idx + i * reduce_axis_stride]);
    }
    
    T simd_result = simd_min(current_min);
    threadgroup T shared_min[32];
    if (simd_lane_id == 0) {
        shared_min[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group_id == 0) {
        T v = (simd_lane_id < num_simdgroups) ? shared_min[simd_lane_id] : numeric_limits<T>::max();
        current_min = simd_min(v);
        if (simd_lane_id == 0)
            dst[tgid * dst_stride] = current_min;
    }
}

template <typename T>
[[kernel]] void min_gg_tgr_nd2(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]],
                           uint2 tgid                   [[threadgroup_position_in_grid]],
                           uint2 tid_vec                [[thread_position_in_threadgroup]],
                           uint2 threads_per_tg_vec     [[threads_per_threadgroup]],
                           uint simd_lane_id            [[thread_index_in_simdgroup]],
                           uint simd_group_id           [[simdgroup_index_in_threadgroup]],
                           uint num_simdgroups          [[simdgroups_per_threadgroup]]) {
    
    uint tid = tid_vec.x;
    uint threads_per_tg = threads_per_tg_vec.x;
    size_m reduced_src_idx = tgid.x * src_strides[1] + tgid.y * src_strides[0];
    T current_min = numeric_limits<T>::max();
    for (size_m i = tid; i < noOfOpp; i+= threads_per_tg) {
        current_min = min(current_min, src[reduced_src_idx + i * reduce_axis_stride]);
    }
    
    T simd_result = simd_min(current_min);
    threadgroup T shared_min[32];
    if (simd_lane_id == 0) {
        shared_min[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group_id == 0) {
        T v = (simd_lane_id < num_simdgroups) ? shared_min[simd_lane_id] : numeric_limits<T>::max();
        current_min = simd_min(v);
        if (simd_lane_id == 0)
            dst[tgid.x * dst_strides[1] + tgid.y * dst_strides[0]] = current_min;
    }
}

template <typename T>
[[kernel]] void min_gg_tgr_nd3(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]],
                           uint3 tgid                   [[threadgroup_position_in_grid]],
                           uint3 tid_vec                [[thread_position_in_threadgroup]],
                           uint3 threads_per_tg_vec     [[threads_per_threadgroup]],
                           uint simd_lane_id            [[thread_index_in_simdgroup]],
                           uint simd_group_id           [[simdgroup_index_in_threadgroup]],
                           uint num_simdgroups          [[simdgroups_per_threadgroup]]) {
    
    uint tid = tid_vec.x;
    uint threads_per_tg = threads_per_tg_vec.x;
    size_m reduced_src_idx = tgid.x * src_strides[2] + tgid.y * src_strides[1] + tgid.z * src_strides[0];
    T current_min = numeric_limits<T>::max();
    for (size_m i = tid; i < noOfOpp; i+= threads_per_tg) {
        current_min = min(current_min, src[reduced_src_idx + i * reduce_axis_stride]);
    }
    
    T simd_result = simd_min(current_min);
    threadgroup T shared_min[32];
    if (simd_lane_id == 0) {
        shared_min[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group_id == 0) {
        T v = (simd_lane_id < num_simdgroups) ? shared_min[simd_lane_id] : numeric_limits<T>::max();
        current_min = simd_min(v);
        if (simd_lane_id == 0)
            dst[tgid.x * dst_strides[2] + tgid.y * dst_strides[1] + tgid.z * dst_strides[0]] = current_min;
    }
}

template <typename T>
[[kernel]] void min_gg_tgr_nd(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]], constant const size_m* shape [[buffer(6)]], constant const int& ndim [[buffer(7)]],
                       uint3 tgid                   [[threadgroup_position_in_grid]],
                       uint3 tid_vec                [[thread_position_in_threadgroup]],
                       uint3 threads_per_tg_vec     [[threads_per_threadgroup]],
                       uint simd_lane_id            [[thread_index_in_simdgroup]],
                       uint simd_group_id           [[simdgroup_index_in_threadgroup]],
                       uint num_simdgroups          [[simdgroups_per_threadgroup]]) {
    
    uint tid = tid_vec.x;
    uint threads_per_tg = threads_per_tg_vec.x;
    size_m dst_idx = tgid.x * dst_strides[ndim - 1] + tgid.y * dst_strides[ndim - 2];
    size_m src_idx = tgid.x * src_strides[ndim - 1] + tgid.y * src_strides[ndim - 2];
    
    size_m remaining = tgid.z;
    for (int i = ndim - 3; i >= 0; i--) {
        size_m coord = remaining % shape[i];
        dst_idx += coord * dst_strides[i];
        src_idx += coord * src_strides[i];
        remaining /= shape[i];
    }
    
    T current_min = numeric_limits<T>::max();
    for (size_m i = tid; i < noOfOpp; i+= threads_per_tg) {
        current_min = min(current_min, src[src_idx + i * reduce_axis_stride]);
    }
    
    T simd_result = simd_min(current_min);
    threadgroup T shared_min[32];
    
    if (simd_lane_id == 0) {
        shared_min[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group_id == 0) {
        T v = (simd_lane_id < num_simdgroups) ? shared_min[simd_lane_id] : numeric_limits<T>::max();
        current_min = simd_min(v);
        if (simd_lane_id == 0)
            dst[dst_idx] = current_min;
    }
}

#define INSTANTIATE_MIN(type_code, type) \
    instantiate_kernel("MinGPU_" #type_code "_0", min_gg_nd1, type); \
    instantiate_kernel("MinGPU_" #type_code "_1", min_gg_nd2, type); \
    instantiate_kernel("MinGPU_" #type_code "_2", min_gg_nd3, type); \
    instantiate_kernel("MinGPU_" #type_code "_3", min_gg, type); \
    instantiate_kernel("MinGPU_" #type_code "_4", min_gg_tgr_nd1, type); \
    instantiate_kernel("MinGPU_" #type_code "_5", min_gg_tgr_nd2, type); \
    instantiate_kernel("MinGPU_" #type_code "_6", min_gg_tgr_nd3, type); \
    instantiate_kernel("MinGPU_" #type_code "_7", min_gg_tgr_nd, type);

INSTANTIATE_MIN(0, float)
INSTANTIATE_MIN(1, half)
INSTANTIATE_MIN(2, uint8_t)
INSTANTIATE_MIN(3, int)
INSTANTIATE_MIN(4, int16_t)
INSTANTIATE_MIN(5, uint32_t)
INSTANTIATE_MIN(6, uint16_t)
