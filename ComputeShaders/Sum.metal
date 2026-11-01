//
//  Sum.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 25/07/25.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

template <typename T>
kernel void SumGPU(device T* out_buffer [[buffer(0)]], device const T* in_buffer [[buffer(1)]], constant size_m& AxisStride [[buffer(2)]], constant size_m& ElStride [[buffer(3)]],constant size_m& noOfOpp [[buffer(4)]], constant size_m* inputStrides [[buffer(5)]], constant size_m* maskedStrides [[buffer(6)]], constant size_m& dims [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
    size_t AxisOffset = 0;
    size_t remaining = gid;
    for (size_t i = 0; i < dims; i++) {
        AxisOffset += (remaining / inputStrides[i]) * maskedStrides[i];
        remaining %= inputStrides[i];
    }
    out_buffer[gid] = 0;
    for (size_t i = 0; i < noOfOpp; i++) {
        out_buffer[gid] += in_buffer[AxisOffset + i * ElStride];
    }
}

instantiate_kernel("SumGPU_0", SumGPU, float);
instantiate_kernel("SumGPU_1", SumGPU, half);
instantiate_kernel("SumGPU_2", SumGPU, uint8_t);
instantiate_kernel("SumGPU_3", SumGPU, int);


template <typename T>
[[kernel]] void sum_gg_nd1(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m& dst_stride [[buffer(4)]], constant const size_m& src_stride [[buffer(5)]], uint index [[thread_position_in_grid]]) {
    size_m src_idx = index * src_stride;
    T current_sum = src[src_idx];
    for (size_m i = 1; i < noOfOpp; i++) {
        T val = src[src_idx + i * reduce_axis_stride];
        current_sum += val;
    }
    dst[index * dst_stride] = current_sum;
}

template <typename T>
[[kernel]] void sum_gg_nd2(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]], uint2 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[1] + index.y * src_strides[0];
    size_m dst_idx = index.x * dst_strides[1] + index.y * dst_strides[0];
    T current_sum = src[src_idx];
    for (size_m i = 1; i < noOfOpp; i++) {
        T val = src[src_idx + i * reduce_axis_stride];
        current_sum += val;
    }
    dst[dst_idx] = current_sum;
}

template <typename T>
[[kernel]] void sum_gg_nd3(device T* dst [[buffer(0)]], const device T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[2] + index.y * src_strides[1] + index.z * src_strides[0];
    size_m dst_idx = index.x * dst_strides[2] + index.y * dst_strides[1] + index.z * dst_strides[0];
    T current_sum = src[src_idx];
    for (size_m i = 1; i < noOfOpp; i++) {
        T val = src[src_idx + i * reduce_axis_stride];
        current_sum += val;
    }
    dst[dst_idx] = current_sum;
}

template <typename T>
[[kernel]] void sum_gg_nd(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]], constant const size_m* shape [[buffer(6)]], constant const int& ndim [[buffer(7)]], uint3 index [[thread_position_in_grid]]) {
    size_m dst_idx = index.x * dst_strides[ndim - 1] + index.y * dst_strides[ndim - 2];
    size_m src_idx = index.x * src_strides[ndim - 1] + index.y * src_strides[ndim - 2];
    size_m remaining = index.z;
    for (int i = ndim - 3; i >= 0; i--) {
        size_m coord = remaining % shape[i];
        dst_idx += coord * dst_strides[i];
        src_idx += coord * src_strides[i];
        remaining /= shape[i];
    }
    T current_sum = src[src_idx];
    for (size_m i = 1; i < noOfOpp; i++) {
        T val = src[src_idx + i * reduce_axis_stride];
        current_sum += val;
    }
    dst[dst_idx] = current_sum;
}





template <typename T>
[[kernel]] void sum_gg_tgr_nd1(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m& dst_stride [[buffer(4)]], constant const size_m& src_stride [[buffer(5)]],
                           uint tgid                   [[threadgroup_position_in_grid]],
                           uint tid                     [[thread_position_in_threadgroup]],
                           uint threads_per_tg          [[threads_per_threadgroup]],
                           uint simd_lane_id            [[thread_index_in_simdgroup]],
                           uint simd_group_id           [[simdgroup_index_in_threadgroup]],
                           uint num_simdgroups          [[simdgroups_per_threadgroup]]) {
    
    // for the reduced axis "threads_per_tg" threads are launched and each threadgoup is responsible for a element of the output
    
    size_m reduced_src_idx = tgid * src_stride;
    
    // phase one of reduction each thread computes some (reduced_axis_shape //  threads_per_tg) elements along the reduced axis
//    T current_sum = src[reduced_src_idx + tid * reduce_axis_stride]; would result in out of bounds access if threads_per_threadgroup >= reduce_axis_shape
    T current_sum = 0;
    
    for (size_m i = tid; i < noOfOpp; i+= threads_per_tg) {
        current_sum += src[reduced_src_idx + i * reduce_axis_stride];
    }
    
    // phase 2 of reduction about 32 threads collase their output
    T simd_result = simd_sum(current_sum);
    
    threadgroup T shared_max[32]; // some large no. so max possible simdgroups_per_threadgroup
    
    if (simd_lane_id == 0) {
        shared_max[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // lets lets assign the max's of each simd_group to the 0th simd_group threads to further find the max
    if (simd_group_id == 0) {
        T v = (simd_lane_id < num_simdgroups) ? shared_max[simd_lane_id] : 0;
        current_sum = simd_sum(v);
        if (simd_lane_id == 0)
            dst[tgid * dst_stride] = current_sum;
    }
}



template <typename T>
[[kernel]] void sum_gg_tgr_nd2(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]],
                           uint2 tgid                   [[threadgroup_position_in_grid]],
                           uint2 tid_vec                [[thread_position_in_threadgroup]],
                           uint2 threads_per_tg_vec     [[threads_per_threadgroup]],
                           uint simd_lane_id            [[thread_index_in_simdgroup]],
                           uint simd_group_id           [[simdgroup_index_in_threadgroup]],
                           uint num_simdgroups          [[simdgroups_per_threadgroup]]) {
    
    uint tid = tid_vec.x;
    uint threads_per_tg = threads_per_tg_vec.x;
    // for the reduced axis "threads_per_tg" threads are launched and each threadgoup is responsible for a element of the output
    
    size_m reduced_src_idx = tgid.x * src_strides[1] + tgid.y * src_strides[0];
    
    // phase one of reduction each thread computes some (reduced_axis_shape //  threads_per_tg) elements along the reduced axis
//    T current_sum = src[reduced_src_idx + tid * reduce_axis_stride]; would result in out of bounds access if threads_per_threadgroup >= reduce_axis_shape
    T current_sum = 0;
    
    for (size_m i = tid; i < noOfOpp; i+= threads_per_tg) {
        current_sum += src[reduced_src_idx + i * reduce_axis_stride];
    }
    
    // phase 2 of reduction about 32 threads collase their output
    T simd_result = simd_sum(current_sum);
    
    threadgroup T shared_max[32]; // some large no. so max possible simdgroups_per_threadgroup
    
    if (simd_lane_id == 0) {
        shared_max[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // lets lets assign the max's of each simd_group to the 0th simd_group threads to further find the max
    if (simd_group_id == 0) {
        T v = (simd_lane_id < num_simdgroups) ? shared_max[simd_lane_id] : 0;
        current_sum = simd_sum(v);
        if (simd_lane_id == 0)
            dst[tgid.x * dst_strides[1] + tgid.y * dst_strides[0]] = current_sum;
    }
}

template <typename T>
[[kernel]] void sum_gg_tgr_nd3(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]],
                           uint3 tgid                   [[threadgroup_position_in_grid]],
                           uint3 tid_vec                [[thread_position_in_threadgroup]],
                           uint3 threads_per_tg_vec     [[threads_per_threadgroup]],
                           uint simd_lane_id            [[thread_index_in_simdgroup]],
                           uint simd_group_id           [[simdgroup_index_in_threadgroup]],
                           uint num_simdgroups          [[simdgroups_per_threadgroup]]) {
    
    uint tid = tid_vec.x;
    uint threads_per_tg = threads_per_tg_vec.x;
    // for the reduced axis "threads_per_tg" threads are launched and each threadgoup is responsible for a element of the output
    
    size_m reduced_src_idx = tgid.x * src_strides[2] + tgid.y * src_strides[1] + tgid.z * src_strides[0];
    
    // phase one of reduction each thread computes some (reduced_axis_shape //  threads_per_tg) elements along the reduced axis
//    T current_sum = src[reduced_src_idx + tid * reduce_axis_stride]; would result in out of bounds access if threads_per_threadgroup >= reduce_axis_shape
    T current_sum = 0;
    
    for (size_m i = tid; i < noOfOpp; i+= threads_per_tg) {
        current_sum += src[reduced_src_idx + i * reduce_axis_stride];
    }
    
    // phase 2 of reduction about 32 threads collase their output
    T simd_result = simd_sum(current_sum);
    
    threadgroup T shared_max[32]; // some large no. so max possible simdgroups_per_threadgroup
    
    if (simd_lane_id == 0) {
        shared_max[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // lets lets assign the max's of each simd_group to the 0th simd_group threads to further find the max
    if (simd_group_id == 0) {
        T v = (simd_lane_id < num_simdgroups) ? shared_max[simd_lane_id] : 0;
        current_sum = simd_sum(v);
        if (simd_lane_id == 0)
            dst[tgid.x * dst_strides[2] + tgid.y * dst_strides[1] + tgid.z * dst_strides[0]] = current_sum;
    }
}

template <typename T>
[[kernel]] void sum_gg_tgr_nd(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& reduce_axis_stride [[buffer(2)]], constant const size_m& noOfOpp [[buffer(3)]], constant const size_m* dst_strides [[buffer(4)]], constant const size_m* src_strides [[buffer(5)]], constant const size_m* shape [[buffer(6)]], constant const int& ndim [[buffer(7)]],
                       uint3 tgid                   [[threadgroup_position_in_grid]],
                       uint3 tid_vec                [[thread_position_in_threadgroup]],
                       uint3 threads_per_tg_vec     [[threads_per_threadgroup]],
                       uint simd_lane_id            [[thread_index_in_simdgroup]],
                       uint simd_group_id           [[simdgroup_index_in_threadgroup]],
                       uint num_simdgroups          [[simdgroups_per_threadgroup]]) {
    
    uint tid = tid_vec.x;
    uint threads_per_tg = threads_per_tg_vec.x;
    //  strides given to the function have their reduce axis strides removed
    // ndim is the reduced rank
    size_m dst_idx = tgid.x * dst_strides[ndim - 1] + tgid.y * dst_strides[ndim - 2];
    size_m src_idx = tgid.x * src_strides[ndim - 1] + tgid.y * src_strides[ndim - 2];
    
    size_m remaining = tgid.z;
    for (int i = ndim - 3; i >= 0; i--) {
        size_m coord = remaining % shape[i];
        dst_idx += coord * dst_strides[i];
        src_idx += coord * src_strides[i];
        remaining /= shape[i];
    }
    
    // phase one each thread computes max of (reduced_axis_shape // threads_per_threadgroup)
    T current_sum = 0;
    
    for (size_m i = tid; i < noOfOpp; i+= threads_per_tg) {
        current_sum += src[src_idx + i * reduce_axis_stride];
    }
    
    // phase two, a simd_group is a group of 32 threads and we will find the max of them
    T simd_result = simd_sum(current_sum);
    
    // each simd_group puts its min in this array
    threadgroup T shared[32];
    
    if (simd_lane_id == 0) {
        shared[simd_group_id] = simd_result;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (simd_group_id == 0) {
        T v = (simd_lane_id < num_simdgroups) ? shared[simd_lane_id] : 0;
        current_sum = simd_sum(v);
        if (simd_lane_id == 0)
            dst[dst_idx] = current_sum;
    }
    
}



#define INSTANTIATE_SUM_ND(type_code, type) \
    instantiate_kernel("SumGPU_" #type_code "_0", sum_gg_nd1, type); \
    instantiate_kernel("SumGPU_" #type_code "_1", sum_gg_nd2, type); \
    instantiate_kernel("SumGPU_" #type_code "_2", sum_gg_nd3, type); \
    instantiate_kernel("SumGPU_" #type_code "_3", sum_gg_nd, type); \
    instantiate_kernel("SumGPU_" #type_code "_4", sum_gg_tgr_nd1, type); \
    instantiate_kernel("SumGPU_" #type_code "_5", sum_gg_tgr_nd2, type); \
    instantiate_kernel("SumGPU_" #type_code "_6", sum_gg_tgr_nd3, type); \
    instantiate_kernel("SumGPU_" #type_code "_7", sum_gg_tgr_nd, type);

INSTANTIATE_SUM_ND(0, float);
INSTANTIATE_SUM_ND(1, half);
INSTANTIATE_SUM_ND(2, uint8_t);
INSTANTIATE_SUM_ND(3, int);
INSTANTIATE_SUM_ND(4, int16_t);
INSTANTIATE_SUM_ND(8, int8_t);
