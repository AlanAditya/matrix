//
//  clamp.metal
//  AdityaIntelligenceProMax
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

// 1D Optimized Clamp
template <typename T>
[[kernel]] void clamp_gg_nd1(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& dst_stride [[buffer(2)]], constant const size_m& src_stride [[buffer(3)]], constant const T& min_val [[buffer(6)]], constant const T& max_val [[buffer(7)]], uint index [[thread_position_in_grid]]) {
    dst[index * dst_stride] = max(min_val, min(src[index * src_stride], max_val));
}

// 2D Optimized Clamp
template <typename T>
[[kernel]] void clamp_gg_nd2(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const T& min_val [[buffer(6)]], constant const T& max_val [[buffer(7)]], uint2 index [[thread_position_in_grid]]) {
    auto src_idx = index.x * src_strides[1] + index.y * src_strides[0];
    auto dst_idx = index.x * dst_strides[1] + index.y * dst_strides[0];
    dst[dst_idx] = max(min_val, min(src[src_idx], max_val));
}

// 3D Optimized Clamp
template <typename T>
[[kernel]] void clamp_gg_nd3(device T* dst [[buffer(0)]], const device T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const T& min_val [[buffer(6)]], constant const T& max_val [[buffer(7)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[2] + index.y * src_strides[1] + index.z * src_strides[0];
    size_m dst_idx = index.x * dst_strides[2] + index.y * dst_strides[1] + index.z * dst_strides[0];
    dst[dst_idx] = max(min_val, min(src[src_idx], max_val));
}

// ND Optimized Clamp
template <typename T>
[[kernel]] void clamp_gg(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const size_m* src_shape [[buffer(4)]], constant const int& ndim [[buffer(5)]], constant const T& min_val [[buffer(6)]], constant const T& max_val [[buffer(7)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[ndim-1] + index.y * src_strides[ndim-2];
    size_m dst_idx = index.x * dst_strides[ndim-1] + index.y * dst_strides[ndim-2];
    uint remA = index.z;
    for (int i = ndim-3; i >= 0; --i) {
        src_idx += remA % src_shape[i] * src_strides[i];
        dst_idx += remA % src_shape[i] * dst_strides[i];
        remA /= src_shape[i];
    }
    dst[dst_idx] = max(min_val, min(src[src_idx], max_val));
}

#define INSTANTIATE_CLAMP(type_idx, type) \
    instantiate_kernel("ClampGPU_nd_" #type_idx "_0", clamp_gg_nd1, type); \
    instantiate_kernel("ClampGPU_nd_" #type_idx "_1", clamp_gg_nd2, type); \
    instantiate_kernel("ClampGPU_nd_" #type_idx "_2", clamp_gg_nd3, type); \
    instantiate_kernel("ClampGPU_nd_" #type_idx "_3", clamp_gg, type);

INSTANTIATE_CLAMP(0, float)
INSTANTIATE_CLAMP(1, half)
INSTANTIATE_CLAMP(3, int)
INSTANTIATE_CLAMP(4, short)
INSTANTIATE_CLAMP(5, uint)
INSTANTIATE_CLAMP(2, uchar)
INSTANTIATE_CLAMP(6, ushort)
