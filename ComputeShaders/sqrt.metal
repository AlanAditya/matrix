//
//  sqrt.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 03/11/25.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

template <typename T>
kernel void SqrtGPU(device T* outMat [[buffer(0)]], device const T* inMat[[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    outMat[gid] = sqrt(inMat[gid]);
}

instantiate_kernel("SqrtGPU_0", SqrtGPU, float);

// 1D Optimized Sqrt
template <typename T>
[[kernel]] void sqrt_gg_nd1(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& dst_stride [[buffer(2)]], constant const size_m& src_stride [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    dst[index * dst_stride] = sqrt(src[index * src_stride]);
}

// 2D Optimized Sqrt
template <typename T>
[[kernel]] void sqrt_gg_nd2(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], uint2 index [[thread_position_in_grid]]) {
    auto src_idx = index.x * src_strides[1] + index.y * src_strides[0];
    auto dst_idx = index.x * dst_strides[1] + index.y * dst_strides[0];
    dst[dst_idx] = sqrt(src[src_idx]);
}

// 3D Optimized Sqrt
template <typename T>
[[kernel]] void sqrt_gg_nd3(device T* dst [[buffer(0)]], const device T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[2] + index.y * src_strides[1] + index.z * src_strides[0];
    size_m dst_idx = index.x * dst_strides[2] + index.y * dst_strides[1] + index.z * dst_strides[0];
    dst[dst_idx] = sqrt(src[src_idx]);
}

// ND Optimized Sqrt
template <typename T>
[[kernel]] void sqrt_gg(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const size_m* src_shape [[buffer(4)]], constant const int& ndim [[buffer(5)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[ndim-1] + index.y * src_strides[ndim-2];
    size_m dst_idx = index.x * dst_strides[ndim-1] + index.y * dst_strides[ndim-2];
    uint remA = index.z;
    for (int i = ndim-3; i >= 0; --i) {
        src_idx += remA % src_shape[i] * src_strides[i];
        dst_idx += remA % src_shape[i] * dst_strides[i];
        remA /= src_shape[i];
    }
    dst[dst_idx] = sqrt(src[src_idx]);
}

#define INSTANTIATE_SQRT_ND(type_idx, type) \
    instantiate_kernel("SqrtGPU_nd_" #type_idx "_0", sqrt_gg_nd1, type); \
    instantiate_kernel("SqrtGPU_nd_" #type_idx "_1", sqrt_gg_nd2, type); \
    instantiate_kernel("SqrtGPU_nd_" #type_idx "_2", sqrt_gg_nd3, type); \
    instantiate_kernel("SqrtGPU_nd_" #type_idx "_3", sqrt_gg, type);

INSTANTIATE_SQRT_ND(0, float)
INSTANTIATE_SQRT_ND(1, half)
