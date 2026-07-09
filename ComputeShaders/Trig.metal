//
//  Trig.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 09/08/25.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

template <typename T>
kernel void SinGPU(device T* outMat [[buffer(0)]], device const T* inMat[[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    outMat[gid] = sin(inMat[gid]);
}

instantiate_kernel("SinGPU_0", SinGPU, float);


template <typename T>
kernel void CosGPU(device T* outMat [[buffer(0)]], device const T* inMat[[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    outMat[gid] = cos(inMat[gid]);
}

instantiate_kernel("CosGPU_0", CosGPU, float);


template <typename T>
kernel void TanGPU(device T* outMat [[buffer(0)]], device const T* inMat[[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    outMat[gid] = tan(inMat[gid]);
}

instantiate_kernel("TanGPU_0", TanGPU, float);

// 1D Optimized Sin
template <typename T>
[[kernel]] void sin_gg_nd1(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& dst_stride [[buffer(2)]], constant const size_m& src_stride [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    dst[index * dst_stride] = sin(src[index * src_stride]);
}

// 2D Optimized Sin
template <typename T>
[[kernel]] void sin_gg_nd2(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], uint2 index [[thread_position_in_grid]]) {
    auto src_idx = index.x * src_strides[1] + index.y * src_strides[0];
    auto dst_idx = index.x * dst_strides[1] + index.y * dst_strides[0];
    dst[dst_idx] = sin(src[src_idx]);
}

// 3D Optimized Sin
template <typename T>
[[kernel]] void sin_gg_nd3(device T* dst [[buffer(0)]], const device T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[2] + index.y * src_strides[1] + index.z * src_strides[0];
    size_m dst_idx = index.x * dst_strides[2] + index.y * dst_strides[1] + index.z * dst_strides[0];
    dst[dst_idx] = sin(src[src_idx]);
}

// ND Optimized Sin
template <typename T>
[[kernel]] void sin_gg(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const size_m* src_shape [[buffer(4)]], constant const int& ndim [[buffer(5)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[ndim-1] + index.y * src_strides[ndim-2];
    size_m dst_idx = index.x * dst_strides[ndim-1] + index.y * dst_strides[ndim-2];
    uint remA = index.z;
    for (int i = ndim-3; i >= 0; --i) {
        src_idx += remA % src_shape[i] * src_strides[i];
        dst_idx += remA % src_shape[i] * dst_strides[i];
        remA /= src_shape[i];
    }
    dst[dst_idx] = sin(src[src_idx]);
}

#define INSTANTIATE_SIN_ND(type_idx, type) \
    instantiate_kernel("SinGPU_nd_" #type_idx "_0", sin_gg_nd1, type); \
    instantiate_kernel("SinGPU_nd_" #type_idx "_1", sin_gg_nd2, type); \
    instantiate_kernel("SinGPU_nd_" #type_idx "_2", sin_gg_nd3, type); \
    instantiate_kernel("SinGPU_nd_" #type_idx "_3", sin_gg, type);

INSTANTIATE_SIN_ND(0, float)
INSTANTIATE_SIN_ND(1, half)

// 1D Optimized Cos
template <typename T>
[[kernel]] void cos_gg_nd1(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& dst_stride [[buffer(2)]], constant const size_m& src_stride [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    dst[index * dst_stride] = cos(src[index * src_stride]);
}

// 2D Optimized Cos
template <typename T>
[[kernel]] void cos_gg_nd2(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], uint2 index [[thread_position_in_grid]]) {
    auto src_idx = index.x * src_strides[1] + index.y * src_strides[0];
    auto dst_idx = index.x * dst_strides[1] + index.y * dst_strides[0];
    dst[dst_idx] = cos(src[src_idx]);
}

// 3D Optimized Cos
template <typename T>
[[kernel]] void cos_gg_nd3(device T* dst [[buffer(0)]], const device T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[2] + index.y * src_strides[1] + index.z * src_strides[0];
    size_m dst_idx = index.x * dst_strides[2] + index.y * dst_strides[1] + index.z * dst_strides[0];
    dst[dst_idx] = cos(src[src_idx]);
}

// ND Optimized Cos
template <typename T>
[[kernel]] void cos_gg(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const size_m* src_shape [[buffer(4)]], constant const int& ndim [[buffer(5)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[ndim-1] + index.y * src_strides[ndim-2];
    size_m dst_idx = index.x * dst_strides[ndim-1] + index.y * dst_strides[ndim-2];
    uint remA = index.z;
    for (int i = ndim-3; i >= 0; --i) {
        src_idx += remA % src_shape[i] * src_strides[i];
        dst_idx += remA % src_shape[i] * dst_strides[i];
        remA /= src_shape[i];
    }
    dst[dst_idx] = cos(src[src_idx]);
}

#define INSTANTIATE_COS_ND(type_idx, type) \
    instantiate_kernel("CosGPU_nd_" #type_idx "_0", cos_gg_nd1, type); \
    instantiate_kernel("CosGPU_nd_" #type_idx "_1", cos_gg_nd2, type); \
    instantiate_kernel("CosGPU_nd_" #type_idx "_2", cos_gg_nd3, type); \
    instantiate_kernel("CosGPU_nd_" #type_idx "_3", cos_gg, type);

INSTANTIATE_COS_ND(0, float)
INSTANTIATE_COS_ND(1, half)

// 1D Optimized Tan
template <typename T>
[[kernel]] void tan_gg_nd1(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& dst_stride [[buffer(2)]], constant const size_m& src_stride [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    dst[index * dst_stride] = tan(src[index * src_stride]);
}

// 2D Optimized Tan
template <typename T>
[[kernel]] void tan_gg_nd2(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], uint2 index [[thread_position_in_grid]]) {
    auto src_idx = index.x * src_strides[1] + index.y * src_strides[0];
    auto dst_idx = index.x * dst_strides[1] + index.y * dst_strides[0];
    dst[dst_idx] = tan(src[src_idx]);
}

// 3D Optimized Tan
template <typename T>
[[kernel]] void tan_gg_nd3(device T* dst [[buffer(0)]], const device T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[2] + index.y * src_strides[1] + index.z * src_strides[0];
    size_m dst_idx = index.x * dst_strides[2] + index.y * dst_strides[1] + index.z * dst_strides[0];
    dst[dst_idx] = tan(src[src_idx]);
}

// ND Optimized Tan
template <typename T>
[[kernel]] void tan_gg(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const size_m* src_shape [[buffer(4)]], constant const int& ndim [[buffer(5)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[ndim-1] + index.y * src_strides[ndim-2];
    size_m dst_idx = index.x * dst_strides[ndim-1] + index.y * dst_strides[ndim-2];
    uint remA = index.z;
    for (int i = ndim-3; i >= 0; --i) {
        src_idx += remA % src_shape[i] * src_strides[i];
        dst_idx += remA % src_shape[i] * dst_strides[i];
        remA /= src_shape[i];
    }
    dst[dst_idx] = tan(src[src_idx]);
}

#define INSTANTIATE_TAN_ND(type_idx, type) \
    instantiate_kernel("TanGPU_nd_" #type_idx "_0", tan_gg_nd1, type); \
    instantiate_kernel("TanGPU_nd_" #type_idx "_1", tan_gg_nd2, type); \
    instantiate_kernel("TanGPU_nd_" #type_idx "_2", tan_gg_nd3, type); \
    instantiate_kernel("TanGPU_nd_" #type_idx "_3", tan_gg, type);

INSTANTIATE_TAN_ND(0, float)
INSTANTIATE_TAN_ND(1, half)

