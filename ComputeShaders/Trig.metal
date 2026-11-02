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

// The default fast-math tanh overflows to NaN for large positive inputs (e.g. tanh(50)), so
// use the precise one, in float so half gets the same behaviour.
template <typename T>
inline T tanh_precise(T x) { return T(precise::tanh(float(x))); }

// 1D Optimized Tanh
template <typename T>
[[kernel]] void tanh_gg_nd1(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& dst_stride [[buffer(2)]], constant const size_m& src_stride [[buffer(3)]], uint index [[thread_position_in_grid]]) {
    dst[index * dst_stride] = tanh_precise(src[index * src_stride]);
}

// 2D Optimized Tanh
template <typename T>
[[kernel]] void tanh_gg_nd2(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], uint2 index [[thread_position_in_grid]]) {
    auto src_idx = index.x * src_strides[1] + index.y * src_strides[0];
    auto dst_idx = index.x * dst_strides[1] + index.y * dst_strides[0];
    dst[dst_idx] = tanh_precise(src[src_idx]);
}

// 3D Optimized Tanh
template <typename T>
[[kernel]] void tanh_gg_nd3(device T* dst [[buffer(0)]], const device T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[2] + index.y * src_strides[1] + index.z * src_strides[0];
    size_m dst_idx = index.x * dst_strides[2] + index.y * dst_strides[1] + index.z * dst_strides[0];
    dst[dst_idx] = tanh_precise(src[src_idx]);
}

// ND Optimized Tanh
template <typename T>
[[kernel]] void tanh_gg(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const size_m* src_shape [[buffer(4)]], constant const int& ndim [[buffer(5)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[ndim-1] + index.y * src_strides[ndim-2];
    size_m dst_idx = index.x * dst_strides[ndim-1] + index.y * dst_strides[ndim-2];
    uint remA = index.z;
    for (int i = ndim-3; i >= 0; --i) {
        src_idx += remA % src_shape[i] * src_strides[i];
        dst_idx += remA % src_shape[i] * dst_strides[i];
        remA /= src_shape[i];
    }
    dst[dst_idx] = tanh_precise(src[src_idx]);
}

#define INSTANTIATE_TANH_ND(type_idx, type) \
    instantiate_kernel("TanhGPU_nd_" #type_idx "_0", tanh_gg_nd1, type); \
    instantiate_kernel("TanhGPU_nd_" #type_idx "_1", tanh_gg_nd2, type); \
    instantiate_kernel("TanhGPU_nd_" #type_idx "_2", tanh_gg_nd3, type); \
    instantiate_kernel("TanhGPU_nd_" #type_idx "_3", tanh_gg, type);

INSTANTIATE_TANH_ND(0, float)
INSTANTIATE_TANH_ND(1, half)


// Preserve IEEE special-value behavior for inverse trigonometry under fast math.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

// Arcsin: 1D stride-based kernel.
template <typename T>
[[kernel]] void arcsin_gg_nd1(
    device T* dst [[buffer(0)]],
    device const T* src [[buffer(1)]],
    constant const size_m* dst_strides [[buffer(2)]],
    constant const size_m* src_strides [[buffer(3)]],
    constant const uint& ndim [[buffer(4)]],
    constant const size_m* shape [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    size_t dst_idx = size_t(index) * dst_strides[0];
    size_t src_idx = size_t(index) * src_strides[0];
    dst[dst_idx] = T(precise::asin(float(src[src_idx])));
}

// Arcsin: 2D stride-based kernel.
template <typename T>
[[kernel]] void arcsin_gg_nd2(
    device T* dst [[buffer(0)]],
    device const T* src [[buffer(1)]],
    constant const size_m* dst_strides [[buffer(2)]],
    constant const size_m* src_strides [[buffer(3)]],
    constant const uint& ndim [[buffer(4)]],
    constant const size_m* shape [[buffer(5)]],
    uint2 index [[thread_position_in_grid]]) {
    size_t dst_idx = size_t(index.x) * dst_strides[1] + size_t(index.y) * dst_strides[0];
    size_t src_idx = size_t(index.x) * src_strides[1] + size_t(index.y) * src_strides[0];
    dst[dst_idx] = T(precise::asin(float(src[src_idx])));
}

// Arcsin: 3D stride-based kernel.
template <typename T>
[[kernel]] void arcsin_gg_nd3(
    device T* dst [[buffer(0)]],
    device const T* src [[buffer(1)]],
    constant const size_m* dst_strides [[buffer(2)]],
    constant const size_m* src_strides [[buffer(3)]],
    constant const uint& ndim [[buffer(4)]],
    constant const size_m* shape [[buffer(5)]],
    uint3 index [[thread_position_in_grid]]) {
    size_t dst_idx = size_t(index.x) * dst_strides[2] + size_t(index.y) * dst_strides[1] + size_t(index.z) * dst_strides[0];
    size_t src_idx = size_t(index.x) * src_strides[2] + size_t(index.y) * src_strides[1] + size_t(index.z) * src_strides[0];
    dst[dst_idx] = T(precise::asin(float(src[src_idx])));
}

// Arcsin: 4D and higher stride-based kernel.
template <typename T>
[[kernel]] void arcsin_gg(
    device T* dst [[buffer(0)]],
    device const T* src [[buffer(1)]],
    constant const size_m* dst_strides [[buffer(2)]],
    constant const size_m* src_strides [[buffer(3)]],
    constant const uint& ndim [[buffer(4)]],
    constant const size_m* shape [[buffer(5)]],
    uint3 index [[thread_position_in_grid]]) {
    size_t dst_idx = size_t(index.x) * dst_strides[ndim - 1] + size_t(index.y) * dst_strides[ndim - 2];
    size_t src_idx = size_t(index.x) * src_strides[ndim - 1] + size_t(index.y) * src_strides[ndim - 2];
    uint rem = index.z;
    for (int i = int(ndim) - 3; i >= 0; --i) {
        uint coord = rem % shape[i];
        rem /= shape[i];
        dst_idx += size_t(coord) * dst_strides[i];
        src_idx += size_t(coord) * src_strides[i];
    }
    dst[dst_idx] = T(precise::asin(float(src[src_idx])));
}

#define INSTANTIATE_ARCSIN_ND(type_idx, type) \
    instantiate_kernel("ArcsinGPU_nd_" #type_idx "_0", arcsin_gg_nd1, type); \
    instantiate_kernel("ArcsinGPU_nd_" #type_idx "_1", arcsin_gg_nd2, type); \
    instantiate_kernel("ArcsinGPU_nd_" #type_idx "_2", arcsin_gg_nd3, type); \
    instantiate_kernel("ArcsinGPU_nd_" #type_idx "_3", arcsin_gg, type);

INSTANTIATE_ARCSIN_ND(0, float)

// Arctan2: 1D stride-based kernel.
template <typename T>
[[kernel]] void arctan2_gg_nd1(
    device T* dst [[buffer(0)]],
    device const T* a [[buffer(1)]],
    device const T* b [[buffer(2)]],
    constant const size_m* dst_strides [[buffer(3)]],
    constant const size_m* a_strides [[buffer(4)]],
    constant const size_m* b_strides [[buffer(5)]],
    constant const uint& ndim [[buffer(6)]],
    constant const size_m* shape [[buffer(7)]],
    uint index [[thread_position_in_grid]]) {
    size_t dst_idx = size_t(index) * dst_strides[0];
    size_t a_idx = size_t(index) * a_strides[0];
    size_t b_idx = size_t(index) * b_strides[0];
    dst[dst_idx] = T(precise::atan2(float(a[a_idx]), float(b[b_idx])));
}

// Arctan2: 2D stride-based kernel.
template <typename T>
[[kernel]] void arctan2_gg_nd2(
    device T* dst [[buffer(0)]],
    device const T* a [[buffer(1)]],
    device const T* b [[buffer(2)]],
    constant const size_m* dst_strides [[buffer(3)]],
    constant const size_m* a_strides [[buffer(4)]],
    constant const size_m* b_strides [[buffer(5)]],
    constant const uint& ndim [[buffer(6)]],
    constant const size_m* shape [[buffer(7)]],
    uint2 index [[thread_position_in_grid]]) {
    size_t dst_idx = size_t(index.x) * dst_strides[1] + size_t(index.y) * dst_strides[0];
    size_t a_idx = size_t(index.x) * a_strides[1] + size_t(index.y) * a_strides[0];
    size_t b_idx = size_t(index.x) * b_strides[1] + size_t(index.y) * b_strides[0];
    dst[dst_idx] = T(precise::atan2(float(a[a_idx]), float(b[b_idx])));
}

// Arctan2: 3D stride-based kernel.
template <typename T>
[[kernel]] void arctan2_gg_nd3(
    device T* dst [[buffer(0)]],
    device const T* a [[buffer(1)]],
    device const T* b [[buffer(2)]],
    constant const size_m* dst_strides [[buffer(3)]],
    constant const size_m* a_strides [[buffer(4)]],
    constant const size_m* b_strides [[buffer(5)]],
    constant const uint& ndim [[buffer(6)]],
    constant const size_m* shape [[buffer(7)]],
    uint3 index [[thread_position_in_grid]]) {
    size_t dst_idx = size_t(index.x) * dst_strides[2] + size_t(index.y) * dst_strides[1] + size_t(index.z) * dst_strides[0];
    size_t a_idx = size_t(index.x) * a_strides[2] + size_t(index.y) * a_strides[1] + size_t(index.z) * a_strides[0];
    size_t b_idx = size_t(index.x) * b_strides[2] + size_t(index.y) * b_strides[1] + size_t(index.z) * b_strides[0];
    dst[dst_idx] = T(precise::atan2(float(a[a_idx]), float(b[b_idx])));
}

// Arctan2: 4D and higher stride-based kernel.
template <typename T>
[[kernel]] void arctan2_gg(
    device T* dst [[buffer(0)]],
    device const T* a [[buffer(1)]],
    device const T* b [[buffer(2)]],
    constant const size_m* dst_strides [[buffer(3)]],
    constant const size_m* a_strides [[buffer(4)]],
    constant const size_m* b_strides [[buffer(5)]],
    constant const uint& ndim [[buffer(6)]],
    constant const size_m* shape [[buffer(7)]],
    uint3 index [[thread_position_in_grid]]) {
    size_t dst_idx = size_t(index.x) * dst_strides[ndim - 1] + size_t(index.y) * dst_strides[ndim - 2];
    size_t a_idx = size_t(index.x) * a_strides[ndim - 1] + size_t(index.y) * a_strides[ndim - 2];
    size_t b_idx = size_t(index.x) * b_strides[ndim - 1] + size_t(index.y) * b_strides[ndim - 2];
    uint rem = index.z;
    for (int i = int(ndim) - 3; i >= 0; --i) {
        uint coord = rem % shape[i];
        rem /= shape[i];
        dst_idx += size_t(coord) * dst_strides[i];
        a_idx += size_t(coord) * a_strides[i];
        b_idx += size_t(coord) * b_strides[i];
    }
    dst[dst_idx] = T(precise::atan2(float(a[a_idx]), float(b[b_idx])));
}

#define INSTANTIATE_ARCTAN2_ND(type_idx, type) \
    instantiate_kernel("Arctan2GPU_nd_" #type_idx "_0", arctan2_gg_nd1, type); \
    instantiate_kernel("Arctan2GPU_nd_" #type_idx "_1", arctan2_gg_nd2, type); \
    instantiate_kernel("Arctan2GPU_nd_" #type_idx "_2", arctan2_gg_nd3, type); \
    instantiate_kernel("Arctan2GPU_nd_" #type_idx "_3", arctan2_gg, type);

INSTANTIATE_ARCTAN2_ND(0, float)
