//
//  Padding.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 14/06/26.
//

#include <metal_stdlib>
#include "Utils.h"
using namespace metal;

// 1D Optimized Copy
template <typename T>
[[kernel]] void padding_gg_nd1(device  T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m& dst_stride [[buffer(2)]], constant const size_m& src_stride [[buffer(3)]], constant const size_m* limits [[buffer(4)]], constant const int& offset [[buffer(5)]], constant const T& value [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    auto dst_idx = index * dst_stride;
    if (limits[0] > index || limits[1] <= index) {
        dst[dst_idx] = value;
    } else {
        auto src_idx = index * src_stride - offset;
        dst[dst_idx] = static_cast<T>(src[src_idx]);
    }
}

// 2D Optimized Copy
template <typename T>
[[kernel]] void padding_gg_nd2(device  T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const size_m* limits [[buffer(4)]], constant const int& offset [[buffer(5)]], constant const T& value [[buffer(6)]], uint2 index [[thread_position_in_grid]]) {
    auto dst_idx = index.x * dst_strides[1] + index.y * dst_strides[0];
    if (limits[0] > index.y || limits[1] <= index.y || limits[2] > index.x || limits[3] <= index.x ) {
        dst[dst_idx] = value;
    } else {
        auto src_idx = (index.x) * src_strides[1] + (index.y) * src_strides[0] - offset;
        dst[dst_idx] = static_cast<T>(src[src_idx]);
    }
}

// 3D Optimized Copy
template <typename T>
[[kernel]] void padding_gg_nd3(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const size_m* limits [[buffer(4)]], constant const int& offset [[buffer(5)]], constant const T& value [[buffer(6)]], uint3 index [[thread_position_in_grid]]) {
    
    auto dst_idx = index.x * dst_strides[2] + index.y * dst_strides[1] + index.z * dst_strides[0];
    if (limits[0] > index.z || limits[1] <= index.z ||
        limits[2] > index.y || limits[3] <= index.y ||
        limits[4] > index.x || limits[5] <= index.x) {
        dst[dst_idx] = value;
    } else {
        auto src_idx = (index.x) * src_strides[2] +
                       (index.y) * src_strides[1] +
                       (index.z) * src_strides[0] - offset;
        dst[dst_idx] = static_cast<T>(src[src_idx]);
    }
}

// ND Optimized Copy
template <typename T, int N = 1, typename IdxT = size_m>
[[kernel]] void padding_gg(device  T* dst [[buffer(0)]], device const T* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const size_m* limits [[buffer(4)]], constant const int& offset [[buffer(5)]], constant const T& value [[buffer(6)]], constant const size_m* dst_shape [[buffer(7)]], constant const int& ndim [[buffer(8)]], uint3 index [[thread_position_in_grid]]) {
    
    size_m src_idx = index.x * src_strides[ndim-1] + index.y * src_strides[ndim-2];
    size_m dst_idx = index.x * dst_strides[ndim-1] + index.y * dst_strides[ndim-2];
    
    bool padding_rgion = false;
    if (limits[2*(ndim-2)] > index.y || limits[2*(ndim-2) + 1] <= index.y ||
        limits[2*(ndim-1)] > index.x || limits[2*(ndim-1) + 1] <= index.x) {
        padding_rgion = true;
    }
    
    uint remA = index.z;
    for (int i = ndim-3; i >= 0; --i) {
        if (limits[2*i] > (remA % dst_shape[i]) || limits[2*i + 1] <= (remA % dst_shape[i])) {
            padding_rgion = true;
        }
        src_idx += (remA % dst_shape[i]) * src_strides[i];
        dst_idx += (remA % dst_shape[i]) * dst_strides[i];
        remA /= dst_shape[i];
    }
    if (padding_rgion) {
        dst[dst_idx] = value;
    } else {
        dst[dst_idx] = static_cast<T>(src[src_idx - offset]);
    }
}

#define INSTANTIATE_TYPE(type_idx, type) \
    instantiate_kernel("PaddingGPU_" #type_idx "_0", padding_gg_nd1, type); \
    instantiate_kernel("PaddingGPU_" #type_idx "_1", padding_gg_nd2, type); \
    instantiate_kernel("PaddingGPU_" #type_idx "_2", padding_gg_nd3, type); \
    instantiate_kernel("PaddingGPU_" #type_idx "_3", padding_gg,     type);

INSTANTIATE_TYPE(0, float)
INSTANTIATE_TYPE(1, half)
INSTANTIATE_TYPE(2, uint8_t)
INSTANTIATE_TYPE(3, int)
INSTANTIATE_TYPE(4, uint16_t)
INSTANTIATE_TYPE(5, uint32_t)
