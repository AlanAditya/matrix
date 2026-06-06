//
//  Copy.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 26/01/26.
//

#include <metal_stdlib>
#include "Utils.h"


using namespace metal;

// 1D Optimized Copy
template <typename T, typename U>
[[kernel]] void copy_gg_nd1(device  T* dst [[buffer(0)]], device const U* src [[buffer(1)]], constant const size_m& dst_stride [[buffer(2)]], constant const size_m& src_stride [[buffer(3)]], constant const int& offset [[buffer(4)]], uint index [[thread_position_in_grid]]) {
    auto src_idx = index * src_stride;
    auto dst_idx = index * dst_stride;
    dst[dst_idx + offset] = static_cast<T>(src[src_idx]);
}

// 2D Optimized Copy
template <typename T, typename U>
[[kernel]] void copy_gg_nd2(device  T* dst [[buffer(0)]], device const U* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const int& offset [[buffer(4)]], uint2 index [[thread_position_in_grid]]) {
    auto src_idx = index.x * src_strides[1] + index.y * src_strides[0];
    auto dst_idx = index.x * dst_strides[1] + index.y * dst_strides[0];
    dst[dst_idx + offset] = static_cast<T>(src[src_idx]);
}

// 3D Optimized Copy
template <typename T, typename U, typename IdxT = size_m>
[[kernel]] void copy_gg_nd3(device  T* dst [[buffer(0)]], const device U* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const int& offset [[buffer(4)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[2] + index.y * src_strides[1] + index.z * src_strides[0];
    size_m dst_idx = index.x * dst_strides[2] + index.y * dst_strides[1] + index.z * dst_strides[0];
    dst[dst_idx + offset] = static_cast<T>(src[src_idx]);
}

// ND Optimized Copy
template <typename T, typename U, int N = 1, typename IdxT = size_m>
[[kernel]] void copy_gg(device  T* dst [[buffer(0)]], device const U* src [[buffer(1)]], constant const size_m* dst_strides [[buffer(2)]], constant const size_m* src_strides [[buffer(3)]], constant const int& offset [[buffer(4)]], constant const size_m* src_shape [[buffer(5)]], constant const int& ndim [[buffer(6)]], uint3 index [[thread_position_in_grid]]) {
    size_m src_idx = index.x * src_strides[ndim-1] + index.y * src_strides[ndim-2];
    size_m dst_idx = index.x * dst_strides[ndim-1] + index.y * dst_strides[ndim-2];
    uint remA = index.z;
    for (int i = ndim-3; i >= 0; --i) {
        src_idx += remA % src_shape[i] * src_strides[i];
        dst_idx += remA % src_shape[i] * dst_strides[i];
        remA /= src_shape[i];
    }
    dst[dst_idx + offset] = static_cast<T>(src[src_idx]);
}

#define INSTANTIATE_FROM_TYPE(dst_idx, src_idx, src_type, dst_type) \
    instantiate_kernel("CopyInplaceGPU_" #dst_idx "_" #src_idx "_0", copy_gg_nd1, dst_type, src_type); \
    instantiate_kernel("CopyInplaceGPU_" #dst_idx "_" #src_idx "_1", copy_gg_nd2, dst_type, src_type); \
    instantiate_kernel("CopyInplaceGPU_" #dst_idx "_" #src_idx "_2", copy_gg_nd3, dst_type, src_type); \
    instantiate_kernel("CopyInplaceGPU_" #dst_idx "_" #src_idx "_3", copy_gg,     dst_type, src_type); \

//INSTANTIATE_FROM_TYPE(0, float, float);
//INSTANTIATE_FROM_TYPE(1, half, half);
//INSTANTIATE_FROM_TYPE(2, uint8_t, uint8_t);
//INSTANTIATE_FROM_TYPE(3, int, int);

#define INSTANTIATE_ALL_SRC(dst_idx, dst_type)          \
    INSTANTIATE_FROM_TYPE(dst_idx, 0, float,    dst_type) \
    INSTANTIATE_FROM_TYPE(dst_idx, 1, half,     dst_type) \
    INSTANTIATE_FROM_TYPE(dst_idx, 2, uint8_t,  dst_type) \
    INSTANTIATE_FROM_TYPE(dst_idx, 3, int,      dst_type) \
    INSTANTIATE_FROM_TYPE(dst_idx, 4, uint16_t, dst_type) \
    INSTANTIATE_FROM_TYPE(dst_idx, 5, uint32_t, dst_type)

// Call once per destination type
INSTANTIATE_ALL_SRC(0, float)
INSTANTIATE_ALL_SRC(1, half)
INSTANTIATE_ALL_SRC(2, uint8_t)
INSTANTIATE_ALL_SRC(3, int)
INSTANTIATE_ALL_SRC(4, uint16_t)
INSTANTIATE_ALL_SRC(5, uint32_t)
