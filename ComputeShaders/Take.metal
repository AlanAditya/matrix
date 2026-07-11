#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

// 1D Take
template <typename T>
[[kernel]] void take_gg_nd1(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], device const int* indices [[buffer(2)]], constant const size_m& dst_stride [[buffer(3)]], constant const size_m& eff_src_stride [[buffer(4)]], constant const size_m& eff_idx_stride [[buffer(5)]], constant const size_m& src_axis_stride [[buffer(6)]], uint index [[thread_position_in_grid]]) {
    size_m dst_idx = index * dst_stride;
    size_m idx_idx = index * eff_idx_stride;
    int idx_val = indices[idx_idx];
    size_m src_idx = index * eff_src_stride + (idx_val * src_axis_stride);
    dst[dst_idx] = src[src_idx];
}

// 2D Take
template <typename T>
[[kernel]] void take_gg_nd2(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], device const int* indices [[buffer(2)]], constant const size_m* dst_strides [[buffer(3)]], constant const size_m* eff_src_strides [[buffer(4)]], constant const size_m* eff_idx_strides [[buffer(5)]], constant const size_m& src_axis_stride [[buffer(6)]], uint2 index [[thread_position_in_grid]]) {
    size_m dst_idx = index.y * dst_strides[0] + index.x * dst_strides[1];
    size_m idx_idx = index.y * eff_idx_strides[0] + index.x * eff_idx_strides[1];
    int idx_val = indices[idx_idx];
    size_m src_idx = index.y * eff_src_strides[0] + index.x * eff_src_strides[1] + (idx_val * src_axis_stride);
    dst[dst_idx] = src[src_idx];
}

// 3D Take
template <typename T>
[[kernel]] void take_gg_nd3(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], device const int* indices [[buffer(2)]], constant const size_m* dst_strides [[buffer(3)]], constant const size_m* eff_src_strides [[buffer(4)]], constant const size_m* eff_idx_strides [[buffer(5)]], constant const size_m& src_axis_stride [[buffer(6)]], uint3 index [[thread_position_in_grid]]) {
    size_m dst_idx = index.z * dst_strides[0] + index.y * dst_strides[1] + index.x * dst_strides[2];
    size_m idx_idx = index.z * eff_idx_strides[0] + index.y * eff_idx_strides[1] + index.x * eff_idx_strides[2];
    int idx_val = indices[idx_idx];
    size_m src_idx = index.z * eff_src_strides[0] + index.y * eff_src_strides[1] + index.x * eff_src_strides[2] + (idx_val * src_axis_stride);
    dst[dst_idx] = src[src_idx];
}

// ND Take
template <typename T>
[[kernel]] void take_gg(device T* dst [[buffer(0)]], device const T* src [[buffer(1)]], device const int* indices [[buffer(2)]], constant const size_m* dst_strides [[buffer(3)]], constant const size_m* eff_src_strides [[buffer(4)]], constant const size_m* eff_idx_strides [[buffer(5)]], constant const size_m& src_axis_stride [[buffer(6)]], constant const size_m* out_shape [[buffer(7)]], constant const int& ndim [[buffer(8)]], uint3 index [[thread_position_in_grid]]) {
    size_m dst_idx = index.x * dst_strides[ndim-1] + index.y * dst_strides[ndim-2];
    size_m idx_idx = index.x * eff_idx_strides[ndim-1] + index.y * eff_idx_strides[ndim-2];
    size_m src_idx = index.x * eff_src_strides[ndim-1] + index.y * eff_src_strides[ndim-2];
    
    uint remA = index.z;
    for (int i = ndim-3; i >= 0; --i) {
        size_m mod = remA % out_shape[i];
        dst_idx += mod * dst_strides[i];
        idx_idx += mod * eff_idx_strides[i];
        src_idx += mod * eff_src_strides[i];
        remA /= out_shape[i];
    }
    
    int idx_val = indices[idx_idx];
    src_idx += (idx_val * src_axis_stride);
    
    dst[dst_idx] = src[src_idx];
}

#define INSTANTIATE_TAKE_ND(type_idx, type) \
    instantiate_kernel("TakeGPU_nd_" #type_idx "_0", take_gg_nd1, type); \
    instantiate_kernel("TakeGPU_nd_" #type_idx "_1", take_gg_nd2, type); \
    instantiate_kernel("TakeGPU_nd_" #type_idx "_2", take_gg_nd3, type); \
    instantiate_kernel("TakeGPU_nd_" #type_idx "_3", take_gg, type);

INSTANTIATE_TAKE_ND(0, float)
INSTANTIATE_TAKE_ND(1, half)
