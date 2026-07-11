#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

template <typename T>
kernel void BrodcastedMaxGPU_1Dgg(device T* result [[buffer(0)]], device const T* A [[buffer(1)]], device const T* B [[buffer(2)]], constant size_m& strideR [[buffer(3)]], constant size_m& strideA [[buffer(4)]], constant size_m& strideB [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    size_t GindexA = gid * (size_t)strideA;
    size_t GindexB = gid * (size_t)strideB;
    size_t indexR =  gid * (size_t)strideR;
    result[indexR] = max(A[GindexA], B[GindexB]);
}

template <typename T>
kernel void BrodcastedMaxGPU_2Dgg(device T* result [[buffer(0)]], device const T* A [[buffer(1)]], device const T* B [[buffer(2)]], constant size_m* strideR [[buffer(3)]], constant size_m* strideA [[buffer(4)]], constant size_m* strideB [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    size_t GindexA = gid.x * (size_t)strideA[1] + gid.y * (size_t)strideA[0];
    size_t GindexB = gid.x * (size_t)strideB[1] + gid.y * (size_t)strideB[0];
    size_t indexR =  gid.x * (size_t)strideR[1] + gid.y * (size_t)strideR[0];
    result[indexR] = max(A[GindexA], B[GindexB]);
}

template <typename T>
kernel void BrodcastedMaxGPU_3Dgg(device T* result [[buffer(0)]], device const T* A [[buffer(1)]], device const T* B [[buffer(2)]], constant size_m* strideR [[buffer(3)]], constant size_m* strideA [[buffer(4)]], constant size_m* strideB [[buffer(5)]], uint3 gid [[thread_position_in_grid]]) {
    size_t GindexA = gid.x * (size_t)strideA[2] + gid.y * (size_t)strideA[1] + gid.z * (size_t)strideA[0];
    size_t GindexB = gid.x * (size_t)strideB[2] + gid.y * (size_t)strideB[1] + gid.z * (size_t)strideB[0];
    size_t indexR  = gid.x * (size_t)strideR[2] + gid.y * (size_t)strideR[1] + gid.z * (size_t)strideR[0];
    result[indexR] = max(A[GindexA], B[GindexB]);
}

template <typename T>
kernel void BrodcastedMaxGPU_NDgg(device T* result [[buffer(0)]], device const T* A [[buffer(1)]], device const T* B [[buffer(2)]], constant size_m* strideR [[buffer(3)]], constant size_m* strideA [[buffer(4)]], constant size_m* strideB [[buffer(5)]], constant const size_m* result_shape [[buffer(6)]], constant int& ndims [[buffer(7)]], uint3 gid [[thread_position_in_grid]]) {
    size_t GindexA = gid.x * (size_t)strideA[ndims-1] + gid.y * (size_t)strideA[ndims-2];
    size_t GindexB = gid.x * (size_t)strideB[ndims-1] + gid.y * (size_t)strideB[ndims-2];
    size_t indexR  = gid.x * (size_t)strideR[ndims-1] + gid.y * (size_t)strideR[ndims-2];
    
    size_t rem = gid.z;
    for (int i = ndims-3; i >= 0; --i) {
        size_t dim_index = rem % result_shape[i];
        GindexA += dim_index * strideA[i];
        GindexB += dim_index * strideB[i];
        indexR  += dim_index * strideR[i];
        rem /= result_shape[i];
    }
    result[indexR] = max(A[GindexA], B[GindexB]);
}

#define INSTANTIATE_FROM_TYPE_MAX(src_idx, type) \
    instantiate_kernel("BrodcastedMaxGPU_" #src_idx "_0", BrodcastedMaxGPU_1Dgg, type); \
    instantiate_kernel("BrodcastedMaxGPU_" #src_idx "_1", BrodcastedMaxGPU_2Dgg, type); \
    instantiate_kernel("BrodcastedMaxGPU_" #src_idx "_2", BrodcastedMaxGPU_3Dgg, type); \
    instantiate_kernel("BrodcastedMaxGPU_" #src_idx "_3", BrodcastedMaxGPU_NDgg, type);

INSTANTIATE_FROM_TYPE_MAX(0, float);
INSTANTIATE_FROM_TYPE_MAX(1, half);
INSTANTIATE_FROM_TYPE_MAX(2, uint8_t);
INSTANTIATE_FROM_TYPE_MAX(3, int);
