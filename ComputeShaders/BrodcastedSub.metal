//
//  BrodcastedSub.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 21/10/25.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

template <typename T>
kernel void BrodcastedSubGPU_1Dgg(device T* result [[buffer(0)]], device const T* A [[buffer(1)]], device const T* B [[buffer(2)]], constant size_m& strideR [[buffer(3)]], constant size_m& strideA [[buffer(4)]], constant size_m& strideB [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    // gid is general and not based on result Buffer
    // Convert GID into respective axis index;
    
    // globalIndex for A, B and R
    size_t GindexA = gid * (size_t)strideA;
    size_t GindexB = gid * (size_t)strideB;
    size_t indexR =  gid * (size_t)strideR;
    
    result[indexR] = A[GindexA] - B[GindexB];
    
}

template <typename T>
kernel void BrodcastedSubGPU_2Dgg(device T* result [[buffer(0)]], device const T* A [[buffer(1)]], device const T* B [[buffer(2)]], constant size_m* strideR [[buffer(3)]], constant size_m* strideA [[buffer(4)]], constant size_m* strideB [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    // gid is general and not based on result Buffer
    // Convert GID into respective axis index;
    
    // globalIndex for A, B and R
    size_t GindexA = gid.x * (size_t)strideA[1] + gid.y * (size_t)strideA[0];
    size_t GindexB = gid.x * (size_t)strideB[1] + gid.y * (size_t)strideB[0];
    size_t indexR =  gid.x * (size_t)strideR[1] + gid.y * (size_t)strideR[0];
    
    result[indexR] = A[GindexA] - B[GindexB];
    
}

template <typename T>
kernel void BrodcastedSubGPU_3Dgg(device T* result [[buffer(0)]], device const T* A [[buffer(1)]], device const T* B [[buffer(2)]], constant size_m* strideR [[buffer(3)]], constant size_m* strideA [[buffer(4)]], constant size_m* strideB [[buffer(5)]], uint3 gid [[thread_position_in_grid]]) {
    // gid is general and not based on result Buffer
    // Convert GID into respective axis index;
    
    // globalIndex for A, B and R
    size_t GindexA = gid.x * (size_t)strideA[2] + gid.y * (size_t)strideA[1] + gid.z * (size_t)strideA[0];
    size_t GindexB = gid.x * (size_t)strideB[2] + gid.y * (size_t)strideB[1] + gid.z * (size_t)strideB[0];
    size_t indexR  = gid.x * (size_t)strideR[2] + gid.y * (size_t)strideR[1] + gid.z * (size_t)strideR[0];
    
    result[indexR] = A[GindexA] - B[GindexB];
    
}

// General General N Dimensional Brodcasted Sub where result buffer as well as any other buffer can be non contiguous in memory
template <typename T>
kernel void BrodcastedSubGPU_NDgg(device T* result [[buffer(0)]], device const T* A [[buffer(1)]], device const T* B [[buffer(2)]], constant size_m* strideR [[buffer(3)]], constant size_m* strideA [[buffer(4)]], constant size_m* strideB [[buffer(5)]], constant const size_m* result_shape [[buffer(6)]], constant int& ndims [[buffer(7)]], uint3 gid [[thread_position_in_grid]]) {
    // gid is general and not based on result Buffer
    // Convert GID into respective axis index;
    // Adopting a fast approach as x is the innermost dimension as it changes the fastest and metal threads are optomised for fast changing x. Plus it avoids 2 mudulo operations which take ~ 60 cycles
    
    // globalIndex for A, B and R
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
    
    result[indexR] = A[GindexA] - B[GindexB];
}

#define INSTANTIATE_FROM_TYPE(src_idx, type) \
    instantiate_kernel("BrodcastedSubGPU_" #src_idx "_0", BrodcastedSubGPU_1Dgg, type); \
    instantiate_kernel("BrodcastedSubGPU_" #src_idx "_1", BrodcastedSubGPU_2Dgg, type); \
    instantiate_kernel("BrodcastedSubGPU_" #src_idx "_2", BrodcastedSubGPU_3Dgg, type); \
    instantiate_kernel("BrodcastedSubGPU_" #src_idx "_3", BrodcastedSubGPU_NDgg, type); \

INSTANTIATE_FROM_TYPE(0, float);
INSTANTIATE_FROM_TYPE(1, half);
INSTANTIATE_FROM_TYPE(2, uint8_t);
INSTANTIATE_FROM_TYPE(3, int);

// Not functional yet
// General Brodcasted Sub ND version where gid is based on result buffer directly as result buffer is contiguous in memory
template <typename T>
kernel void BrodcastedSubGPU_NDg(device T* result [[buffer(0)]], device const T* A [[buffer(1)]], device const T* B [[buffer(2)]], constant size_m* strideR [[buffer(3)]], constant size_m* strideA [[buffer(4)]], constant size_m* strideB [[buffer(5)]], constant int& dims [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    // gid is based on result Buffer
    // Convert GID into respective axis index;
    
    // globalIndex for A and B
    size_t GindexA = 0;
    size_t GindexB = 0;
    
    // axis Index for Result like temp storage for i, j, k, l ... along each axis for result
    size_t indexR = 0;
    
    
    
    int rem = gid;
    for (int i = 0; i < dims; i++) {
        indexR = rem / strideR[i];
        GindexA += indexR * strideA[i];
        GindexB += indexR * strideB[i];
        rem %= strideR[i];
    }
    
    result[gid] = A[GindexA] - B[GindexB];
    
}

instantiate_kernel("BrodcastedSubGPU_0", BrodcastedSubGPU_NDg, float);
instantiate_kernel("BrodcastedSubGPU_1", BrodcastedSubGPU_NDg, half);
instantiate_kernel("BrodcastedSubGPU_2", BrodcastedSubGPU_NDg, uint8_t);
instantiate_kernel("BrodcastedSubGPU_3", BrodcastedSubGPU_NDg, int);


// SUDO CODE

// gid is based on result Buffer
// Convert GID into respective axis index;
//size_m* indicies = new size_m[dims];
//
//int rem = gid;
//for (int i = 0; i < dims; i++) {
//    indicies[i] = rem / strideR[i];
//    rem %= strideR[i];
//}
//
//result[gid] = A[indicies dot strideA] + B[indicies dot strideB];



