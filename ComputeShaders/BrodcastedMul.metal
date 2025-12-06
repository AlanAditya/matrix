//
//  BrodcastedMul.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 21/10/25.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;


template <typename T>
kernel void BrodcastedMulGPU(device T* result [[buffer(0)]], device const T* A [[buffer(1)]], device const T* B [[buffer(2)]], constant size_m* strideR [[buffer(3)]], constant size_m* strideA [[buffer(4)]], constant size_m* strideB [[buffer(5)]], constant int& dims [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
    // gid is based on result Buffer
    // Convert GID into respective axis index;
    
    // globalIndex for A and B in the memory where the matrix is in a flattened state
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
    result[gid] =  A[GindexA] * B[GindexB];
    
}

instantiate_kernel("BrodcastedMulGPU_0", BrodcastedMulGPU, float);
instantiate_kernel("BrodcastedMulGPU_1", BrodcastedMulGPU, half);
instantiate_kernel("BrodcastedMulGPU_2", BrodcastedMulGPU, uint8_t);
instantiate_kernel("BrodcastedMulGPU_3", BrodcastedMulGPU, int);


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
//result[gid] = A[indicies dot strideA] * B[indicies dot strideB];
