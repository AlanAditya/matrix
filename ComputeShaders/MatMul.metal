//
//  MatMul.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 14/07/25.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;


template <typename T>
T Dot(device const T* A, device const T* B, size_t size) {
    T accumulate = 0;
    for (size_t i = 0; i < size; i++) {
        accumulate += A[i] * B[i];
    }
    return accumulate;
}

// In A * B => dot(rows of A, col of B) --> Matmul(A, B.T) --> dot(rows of A, cols now become rows of B)
// This is done as our matrices our row major matrices so doting rows is more efficient
template <typename T>
kernel void MatMul(device T* out_buffer [[buffer(0)]], device const T* A [[buffer(1)]], device const T* B [[buffer(2)]], constant size_m* shape1 [[buffer(3)]], constant size_m* shape2 [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    // Orignal: [m, k] * [k, n] => [m, n]
    // Transposed:  [m, k] * [n, k] => [m, n]
    // shape 2 => [k, n]
    // shape 1 => [m, k]
    
    size_t m = shape1[0];
    size_t k = shape1[1]; // Also shape2[0]
    size_t n = shape2[1];
    
    int i = gid / n;
    int j = gid % n;
    out_buffer[gid] = Dot(A + i * k, B + j * k, k);
}

instantiate_kernel("MatMulGPU_0", MatMul, float);
instantiate_kernel("MatMulGPU_1", MatMul, half);
instantiate_kernel("MatMulGPU_3", MatMul, int);


// Specialised kernel for 3D tensors using explicit strides
template <typename T>
kernel void BatchedMatMul_3Dgg(device T* out_buffer [[buffer(0)]], 
                               device const T* A [[buffer(1)]], 
                               device const T* B [[buffer(2)]], 
                               constant size_m* stridesA [[buffer(3)]],
                               constant size_m* stridesB [[buffer(4)]],
                               constant size_m* stridesC [[buffer(5)]],
                               constant size_m& k [[buffer(6)]],
                               uint3 gid [[thread_position_in_grid]]) {
    
    // gid maps to: z -> batch, y -> row, x -> col for out_shape of [Batch, M, N]
    out_buffer[(gid.z * stridesC[0]) + (gid.y * stridesC[1]) + (gid.x * stridesC[2])] = Dot(A + (gid.z * stridesA[0]) + (gid.y * stridesA[1]), B + (gid.z * stridesB[0]) + (gid.x * stridesB[1]), k);
}

template <typename T>
kernel void BatchedMatMul_NDgg(device T* out_buffer [[buffer(0)]],
                               device const T* A [[buffer(1)]],
                               device const T* B [[buffer(2)]],
                               constant size_m* stridesA [[buffer(3)]],
                               constant size_m* stridesB [[buffer(4)]],
                               constant size_m* stridesR [[buffer(5)]],
                               constant size_m* result_shape [[buffer(6)]],
                               constant size_m& k [[buffer(7)]],
                               constant int& ndims [[buffer(8)]],
                               uint3 gid [[thread_position_in_grid]]) {
    
    // gid is general and not based on result Buffer
    // Convert GID into respective axis index;
    // Adopting a fast approach as x is the innermost dimension as it changes the fastest and metal threads are optomised for fast changing x. Plus it avoids 2 mudulo operations which take ~ 60 cycles
    
    // globalIndex for A, B and R
    size_t GindexA = gid.y * (size_t)stridesA[ndims-2];
    size_t GindexB = gid.x * (size_t)stridesB[ndims-2];
    size_t indexR  = gid.x * (size_t)stridesR[ndims-1] + gid.y * (size_t)stridesR[ndims-2];
    
    
    size_t rem = gid.z;
    for (int i = ndims-3; i >= 0; --i) {
        size_t dim_index = rem % result_shape[i];
        GindexA += dim_index * stridesA[i];
        GindexB += dim_index * stridesB[i];
        indexR  += dim_index * stridesR[i];
        rem /= result_shape[i];
    }
    
    // gid maps to: z -> batch, y -> row, x -> col for out_shape of [Batch, M, N]
    out_buffer[indexR] = Dot(A + GindexA, B + GindexB, k);
}

instantiate_kernel("BatchedMatMul_3Dgg_0", BatchedMatMul_3Dgg, float);
instantiate_kernel("BatchedMatMul_3Dgg_1", BatchedMatMul_3Dgg, half);
instantiate_kernel("BatchedMatMul_3Dgg_3", BatchedMatMul_3Dgg, int);

instantiate_kernel("BatchedMatMul_NDgg_0", BatchedMatMul_NDgg, float);
instantiate_kernel("BatchedMatMul_NDgg_1", BatchedMatMul_NDgg, half);
instantiate_kernel("BatchedMatMul_NDgg_3", BatchedMatMul_NDgg, int);
