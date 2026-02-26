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
