//
//  Transpose.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 13/07/25.
//

#include <metal_stdlib>
#include "Utils.h"


using namespace metal;

template <typename T>
kernel void TransposeGPU(device T* out_buffer [[buffer(0)]], device const T* in_buffer [[buffer(1)]], constant size_m* inputStrides [[buffer(2)]], constant size_m* outputStrides [[buffer(3)]], constant int& dims [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    size_t index = 0;
    uint remainder = gid;
    for (int i = 0; i < dims; i++) {
        index += (remainder / inputStrides[i]) * outputStrides[i];
        remainder %= inputStrides[i];
    }
    out_buffer[index] = in_buffer[gid];
}

instantiate_kernel("TransposeGPU_0", TransposeGPU, float);
instantiate_kernel("TransposeGPU_1", TransposeGPU, half);
instantiate_kernel("TransposeGPU_2", TransposeGPU, uint8_t);
instantiate_kernel("TransposeGPU_3", TransposeGPU, int);


