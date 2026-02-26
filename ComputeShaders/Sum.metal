//
//  Sum.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 25/07/25.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

template <typename T>
kernel void SumGPU(device T* out_buffer [[buffer(0)]], device const T* in_buffer [[buffer(1)]], constant size_m& AxisStride [[buffer(2)]], constant size_m& ElStride [[buffer(3)]],constant size_m& noOfOpp [[buffer(4)]], constant size_m* inputStrides [[buffer(5)]], constant size_m* maskedStrides [[buffer(6)]], constant size_m& dims [[buffer(7)]], uint gid [[thread_position_in_grid]]) {
    size_t AxisOffset = 0;
    size_t remaining = gid;
    for (size_t i = 0; i < dims; i++) {
        AxisOffset += (remaining / inputStrides[i]) * maskedStrides[i];
        remaining %= inputStrides[i];
    }
    for (size_t i = 0; i < noOfOpp; i++) {
        out_buffer[gid] += in_buffer[AxisOffset + i * ElStride];
    }
}

instantiate_kernel("SumGPU_0", SumGPU, float);
instantiate_kernel("SumGPU_1", SumGPU, half);
instantiate_kernel("SumGPU_2", SumGPU, uint8_t);
instantiate_kernel("SumGPU_3", SumGPU, int);
