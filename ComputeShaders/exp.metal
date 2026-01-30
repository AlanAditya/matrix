//
//  exp.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 05/11/25.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

template <typename T>
kernel void ExpGPU(device T* outMat [[buffer(0)]], device const T* inMat[[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    outMat[gid] = exp(inMat[gid]);
}

instantiate_kernel("ExpGPU_0", ExpGPU, float);


