//
//  sqrt.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 03/11/25.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

template <typename T>
kernel void SqrtGPU(device T* outMat [[buffer(0)]], device const T* inMat[[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    outMat[gid] = sqrt(inMat[gid]);
}

instantiate_kernel("SqrtGPU_0", SqrtGPU, float);
