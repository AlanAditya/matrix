//
//  Trig.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 09/08/25.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

template <typename T>
kernel void SinGPU(device T* outMat [[buffer(0)]], device const T* inMat[[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    outMat[gid] = sin(inMat[gid]);
}

instantiate_kernel("SinGPU_0", SinGPU, float);


template <typename T>
kernel void CosGPU(device T* outMat [[buffer(0)]], device const T* inMat[[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    outMat[gid] = cos(inMat[gid]);
}

instantiate_kernel("CosGPU_0", CosGPU, float);


template <typename T>
kernel void TanGPU(device T* outMat [[buffer(0)]], device const T* inMat[[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    outMat[gid] = tan(inMat[gid]);
}

instantiate_kernel("TanGPU_0", TanGPU, float);

