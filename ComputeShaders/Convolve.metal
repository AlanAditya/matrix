//
//  Convolve.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 15/08/25.
//

#include <metal_stdlib>
#include "Utils.h"


using namespace metal;

// This in Convolve operation that allows it to apply on a group of contiguous axis simultaneously so
// eg shape of an image is [1080, 1920, 4] and i want to conv it with a [3, 3] filter i can now do that by passing the param convDims = 2 now it will exclude from axis = 2 ownwards;
// the gid is based on the total size of the included axis and the element stride is based on the product of shapes of excluded axises
// this way i am treating the last dim of size 4 as a single element
// and later looping over the them and performing opp individually
//template <typename T>
//kernel void ConvolveGPU(device T* outMat [[buffer(0)]], device const T* inMat [[buffer(1)]], device const T* Kernel [[buffer(2)]], constant size_t& kernelSize [[buffer(3)]], constant size_m* shape1 [[buffer(4)]], constant size_m* shape2 [[buffer(5)]], constant size_m* strides [[buffer(6)]], constant size_m& elementStride [[buffer(7)]], constant size_m& convDims [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
//    int offset = 0;
//    int kernelIndex = 0;
//    int gridDimIndex = 0;
//    int COMIndex = 0;
//    bool outOfBounds = false;
//    for (int i = 0; i < kernelSize; i++) {
//        offset = 0;
//        uint rem = gid;
//        uint remKernel = i;
//        for (int j = convDims-1; j >= 0; j--) {
//            kernelIndex = (remKernel % shape2[j]) - (shape2[j] / 2);
//            gridDimIndex = (rem % shape1[j]) + kernelIndex;
////            if (i == 0) {
////                COMIndex += (rem % shape1[j]) * strides[j];
////            }
//            if (gridDimIndex < 0 || gridDimIndex >= shape1[j]) {
//                outOfBounds = true;
//                break;
//            }
//            offset += strides[j] * (gridDimIndex);
////            if (gid == 1) {
////                metal::os_log_default.log_info("i = %d, convolve Dims = %d, j = %d, kernInd = %d, gridInd = %d, gridIndKern = %d, offset = %d, stride = %u", i, convDims, j, kernelIndex, rem % shape1[j] ,gridDimIndex, offset, strides[j]);
////            }
//            rem /= shape1[j];
//            remKernel /= shape2[j];
//            
//        }
//        if (outOfBounds) {outOfBounds = false; continue;}
////        if (gid == 1) {
////            metal::os_log_default.log_info("i = %d", i);
////        }
//        for (uint l = 0; l < elementStride; l++) {
////            if (gid == 1) {
////                metal::os_log_default.log_info("l = %u, inMatIndex= %d", l, (offset));
////            }
////            outMat[gid * elementStride + l] += inMat[(offset) * elementStride + l] * Kernel[i]; // upgrade now no need for element stride as offset during its creation is now mulptiplied by the actual strides of the full stack rather than the convolve stack allowing for processing of non-contiguous buffers
//            outMat[gid * elementStride + l] += inMat[(offset) + l] * Kernel[i];
//        }
//    }
//}

template <typename T>
kernel void ConvolveGPU(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    device const T* Kernel [[buffer(2)]],
    constant size_t& kernelSize [[buffer(3)]],
    constant size_m* shape1 [[buffer(4)]],
    constant size_m* shape2 [[buffer(5)]],
    constant size_m* strides [[buffer(6)]],
    constant size_m& elementStride [[buffer(7)]],
    constant size_m& convDims [[buffer(8)]],
    uint gid [[thread_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgid [[threadgroup_position_in_grid]])
{
    // Pre-calculate kernel half-sizes (constant across iterations)
    int kernelHalf[8]; // Max 8 dimensions
    for (int j = 0; j < convDims; j++) {
        kernelHalf[j] = shape2[j] / 2;
    }
    
    // Pre-calculate grid position indices (constant across kernel loop)
    int gridPos[8];
    uint rem = gid;
    for (int j = convDims-1; j >= 0; j--) {
        gridPos[j] = rem % shape1[j];
        rem /= shape1[j];
    }
    
    // Calculate output base offset once
    device T* outBase = outMat + gid * elementStride;
    
    // Initialize accumulator in registers (not memory)
    T accum[16]; // Assuming elementStride <= 16, adjust as needed
    for (uint l = 0; l < elementStride; l++) {
        accum[l] = 0;
    }
    
    // Main convolution loop
    for (int i = 0; i < kernelSize; i++) {
        int offset = 0;
        uint remKernel = i;
        bool outOfBounds = false;
        
        // Calculate input offset for this kernel element
        for (int j = convDims-1; j >= 0; j--) {
            int kernelIndex = (remKernel % shape2[j]) - kernelHalf[j];
            int gridDimIndex = gridPos[j] + kernelIndex;
            
            // Bounds check
            if (gridDimIndex < 0 || gridDimIndex >= shape1[j]) {
                outOfBounds = true;
                break;
            }
            
            offset += strides[j] * gridDimIndex;
            remKernel /= shape2[j];
        }
        
        if (outOfBounds) continue;
        
        // Accumulate in registers
        T kernelVal = Kernel[i];
        device const T* inBase = inMat + offset;
        
        // Vectorized accumulation (loop unrolling hint)
        for (uint l = 0; l < elementStride; l++) {
            accum[l] += inBase[l] * kernelVal;
        }
    }
    
    // Write back accumulated results
    for (uint l = 0; l < elementStride; l++) {
        outBase[l] = accum[l];
    }
}

template <typename T>
kernel void ConvolveGPU_FULL(device T* outMat [[buffer(0)]], device const T* inMat [[buffer(1)]], device const T* Kernel [[buffer(2)]], constant size_t& kernelSize [[buffer(3)]], constant size_m* shape1 [[buffer(4)]], constant size_m* shape2 [[buffer(5)]], constant size_m* strides [[buffer(6)]], constant size_m& elementStride [[buffer(7)]], constant size_m& convDims [[buffer(8)]], constant uint8_t& mode [[buffer(9)]], uint gid [[thread_position_in_grid]]) {
    int offset = 0;
    int kernelIndex = 0;
    int gridDimIndex = 0;
    bool outOfBounds = false;
    
    for (int i = 0; i < kernelSize; i++) {
        offset = 0;
        uint rem = gid;
        uint remKernel = i;
        for (int j = convDims-1; j >= 0; j--) {
            // mathematically we always flip the kernel before convolving this line effectively does that
            kernelIndex = (remKernel % shape2[j]);
            gridDimIndex = (rem % (shape1[j] + shape2[j] -1)) - kernelIndex;

            if (gridDimIndex < 0) {
                outOfBounds = true;
                if (mode == 0) {break;}
                if (mode == 1) { gridDimIndex = 0; } // extend the corner patter
            }
            if ( gridDimIndex >= shape1[j]) {
                outOfBounds = true;
                if (mode == 0) {break;}
                if (mode == 1) { gridDimIndex = shape1[j] -1; } // extend the corner patter
            }
            offset += strides[j] * (gridDimIndex);
            rem /= (shape1[j] + shape2[j] -1);
            remKernel /= shape2[j];
            
        }
        if (outOfBounds && mode == 0) {outOfBounds = false; continue;}
//        if (gid == 1) {
//            metal::os_log_default.log_info("i = %d", i);
//        }
        for (uint l = 0; l < elementStride; l++) {
//            if (gid == 1) {
//                metal::os_log_default.log_info("l = %u, inMatIndex= %d with mode %u", l, (offset), (uint)mode);
//            }
//            outMat[gid * elementStride + l] += inMat[(offset) * elementStride + l] * Kernel[i];
            outMat[gid * elementStride + l] += inMat[(offset) + l] * Kernel[i];
        }
    }
}


instantiate_kernel("ConvolveGPU_0", ConvolveGPU, float);
instantiate_kernel("ConvolveGPU_FULL_0", ConvolveGPU_FULL, float);
