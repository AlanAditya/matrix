#include <metal_stdlib>
#include "Utils.h"
using namespace metal;

// 1D Kernel
template <typename T>
kernel void BrodcastedCrossGPU_1Dgg(
    device T* result [[buffer(0)]], 
    device const T* A [[buffer(1)]], 
    device const T* B [[buffer(2)]], 
    constant size_m& strideR [[buffer(3)]], 
    constant size_m& strideA [[buffer(4)]], 
    constant size_m& strideB [[buffer(5)]], 
    uint gid [[thread_position_in_grid]]) 
{
    size_t GindexA = gid * strideA;
    size_t GindexB = gid * strideB;
    size_t GindexR = gid * strideR;
    
    vec<T, 3> a = vec<T, 3>(A[GindexA], A[GindexA+1], A[GindexA+2]);
    vec<T, 3> b = vec<T, 3>(B[GindexB], B[GindexB+1], B[GindexB+2]);
    vec<T, 3> r = cross(a, b);
    
    result[GindexR]   = r.x;
    result[GindexR+1] = r.y;
    result[GindexR+2] = r.z;
}

// 2D Kernel
template <typename T>
kernel void BrodcastedCrossGPU_2Dgg(
    device T* result [[buffer(0)]], 
    device const T* A [[buffer(1)]], 
    device const T* B [[buffer(2)]], 
    constant size_m* strideR [[buffer(3)]], 
    constant size_m* strideA [[buffer(4)]], 
    constant size_m* strideB [[buffer(5)]], 
    uint2 gid [[thread_position_in_grid]]) 
{
    size_t GindexA = gid.x * strideA[0] + gid.y * strideA[1];
    size_t GindexB = gid.x * strideB[0] + gid.y * strideB[1];
    size_t GindexR = gid.x * strideR[0] + gid.y * strideR[1];
    
    vec<T, 3> a = vec<T, 3>(A[GindexA], A[GindexA+1], A[GindexA+2]);
    vec<T, 3> b = vec<T, 3>(B[GindexB], B[GindexB+1], B[GindexB+2]);
    vec<T, 3> r = cross(a, b);
    
    result[GindexR]   = r.x;
    result[GindexR+1] = r.y;
    result[GindexR+2] = r.z;
}

// 3D Kernel
template <typename T>
kernel void BrodcastedCrossGPU_3Dgg(
    device T* result [[buffer(0)]], 
    device const T* A [[buffer(1)]], 
    device const T* B [[buffer(2)]], 
    constant size_m* strideR [[buffer(3)]], 
    constant size_m* strideA [[buffer(4)]], 
    constant size_m* strideB [[buffer(5)]], 
    uint3 gid [[thread_position_in_grid]]) 
{
    size_t GindexA = gid.x * strideA[0] + gid.y * strideA[1] + gid.z * strideA[2];
    size_t GindexB = gid.x * strideB[0] + gid.y * strideB[1] + gid.z * strideB[2];
    size_t GindexR = gid.x * strideR[0] + gid.y * strideR[1] + gid.z * strideR[2];
    
    vec<T, 3> a = vec<T, 3>(A[GindexA], A[GindexA+1], A[GindexA+2]);
    vec<T, 3> b = vec<T, 3>(B[GindexB], B[GindexB+1], B[GindexB+2]);
    vec<T, 3> r = cross(a, b);
    
    result[GindexR]   = r.x;
    result[GindexR+1] = r.y;
    result[GindexR+2] = r.z;
}

// ND Kernel
template <typename T>
kernel void BrodcastedCrossGPU_NDgg(
    device T* result [[buffer(0)]], 
    device const T* A [[buffer(1)]], 
    device const T* B [[buffer(2)]], 
    constant size_m* strideR [[buffer(3)]], 
    constant size_m* strideA [[buffer(4)]], 
    constant size_m* strideB [[buffer(5)]], 
    constant int& dims [[buffer(6)]], 
    uint gid [[thread_position_in_grid]]) 
{
    size_t GindexA = 0;
    size_t GindexB = 0;
    size_t GindexR = 0;
    
    int rem = gid;
    for (int i = 0; i < dims; i++) {
        size_t idx = rem / strideR[i];
        GindexA += idx * strideA[i];
        GindexB += idx * strideB[i];
        GindexR += idx * strideR[i];
        rem %= strideR[i];
    }
    
    vec<T, 3> a = vec<T, 3>(A[GindexA], A[GindexA+1], A[GindexA+2]);
    vec<T, 3> b = vec<T, 3>(B[GindexB], B[GindexB+1], B[GindexB+2]);
    vec<T, 3> r = cross(a, b);
    
    result[GindexR]   = r.x;
    result[GindexR+1] = r.y;
    result[GindexR+2] = r.z;
}

#define INSTANTIATE_FROM_TYPE(src_idx, type) \
    instantiate_kernel("BrodcastedCrossGPU_" #src_idx "_0", BrodcastedCrossGPU_1Dgg, type); \
    instantiate_kernel("BrodcastedCrossGPU_" #src_idx "_1", BrodcastedCrossGPU_2Dgg, type); \
    instantiate_kernel("BrodcastedCrossGPU_" #src_idx "_2", BrodcastedCrossGPU_3Dgg, type); \
    instantiate_kernel("BrodcastedCrossGPU_" #src_idx "_3", BrodcastedCrossGPU_NDgg, type); \

INSTANTIATE_FROM_TYPE(0, float);
INSTANTIATE_FROM_TYPE(1, half);
