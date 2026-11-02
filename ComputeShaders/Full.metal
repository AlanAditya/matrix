#include <metal_stdlib>
#include "Utils.h"
using namespace metal;
// Match IEEE NaN/Inf and signed-zero behavior even when the app enables fast math.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

template <typename T>
kernel void FullGPU_0(
    device T* out [[buffer(0)]],
    device const T* value [[buffer(1)]],
    constant size_m* shape [[buffer(2)]],
    constant size_m* svalue [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    size_t o = gid;
    size_t ivalue = size_t(gid) * svalue[0];
    out[o] = value[ivalue];
}

template <typename T>
kernel void FullGPU_1(
    device T* out [[buffer(0)]],
    device const T* value [[buffer(1)]],
    constant size_m* shape [[buffer(2)]],
    constant size_m* svalue [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    size_t o = size_t(gid.y) * shape[1] + gid.x;
    size_t ivalue = size_t(gid.y) * svalue[0] + size_t(gid.x) * svalue[1];
    out[o] = value[ivalue];
}

template <typename T>
kernel void FullGPU_2(
    device T* out [[buffer(0)]],
    device const T* value [[buffer(1)]],
    constant size_m* shape [[buffer(2)]],
    constant size_m* svalue [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    size_t o = (size_t(gid.z) * shape[1] + gid.y) * shape[2] + gid.x;
    size_t ivalue = size_t(gid.z) * svalue[0] + size_t(gid.y) * svalue[1] + size_t(gid.x) * svalue[2];
    out[o] = value[ivalue];
}

template <typename T>
kernel void FullGPU_3(
    device T* out [[buffer(0)]],
    device const T* value [[buffer(1)]],
    constant size_m* shape [[buffer(2)]],
    constant size_m* svalue [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    size_t o = (size_t(gid.z) * shape[rank - 2] + gid.y) * shape[rank - 1] + gid.x;
    size_t ivalue = size_t(gid.y) * svalue[rank - 2] + size_t(gid.x) * svalue[rank - 1];
    size_t rem = gid.z;
    for (int d = int(rank) - 3; d >= 0; --d) {
        size_t index = rem % shape[d];
        rem /= shape[d];
        ivalue += index * svalue[d];
    }
    out[o] = value[ivalue];
}

#define INSTANTIATE_FULL(code, type) \
    instantiate_kernel("FullGPU_" #code "_0", FullGPU_0, type); \
    instantiate_kernel("FullGPU_" #code "_1", FullGPU_1, type); \
    instantiate_kernel("FullGPU_" #code "_2", FullGPU_2, type); \
    instantiate_kernel("FullGPU_" #code "_3", FullGPU_3, type);

INSTANTIATE_FULL(0, float)
INSTANTIATE_FULL(1, half)
INSTANTIATE_FULL(2, uint8_t)
INSTANTIATE_FULL(3, int)
INSTANTIATE_FULL(4, int16_t)
INSTANTIATE_FULL(5, uint32_t)
INSTANTIATE_FULL(6, uint16_t)
INSTANTIATE_FULL(7, bool)
INSTANTIATE_FULL(8, int8_t)
INSTANTIATE_FULL(10, bfloat)
INSTANTIATE_FULL(11, int64_t)
INSTANTIATE_FULL(12, uint64_t)
