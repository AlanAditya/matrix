#include <metal_stdlib>
#include "Utils.h"
using namespace metal;
// Match IEEE NaN/Inf and signed-zero behavior even when the app enables fast math.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

template <typename T>
kernel void GreaterGPU_0(
    device bool* out [[buffer(0)]],
    device const T* a [[buffer(1)]],
    device const T* b [[buffer(2)]],
    constant size_m* shape [[buffer(3)]],
    constant size_m* sa [[buffer(4)]],
    constant size_m* sb [[buffer(5)]],
    constant uint& rank [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    size_t o = gid;
    size_t ia = size_t(gid) * sa[0];
    size_t ib = size_t(gid) * sb[0];
    out[o] = a[ia] > b[ib];
}

template <typename T>
kernel void GreaterGPU_1(
    device bool* out [[buffer(0)]],
    device const T* a [[buffer(1)]],
    device const T* b [[buffer(2)]],
    constant size_m* shape [[buffer(3)]],
    constant size_m* sa [[buffer(4)]],
    constant size_m* sb [[buffer(5)]],
    constant uint& rank [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
    size_t o = size_t(gid.y) * shape[1] + gid.x;
    size_t ia = size_t(gid.y) * sa[0] + size_t(gid.x) * sa[1];
    size_t ib = size_t(gid.y) * sb[0] + size_t(gid.x) * sb[1];
    out[o] = a[ia] > b[ib];
}

template <typename T>
kernel void GreaterGPU_2(
    device bool* out [[buffer(0)]],
    device const T* a [[buffer(1)]],
    device const T* b [[buffer(2)]],
    constant size_m* shape [[buffer(3)]],
    constant size_m* sa [[buffer(4)]],
    constant size_m* sb [[buffer(5)]],
    constant uint& rank [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
    size_t o = (size_t(gid.z) * shape[1] + gid.y) * shape[2] + gid.x;
    size_t ia = size_t(gid.z) * sa[0] + size_t(gid.y) * sa[1] + size_t(gid.x) * sa[2];
    size_t ib = size_t(gid.z) * sb[0] + size_t(gid.y) * sb[1] + size_t(gid.x) * sb[2];
    out[o] = a[ia] > b[ib];
}

template <typename T>
kernel void GreaterGPU_3(
    device bool* out [[buffer(0)]],
    device const T* a [[buffer(1)]],
    device const T* b [[buffer(2)]],
    constant size_m* shape [[buffer(3)]],
    constant size_m* sa [[buffer(4)]],
    constant size_m* sb [[buffer(5)]],
    constant uint& rank [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]]) {
    size_t o = (size_t(gid.z) * shape[rank - 2] + gid.y) * shape[rank - 1] + gid.x;
    size_t ia = size_t(gid.y) * sa[rank - 2] + size_t(gid.x) * sa[rank - 1];
    size_t ib = size_t(gid.y) * sb[rank - 2] + size_t(gid.x) * sb[rank - 1];
    size_t rem = gid.z;
    for (int d = int(rank) - 3; d >= 0; --d) {
        size_t index = rem % shape[d];
        rem /= shape[d];
        ia += index * sa[d];
        ib += index * sb[d];
    }
    out[o] = a[ia] > b[ib];
}

#define INSTANTIATE_GREATER(code, type) \
    instantiate_kernel("GreaterGPU_" #code "_0", GreaterGPU_0, type); \
    instantiate_kernel("GreaterGPU_" #code "_1", GreaterGPU_1, type); \
    instantiate_kernel("GreaterGPU_" #code "_2", GreaterGPU_2, type); \
    instantiate_kernel("GreaterGPU_" #code "_3", GreaterGPU_3, type);

INSTANTIATE_GREATER(0, float)
INSTANTIATE_GREATER(1, half)
INSTANTIATE_GREATER(2, uint8_t)
INSTANTIATE_GREATER(3, int)
INSTANTIATE_GREATER(4, int16_t)
INSTANTIATE_GREATER(5, uint32_t)
INSTANTIATE_GREATER(6, uint16_t)
INSTANTIATE_GREATER(7, bool)
INSTANTIATE_GREATER(8, int8_t)
INSTANTIATE_GREATER(10, bfloat)
INSTANTIATE_GREATER(11, int64_t)
INSTANTIATE_GREATER(12, uint64_t)
