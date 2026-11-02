#include <metal_stdlib>
#include "Utils.h"
using namespace metal;
// Match IEEE NaN/Inf and signed-zero behavior even when the app enables fast math.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

template <typename T>
kernel void LessGPU_0(
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
    out[o] = a[ia] < b[ib];
}

template <typename T>
kernel void LessGPU_1(
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
    out[o] = a[ia] < b[ib];
}

template <typename T>
kernel void LessGPU_2(
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
    out[o] = a[ia] < b[ib];
}

template <typename T>
kernel void LessGPU_3(
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
    out[o] = a[ia] < b[ib];
}

#define INSTANTIATE_LESS(code, type) \
    instantiate_kernel("LessGPU_" #code "_0", LessGPU_0, type); \
    instantiate_kernel("LessGPU_" #code "_1", LessGPU_1, type); \
    instantiate_kernel("LessGPU_" #code "_2", LessGPU_2, type); \
    instantiate_kernel("LessGPU_" #code "_3", LessGPU_3, type);

INSTANTIATE_LESS(0, float)
INSTANTIATE_LESS(1, half)
INSTANTIATE_LESS(2, uint8_t)
INSTANTIATE_LESS(3, int)
INSTANTIATE_LESS(4, int16_t)
INSTANTIATE_LESS(5, uint32_t)
INSTANTIATE_LESS(6, uint16_t)
INSTANTIATE_LESS(7, bool)
INSTANTIATE_LESS(8, int8_t)
INSTANTIATE_LESS(10, bfloat)
INSTANTIATE_LESS(11, int64_t)
INSTANTIATE_LESS(12, uint64_t)
