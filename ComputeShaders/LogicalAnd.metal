#include <metal_stdlib>
#include "Utils.h"
using namespace metal;
// Match IEEE NaN/Inf and signed-zero behavior even when the app enables fast math.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

template <typename T>
kernel void LogicalAndGPU_0(
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
    out[o] = bool(a[ia]) && bool(b[ib]);
}

template <typename T>
kernel void LogicalAndGPU_1(
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
    out[o] = bool(a[ia]) && bool(b[ib]);
}

template <typename T>
kernel void LogicalAndGPU_2(
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
    out[o] = bool(a[ia]) && bool(b[ib]);
}

template <typename T>
kernel void LogicalAndGPU_3(
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
    out[o] = bool(a[ia]) && bool(b[ib]);
}

#define INSTANTIATE_LOGICALAND(code, type) \
    instantiate_kernel("LogicalAndGPU_" #code "_0", LogicalAndGPU_0, type); \
    instantiate_kernel("LogicalAndGPU_" #code "_1", LogicalAndGPU_1, type); \
    instantiate_kernel("LogicalAndGPU_" #code "_2", LogicalAndGPU_2, type); \
    instantiate_kernel("LogicalAndGPU_" #code "_3", LogicalAndGPU_3, type);

INSTANTIATE_LOGICALAND(7, bool)
