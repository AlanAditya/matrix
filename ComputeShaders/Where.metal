#include <metal_stdlib>
#include "Utils.h"
using namespace metal;
// Match IEEE NaN/Inf and signed-zero behavior even when the app enables fast math.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

template <typename T>
kernel void WhereGPU_0(
    device T* out [[buffer(0)]],
    device const bool* cond [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const T* b [[buffer(3)]],
    constant size_m* shape [[buffer(4)]],
    constant size_m* scond [[buffer(5)]],
    constant size_m* sa [[buffer(6)]],
    constant size_m* sb [[buffer(7)]],
    constant uint& rank [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    size_t o = gid;
    size_t icond = size_t(gid) * scond[0];
    size_t ia = size_t(gid) * sa[0];
    size_t ib = size_t(gid) * sb[0];
    out[o] = cond[icond] ? a[ia] : b[ib];
}

template <typename T>
kernel void WhereGPU_1(
    device T* out [[buffer(0)]],
    device const bool* cond [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const T* b [[buffer(3)]],
    constant size_m* shape [[buffer(4)]],
    constant size_m* scond [[buffer(5)]],
    constant size_m* sa [[buffer(6)]],
    constant size_m* sb [[buffer(7)]],
    constant uint& rank [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]) {
    size_t o = size_t(gid.y) * shape[1] + gid.x;
    size_t icond = size_t(gid.y) * scond[0] + size_t(gid.x) * scond[1];
    size_t ia = size_t(gid.y) * sa[0] + size_t(gid.x) * sa[1];
    size_t ib = size_t(gid.y) * sb[0] + size_t(gid.x) * sb[1];
    out[o] = cond[icond] ? a[ia] : b[ib];
}

template <typename T>
kernel void WhereGPU_2(
    device T* out [[buffer(0)]],
    device const bool* cond [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const T* b [[buffer(3)]],
    constant size_m* shape [[buffer(4)]],
    constant size_m* scond [[buffer(5)]],
    constant size_m* sa [[buffer(6)]],
    constant size_m* sb [[buffer(7)]],
    constant uint& rank [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]) {
    size_t o = (size_t(gid.z) * shape[1] + gid.y) * shape[2] + gid.x;
    size_t icond = size_t(gid.z) * scond[0] + size_t(gid.y) * scond[1] + size_t(gid.x) * scond[2];
    size_t ia = size_t(gid.z) * sa[0] + size_t(gid.y) * sa[1] + size_t(gid.x) * sa[2];
    size_t ib = size_t(gid.z) * sb[0] + size_t(gid.y) * sb[1] + size_t(gid.x) * sb[2];
    out[o] = cond[icond] ? a[ia] : b[ib];
}

template <typename T>
kernel void WhereGPU_3(
    device T* out [[buffer(0)]],
    device const bool* cond [[buffer(1)]],
    device const T* a [[buffer(2)]],
    device const T* b [[buffer(3)]],
    constant size_m* shape [[buffer(4)]],
    constant size_m* scond [[buffer(5)]],
    constant size_m* sa [[buffer(6)]],
    constant size_m* sb [[buffer(7)]],
    constant uint& rank [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]]) {
    size_t o = (size_t(gid.z) * shape[rank - 2] + gid.y) * shape[rank - 1] + gid.x;
    size_t icond = size_t(gid.y) * scond[rank - 2] + size_t(gid.x) * scond[rank - 1];
    size_t ia = size_t(gid.y) * sa[rank - 2] + size_t(gid.x) * sa[rank - 1];
    size_t ib = size_t(gid.y) * sb[rank - 2] + size_t(gid.x) * sb[rank - 1];
    size_t rem = gid.z;
    for (int d = int(rank) - 3; d >= 0; --d) {
        size_t index = rem % shape[d];
        rem /= shape[d];
        icond += index * scond[d];
        ia += index * sa[d];
        ib += index * sb[d];
    }
    out[o] = cond[icond] ? a[ia] : b[ib];
}

#define INSTANTIATE_WHERE(code, type) \
    instantiate_kernel("WhereGPU_" #code "_0", WhereGPU_0, type); \
    instantiate_kernel("WhereGPU_" #code "_1", WhereGPU_1, type); \
    instantiate_kernel("WhereGPU_" #code "_2", WhereGPU_2, type); \
    instantiate_kernel("WhereGPU_" #code "_3", WhereGPU_3, type);

INSTANTIATE_WHERE(0, float)
INSTANTIATE_WHERE(1, half)
INSTANTIATE_WHERE(2, uint8_t)
INSTANTIATE_WHERE(3, int)
INSTANTIATE_WHERE(4, int16_t)
INSTANTIATE_WHERE(5, uint32_t)
INSTANTIATE_WHERE(6, uint16_t)
INSTANTIATE_WHERE(7, bool)
INSTANTIATE_WHERE(8, int8_t)
INSTANTIATE_WHERE(10, bfloat)
INSTANTIATE_WHERE(11, int64_t)
INSTANTIATE_WHERE(12, uint64_t)
