#include <metal_stdlib>
#include "Utils.h"
using namespace metal;
// Match IEEE NaN/Inf and signed-zero behavior even when the app enables fast math.
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

template <typename T>
kernel void FloorGPU_0(
    device T* out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    constant size_m* shape [[buffer(2)]],
    constant size_m* sinput [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    size_t o = gid;
    size_t iinput = size_t(gid) * sinput[0];
    if constexpr (is_integral_v<T>) out[o] = input[iinput]; else out[o] = T(floor(float(input[iinput])));
}

template <typename T>
kernel void FloorGPU_1(
    device T* out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    constant size_m* shape [[buffer(2)]],
    constant size_m* sinput [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]) {
    size_t o = size_t(gid.y) * shape[1] + gid.x;
    size_t iinput = size_t(gid.y) * sinput[0] + size_t(gid.x) * sinput[1];
    if constexpr (is_integral_v<T>) out[o] = input[iinput]; else out[o] = T(floor(float(input[iinput])));
}

template <typename T>
kernel void FloorGPU_2(
    device T* out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    constant size_m* shape [[buffer(2)]],
    constant size_m* sinput [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    size_t o = (size_t(gid.z) * shape[1] + gid.y) * shape[2] + gid.x;
    size_t iinput = size_t(gid.z) * sinput[0] + size_t(gid.y) * sinput[1] + size_t(gid.x) * sinput[2];
    if constexpr (is_integral_v<T>) out[o] = input[iinput]; else out[o] = T(floor(float(input[iinput])));
}

template <typename T>
kernel void FloorGPU_3(
    device T* out [[buffer(0)]],
    device const T* input [[buffer(1)]],
    constant size_m* shape [[buffer(2)]],
    constant size_m* sinput [[buffer(3)]],
    constant uint& rank [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]) {
    size_t o = (size_t(gid.z) * shape[rank - 2] + gid.y) * shape[rank - 1] + gid.x;
    size_t iinput = size_t(gid.y) * sinput[rank - 2] + size_t(gid.x) * sinput[rank - 1];
    size_t rem = gid.z;
    for (int d = int(rank) - 3; d >= 0; --d) {
        size_t index = rem % shape[d];
        rem /= shape[d];
        iinput += index * sinput[d];
    }
    if constexpr (is_integral_v<T>) out[o] = input[iinput]; else out[o] = T(floor(float(input[iinput])));
}

#define INSTANTIATE_FLOOR(code, type) \
    instantiate_kernel("FloorGPU_" #code "_0", FloorGPU_0, type); \
    instantiate_kernel("FloorGPU_" #code "_1", FloorGPU_1, type); \
    instantiate_kernel("FloorGPU_" #code "_2", FloorGPU_2, type); \
    instantiate_kernel("FloorGPU_" #code "_3", FloorGPU_3, type);

INSTANTIATE_FLOOR(0, float)
INSTANTIATE_FLOOR(1, half)
INSTANTIATE_FLOOR(2, uint8_t)
INSTANTIATE_FLOOR(3, int)
INSTANTIATE_FLOOR(4, int16_t)
INSTANTIATE_FLOOR(5, uint32_t)
INSTANTIATE_FLOOR(6, uint16_t)
INSTANTIATE_FLOOR(7, bool)
INSTANTIATE_FLOOR(8, int8_t)
INSTANTIATE_FLOOR(10, bfloat)
INSTANTIATE_FLOOR(11, int64_t)
INSTANTIATE_FLOOR(12, uint64_t)
