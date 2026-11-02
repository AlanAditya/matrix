#include <metal_stdlib>
#include "Utils.h"
using namespace metal;
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)

template <typename T>
kernel void ArangeGPU(device T* out [[buffer(0)]], device const T* start [[buffer(1)]],
                      device const T* step [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if constexpr (is_integral_v<T>) out[gid] = T(ulong(start[0]) + ulong(gid) * ulong(step[0]));
    else out[gid] = T(float(start[0]) + float(gid) * float(step[0]));
}

#define INSTANTIATE_ARANGE(code, type) instantiate_kernel("ArangeGPU_" #code, ArangeGPU, type);

INSTANTIATE_ARANGE(0, float)
INSTANTIATE_ARANGE(1, half)
INSTANTIATE_ARANGE(2, uint8_t)
INSTANTIATE_ARANGE(3, int)
INSTANTIATE_ARANGE(4, int16_t)
INSTANTIATE_ARANGE(5, uint32_t)
INSTANTIATE_ARANGE(6, uint16_t)
INSTANTIATE_ARANGE(8, int8_t)
INSTANTIATE_ARANGE(10, bfloat)
INSTANTIATE_ARANGE(11, int64_t)
INSTANTIATE_ARANGE(12, uint64_t)
