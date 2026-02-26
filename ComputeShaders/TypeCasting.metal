//
//  Test.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 11/07/25.
//

#include <metal_stdlib>

using namespace metal;


#define instantiate_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

template <typename T>
kernel void AddGPU(device const T* A [[buffer(0)]], device const T* B [[buffer(1)]], device T* C [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    C[gid] = A[gid] + B[gid];
}

instantiate_kernel("AddGPU_F", AddGPU, float);
instantiate_kernel("AddGPU_I", AddGPU, int);


template <typename InT, typename OutT>
kernel void TypeCastingGPU(device OutT* out_buffer [[buffer(0)]], device const InT* in_buffer [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    // Perform the explicit type cast for each element
    out_buffer[gid] = (OutT)in_buffer[gid];
}



template <typename InT, typename OutT, int m, int n>
kernel void TypeCastingGPUStepUpDown(device const InT* in_buffer [[buffer(0)]], device OutT* out_buffer [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    // Perform the explicit type cast for each element
    // [ a1, a2, a3, a4 ] <-- [b1, b2] [b1. b2]
    // [ a1, a2, a3 ] [ a1, a2, a3 ]  <-- [b1, b2] [b1. b2] [b1. b2]
    // GID based on n
    
//    if constexpr (m > 1 && n > 1) {
//        for (int i = 0; i < n; i++) {
//            out_buffer[gid][i] = (OutT)in_buffer[gid * (n / m) + (i / m)][(gid + i) % m];
//        }
//    } else if constexpr (m == 1 && n > 1) {
//        for (int i = 0; i < n; i++) {
//            out_buffer[gid][i] = in_buffer[gid * (n / m) + (i / m)];
//        }
//    } else if constexpr (m > 1 && n == 1) {
//        for (int i = 0; i < n; i++) {
//            out_buffer[gid][i] = (OutT)in_buffer[gid * (n / m) + (i / m)][(gid + i) % m];
//        }
//    }

}


// 0 = float
// 1 = float
// 2 = uint8_t
// 3 = int
// 4 = uint16_t
// 5 = uint32_t
// 6 = simd_float2
// 7 = simd_float3
// 8 = simd_float4
#define INSTANTIATE_FROM_TYPE(src_idx, src_type) \
    instantiate_kernel("TypeCastingGPU_" #src_idx "_0", TypeCastingGPU, src_type, float); \
    instantiate_kernel("TypeCastingGPU_" #src_idx "_1", TypeCastingGPU, src_type, half); \
    instantiate_kernel("TypeCastingGPU_" #src_idx "_2", TypeCastingGPU, src_type, uint8_t); \
    instantiate_kernel("TypeCastingGPU_" #src_idx "_3", TypeCastingGPU, src_type, int); \
    instantiate_kernel("TypeCastingGPU_" #src_idx "_4", TypeCastingGPU, src_type, uint16_t); \
    instantiate_kernel("TypeCastingGPU_" #src_idx "_5", TypeCastingGPU, src_type, uint32_t); \

// All possible type casting kernel instantiations (64 total combinations)
INSTANTIATE_FROM_TYPE(0, float);
INSTANTIATE_FROM_TYPE(1, half);
INSTANTIATE_FROM_TYPE(2, uint8_t);
INSTANTIATE_FROM_TYPE(3, int);
INSTANTIATE_FROM_TYPE(4, uint16_t);
INSTANTIATE_FROM_TYPE(5, uint32_t);
