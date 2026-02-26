//
//  Concat.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 28/12/25.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

inline void memcpyGPU(device char* dest, device char* source, size_t no_of_bytes) {
    for (size_t i = 0; i < no_of_bytes; i++) {
        dest[i] = source[i];
    }
}

template <typename T>

kernel void concatGPU_2M(device T* out_buffer [[buffer(0)]], device const T* A [[buffer(1)]], device const T* B [[buffer(2)]], constant size_m& stride_out [[buffer(3)]],  constant size_m& stride_inA [[buffer(4)]], constant size_m& stride_inB [[buffer(5)]], constant size_m& batch_size [[buffer(6)]], uint gid [[thread_position_in_grid]] ) {

    for (uint32_t i = 0; i < batch_size; i++) {

        int index = gid * batch_size + i;

        memcpyGPU((device char*)(out_buffer + index * stride_out), (device char*)(A + index * stride_inA), sizeof(T) * stride_inA);

        memcpyGPU((device char*)(out_buffer + stride_inA + index * stride_out), (device char*)(B + index * stride_inB), sizeof(T) * stride_inB);

        

    }

}

instantiate_kernel("concatGPU_2M_0", concatGPU_2M, float);
instantiate_kernel("concatGPU_2M_1", concatGPU_2M, half);
instantiate_kernel("concatGPU_2M_2", concatGPU_2M, uint8_t);
instantiate_kernel("concatGPU_2M_3", concatGPU_2M, int);

template <typename IdxT = int64_t>
IdxT elem_to_loc_1(uint elem, constant const IdxT& stride) {
  return elem * IdxT(stride);
}
template <typename IdxT = int64_t>
IdxT elem_to_loc_2(uint2 elem, constant const IdxT strides[2]) {
  return elem.x * IdxT(strides[1]) + elem.y * IdxT(strides[0]);
}

template <typename IdxT = int64_t>
IdxT elem_to_loc_3(uint3 elem, constant const IdxT strides[3]) {
  return elem.x * IdxT(strides[0]) + elem.y * IdxT(strides[1]) +
      elem.z * IdxT(strides[2]);
}
template <typename IdxT = int64_t>
vec<IdxT, 2> elem_to_loc_2_nd(
    uint3 elem,
    constant const size_m* shape,
    constant const size_m* a_strides,
    constant const size_m* b_strides,
    int ndim) {
  vec<IdxT, 2> loc = {
      IdxT(
          elem.x * IdxT(a_strides[ndim - 1]) +
          IdxT(elem.y) * IdxT(a_strides[ndim - 2])),
      IdxT(
          elem.x * IdxT(b_strides[ndim - 1]) +
          elem.y * IdxT(b_strides[ndim - 2]))};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * IdxT(a_strides[d]);
    loc.y += l * IdxT(b_strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}


