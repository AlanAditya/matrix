//
//  Utils.h
//  WorldOf3D
//
//  Created by Aditya Dudeja on 14/07/25.
//

#ifndef Utils_h
#define Utils_h

#define MAX_TENSOR_RANK 8
using size_m = uint32_t;
#define instantiate_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;

// One tensor block of a gpu_func kernel (matrix.h, GpuFunc): `name`, `name_shape`,
// `name_strides` (in elements) and `name_dims` at buffer slots 4b..4b+3. Output is block 0,
// host_op input i is block i + 1. Use `const T` for inputs.
#define TENSOR_ARG(name, T, b) \
  device T* name [[buffer(4 * (b))]], \
  constant const size_m* name##_shape [[buffer(4 * (b) + 1)]], \
  constant const size_m* name##_strides [[buffer(4 * (b) + 2)]], \
  constant const uint& name##_dims [[buffer(4 * (b) + 3)]]



template <typename IdxT = int64_t>
IdxT elem_to_loc(IdxT elem, constant const size_m* shape, constant const size_m* strides, int ndim) {
    IdxT loc = 0;
    for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
        loc += (elem % shape[i]) * IdxT(strides[i]);
        elem /= shape[i];
    }
    return loc;
}

#endif /* Utils_h */
