//
//  Utils.h
//  WorldOf3D
//
//  Created by Aditya Dudeja on 14/07/25.
//

#ifndef Utils_h
#define Utils_h

using size_m = uint32_t;
#define instantiate_kernel(name, func, ...) \
  template [[host_name(name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;



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
