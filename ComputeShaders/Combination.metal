#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

// Broadcasted elementwise binomial coefficient: result = nCr(N, R), in the (promoted) input
// dtype. Inputs are truncated to integers; R < 0, N < 0 or R > N give 0. Strides come from
// BroadcastDescriptor + collapse_dims (0 on broadcast dims), see matrix::C. The CPU path is
// matrix::C_backend in Matrix.mm.

// Integer dtypes: exact, saturating at the dtype's max. Each step is r = r * (n - k + i) / i,
// which is C(n - k + i, i) (an integer that only grows with i); dividing out gcd(r, i) first keeps
// the product from overflowing before the answer does. Every dtype is at most 32 bits, so this
// runs in 32 bits and stops at the first step past the max (the CPU does it in 64 bits and
// saturates at the end: same result). 64-bit division here crashes the pipeline compiler.
template <typename T> inline T comb(T n_in, T k_in) {
    long n = long(n_in), k = long(k_in);
    if (n < 0 || k < 0 || k > n) return T(0);
    k = min(k, n - k);
    const uint max_v = uint(numeric_limits<T>::max());
    const uint m = uint(n - k);
    uint r = 1;
    for (uint i = 1; i <= uint(k); i++) {
        uint g = r, y = i;
        while (y != 0) { uint t = g % y; g = y; y = t; }   // g = gcd(r, i)
        uint a = r / g, b = (m + i) / (i / g);
        if (mulhi(a, b) != 0 || a * b > max_v) return numeric_limits<T>::max();
        r = a * b;
    }
    return T(r);
}

// Float/Float16: the same formula in float. Every step's value is an integer, so it is rounded
// back to one: fast-math division is not correctly rounded (116280 came out as 116280.008), and
// with the rint the result is exact while the products stay below 2^24.
inline float comb_f32(float n, float k) {
    n = trunc(n);
    k = trunc(k);
    if (n < 0 || k < 0 || k > n) return 0.0f;
    k = min(k, n - k);
    float r = 1.0f;
    for (float i = 1.0f; i <= k && !isinf(r); i += 1.0f) r = rint(r * (n - k + i) / i);
    return r;
}

template <> inline float comb(float n, float k) { return comb_f32(n, k); }
template <> inline half comb(half n, half k) { return half(comb_f32(float(n), float(k))); }

template <typename T>
kernel void CombinationGPU_1Dgg(device T* result [[buffer(0)]], device const T* N [[buffer(1)]], device const T* R [[buffer(2)]], constant size_m* strideR [[buffer(3)]], constant size_m* strideN [[buffer(4)]], constant size_m* strideK [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    size_t GindexN = gid * (size_t)strideN[0];
    size_t GindexK = gid * (size_t)strideK[0];
    size_t indexR  = gid * (size_t)strideR[0];
    result[indexR] = comb(N[GindexN], R[GindexK]);
}

template <typename T>
kernel void CombinationGPU_2Dgg(device T* result [[buffer(0)]], device const T* N [[buffer(1)]], device const T* R [[buffer(2)]], constant size_m* strideR [[buffer(3)]], constant size_m* strideN [[buffer(4)]], constant size_m* strideK [[buffer(5)]], uint2 gid [[thread_position_in_grid]]) {
    size_t GindexN = gid.x * (size_t)strideN[1] + gid.y * (size_t)strideN[0];
    size_t GindexK = gid.x * (size_t)strideK[1] + gid.y * (size_t)strideK[0];
    size_t indexR  = gid.x * (size_t)strideR[1] + gid.y * (size_t)strideR[0];
    result[indexR] = comb(N[GindexN], R[GindexK]);
}

template <typename T>
kernel void CombinationGPU_3Dgg(device T* result [[buffer(0)]], device const T* N [[buffer(1)]], device const T* R [[buffer(2)]], constant size_m* strideR [[buffer(3)]], constant size_m* strideN [[buffer(4)]], constant size_m* strideK [[buffer(5)]], uint3 gid [[thread_position_in_grid]]) {
    size_t GindexN = gid.x * (size_t)strideN[2] + gid.y * (size_t)strideN[1] + gid.z * (size_t)strideN[0];
    size_t GindexK = gid.x * (size_t)strideK[2] + gid.y * (size_t)strideK[1] + gid.z * (size_t)strideK[0];
    size_t indexR  = gid.x * (size_t)strideR[2] + gid.y * (size_t)strideR[1] + gid.z * (size_t)strideR[0];
    result[indexR] = comb(N[GindexN], R[GindexK]);
}

template <typename T>
kernel void CombinationGPU_NDgg(device T* result [[buffer(0)]], device const T* N [[buffer(1)]], device const T* R [[buffer(2)]], constant size_m* strideR [[buffer(3)]], constant size_m* strideN [[buffer(4)]], constant size_m* strideK [[buffer(5)]], constant const size_m* result_shape [[buffer(6)]], constant int& ndims [[buffer(7)]], uint3 gid [[thread_position_in_grid]]) {
    size_t GindexN = gid.x * (size_t)strideN[ndims-1] + gid.y * (size_t)strideN[ndims-2];
    size_t GindexK = gid.x * (size_t)strideK[ndims-1] + gid.y * (size_t)strideK[ndims-2];
    size_t indexR  = gid.x * (size_t)strideR[ndims-1] + gid.y * (size_t)strideR[ndims-2];

    size_t rem = gid.z;
    for (int i = ndims-3; i >= 0; --i) {
        size_t dim_index = rem % result_shape[i];
        GindexN += dim_index * strideN[i];
        GindexK += dim_index * strideK[i];
        indexR  += dim_index * strideR[i];
        rem /= result_shape[i];
    }
    result[indexR] = comb(N[GindexN], R[GindexK]);
}

#define INSTANTIATE_FROM_TYPE_COMBINATION(src_idx, type) \
    instantiate_kernel("CombinationGPU_" #src_idx "_0", CombinationGPU_1Dgg, type); \
    instantiate_kernel("CombinationGPU_" #src_idx "_1", CombinationGPU_2Dgg, type); \
    instantiate_kernel("CombinationGPU_" #src_idx "_2", CombinationGPU_3Dgg, type); \
    instantiate_kernel("CombinationGPU_" #src_idx "_3", CombinationGPU_NDgg, type);

INSTANTIATE_FROM_TYPE_COMBINATION(0, float);
INSTANTIATE_FROM_TYPE_COMBINATION(1, half);
INSTANTIATE_FROM_TYPE_COMBINATION(2, uint8_t);
INSTANTIATE_FROM_TYPE_COMBINATION(3, int);
INSTANTIATE_FROM_TYPE_COMBINATION(4, int16_t);
INSTANTIATE_FROM_TYPE_COMBINATION(5, uint32_t);
INSTANTIATE_FROM_TYPE_COMBINATION(6, uint16_t);
