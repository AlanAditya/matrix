//
//  complex.cpp
//  WorldOf3D
//
//  Created by Aditya Dudeja on 27/10/26.
//


#include<iostream>
#include <simd/simd.h>

struct complex64 {
    simd_float2 z;
    complex64() = default;
    constexpr complex64(float re, float im = 0.0f) : z{re, im} {}
    explicit constexpr complex64(simd_float2 v) : z(v) {}
    float re() const { return z.x; }
    float im() const { return z.y; }
};

inline complex64 operator+(const complex64& lhs, const complex64& rhs) {
    return complex64(lhs.z + rhs.z);
}

inline complex64 operator-(const complex64& lhs, const complex64& rhs) {
    return complex64(lhs.z - rhs.z);
}

// (a + bi)(c + di) = (a, b)·c + (-b, a)·d: two vector multiply-adds instead of four scalar products
inline complex64 operator*(const complex64& lhs, const complex64& rhs) {
    return complex64(lhs.z * rhs.z.x + simd_make_float2(-lhs.z.y, lhs.z.x) * rhs.z.y);
}

inline complex64 conj(complex64 z) { return complex64(z.z * simd_make_float2(1.0f, -1.0f)); }

inline float abs(complex64 z) {return simd_length(z.z);}
inline float mag_sq(complex64 z) { return simd_length_squared(z.z); }

inline complex64 operator/(const complex64& lhs, float rhs) {
    return complex64(lhs.z / rhs);
}

inline complex64 operator/(const complex64& lhs, const complex64& rhs) {
    return lhs * conj(rhs) / mag_sq(rhs);
}