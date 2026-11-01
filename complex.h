//
//  complex.h
//  WorldOf3D
//
//  Created by Aditya Dudeja on 27/10/26.
//
//  complex64: the element type of dtype::Complex64. Same 8-byte layout as numpy's complex64,
//  std::complex<float> and Metal's float2 (re first), so a contiguous [..., 2] float buffer
//  is a complex64 buffer.
//
//  C++ only. The Metal kernels have their own in ComputeShaders/complex_metal.h: keep the two
//  layouts identical.

#pragma once

#include <cmath>
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
    return complex64(lhs.z * rhs.z.x + simd_float2{-lhs.z.y, lhs.z.x} * rhs.z.y);
}

inline complex64 conj(complex64 z) { return complex64(z.z * simd_float2{1.0f, -1.0f}); }

inline float abs(complex64 z) { return simd_length(z.z); }
inline float mag_sq(complex64 z) { return simd_length_squared(z.z); }

inline complex64 operator/(const complex64& lhs, float rhs) {
    return complex64(lhs.z / rhs);
}

inline complex64 operator/(const complex64& lhs, const complex64& rhs) {
    return lhs * conj(rhs) / mag_sq(rhs);
}

// Principal root (as std::sqrt(std::complex), numpy, MLX): re >= 0, im has b's sign, so b = ±0 on the
// negative real axis gives ±i·sqrt|a|. Only the larger part comes from a sqrt; the other is b / 2t,
// which avoids the cancellation in |z| - a.
inline complex64 sqrt(const complex64& z) {
    float a = z.re(), b = z.im();
    if (a == 0.0f && b == 0.0f) return complex64(0.0f, b);   // keeps -0
    float t = std::sqrt((std::fabs(a) + abs(z)) * 0.5f);
    if (a >= 0.0f) return complex64(t, b / (2.0f * t));
    return complex64(std::fabs(b) / (2.0f * t), std::copysign(t, b));
}

// Principal log (as std::log(std::complex), numpy, MLX): ln|z| + i·atan2(b, a), im in (-π, π].
// b = ±0 on the negative real axis gives ±iπ; log(0) = -inf + i·atan2(b, a). hypot, not abs(z), so
// |z| doesn't overflow / underflow; near |z| = 1, log(r) is the log of something close to 1, and
// log1p of (a - 1)(a + 1) + b² keeps the digits.
inline complex64 log(const complex64& z) {
    float a = z.re(), b = z.im();
    float r = std::hypot(a, b);
    float re = (r > 0.5f && r < 2.0f) ? 0.5f * std::log1p((a - 1.0f) * (a + 1.0f) + b * b) : std::log(r);
    return complex64(re, std::atan2(b, a));
}

static_assert(sizeof(complex64) == 8 && alignof(complex64) == 8, "complex64 must match numpy complex64 / Metal float2");
