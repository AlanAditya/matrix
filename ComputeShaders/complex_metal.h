//
//  complex_metal.h
//  WorldOf3D
//
//  Created by Aditya Dudeja on 28/10/26.
//
//  complex64 for the Metal kernels (ComputeShaders/Utils.h includes it). The C++ side has its own in
//  matrix/complex.h: keep the two layouts identical (8 bytes, re first, = float2).
//
//  The app compiles shaders as Metal 3.2 (deployment target macOS 15), where every reference needs an
//  address space, so everything here is passed by value (8 bytes, one register).

#pragma once

#include <metal_stdlib>

struct complex64 {
    float2 z;
    complex64() = default;
    constexpr complex64(float re, float im = 0.0f) : z{re, im} {}
    explicit constexpr complex64(float2 v) : z(v) {}
    float re() const { return z.x; }
    float im() const { return z.y; }
};

inline complex64 operator+(complex64 lhs, complex64 rhs) {
    return complex64(lhs.z + rhs.z);
}

inline complex64 operator-(complex64 lhs, complex64 rhs) {
    return complex64(lhs.z - rhs.z);
}

// (a + bi)(c + di) = (a, b)·c + (-b, a)·d: two vector multiply-adds instead of four scalar products
inline complex64 operator*(complex64 lhs, complex64 rhs) {
    return complex64(lhs.z * rhs.z.x + float2{-lhs.z.y, lhs.z.x} * rhs.z.y);
}

inline complex64 conj(complex64 z) { return complex64(z.z * float2{1.0f, -1.0f}); }

inline float abs(complex64 z) { return metal::length(z.z); }
inline float mag_sq(complex64 z) { return metal::length_squared(z.z); }

inline complex64 operator/(complex64 lhs, float rhs) {
    return complex64(lhs.z / rhs);
}

inline complex64 operator/(complex64 lhs, complex64 rhs) {
    return lhs * conj(rhs) / mag_sq(rhs);
}

// Principal root, same as matrix/complex.h: re >= 0, im has b's sign; b / 2t avoids |z| - a cancelling.
inline complex64 sqrt(complex64 z) {
    float a = z.re(), b = z.im();
    if (a == 0.0f && b == 0.0f) return complex64(0.0f, b);   // keeps -0
    float t = metal::sqrt((metal::fabs(a) + abs(z)) * 0.5f);
    if (a >= 0.0f) return complex64(t, b / (2.0f * t));
    return complex64(metal::fabs(b) / (2.0f * t), metal::copysign(t, b));
}

// Principal log, same as matrix/complex.h: ln|z| + i·atan2(b, a), im in (-π, π]. Metal has no hypot
// or log1p: |z| is scaled by max(|a|, |b|) so it doesn't overflow / underflow, and log1p(x) is
// 2·atanh(x / (2 + x)) (Kahan's log(u)·x / (u - 1) needs u - 1 exact, which fast math doesn't keep).
inline complex64 log(complex64 z) {
    float a = z.re(), b = z.im();
    float m = metal::max(metal::fabs(a), metal::fabs(b));
    float r = (m == 0.0f || metal::isinf(m)) ? m : m * metal::sqrt((a / m) * (a / m) + (b / m) * (b / m));
    float re;
    if (r > 0.5f && r < 2.0f) {
        float x = (a - 1.0f) * (a + 1.0f) + b * b;
        re = metal::precise::atanh(x / (2.0f + x));
    } else {
        re = metal::precise::log(r);
    }
    return complex64(re, metal::precise::atan2(b, a));
}

static_assert(sizeof(complex64) == 8 && alignof(complex64) == 8, "complex64 must match the C++ complex64 / float2");
