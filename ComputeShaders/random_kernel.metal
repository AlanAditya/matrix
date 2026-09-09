//
//  random_kernel.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 09/09/26.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

union rbits {
  uint2 val;
  uchar4 bytes[2];
};

static constexpr constant uint32_t rotations[2][4] = {
    {13, 15, 26, 6},
    {17, 29, 16, 24}
};

// ============================================================================
// Threefry-2x32-20  (same as JAX / MLX)
//
// Counter-based PRNG: a reduced-round Threefish block cipher in CTR mode.
//   key   = which random stream   (your seed)
//   count = position in it        (element index)
// Stateless: thread N computes its value directly, no shared state.
// ARX only (add / rotate / xor) - no multiplies, no 64-bit ops.
// ============================================================================

inline rbits threefry2x32(const thread uint2& key, uint2 count) {
    // 3 subkey words: yours, plus a parity word (0x1BD11BDA = Threefish C240, hi 32)
    uint32_t ks[3] = {key.x, key.y, key.x ^ key.y ^ 0x1BD11BDA};
    
    rbits v;
    
    v.val.x = count.x + ks[0];
    v.val.y = count.y + ks[1];
    
    // 5 iterations x 4 rounds = 20 rounds
    for (int i = 0; i < 5; i++) {
        for (auto r : rotations[i%2]) {
            v.val.x += v.val.y;
            v.val.y = (v.val.y << r) | (v.val.y >> (32 - r));
            v.val.y ^= v.val.x;
        }
        // re-inject key every 4 rounds; +i+1 breaks round symmetry
        v.val.x += ks[(i + 1) % 3];
        v.val.y += ks[(i + 2) % 3] + i + 1;
    }
    return v;
}

// ============================================================================
// Distributions: pure transforms on random bits.
// ============================================================================

// [0,1) via mantissa stuffing: no division, exactly 2^-24 spacing
inline float bits_to_unit(uint32_t b) {
    return as_type<float>((b >> 9) | 0x3F800000) - 1.0f;
}
// (0,1] - safe for log() in Box-Muller
inline float bits_to_unit_nz(uint32_t b) {
    return ((float)(b >> 8) + 1.0f) * (1.0f / 16777216.0f);
}

struct UniformF {
    float apply(uint32_t b) const { return bits_to_unit(b); }
};

struct RandIntD {
    int      lo;
    uint32_t range;                  // hi - lo, computed unsigned on the host
    int apply(uint32_t b) const {
        // multiply-shift instead of %: one mulhi, no integer division
        return lo + (int)(((uint64_t)b * (uint64_t)range) >> 32);
    }
};

struct RandInt64D {
    long     lo;
    uint64_t range;
    // high half of a 128-bit product, built from 32-bit multiplies
    static uint64_t mulhi64(uint64_t a, uint64_t b) {
        uint32_t alo = (uint32_t)a, ahi = (uint32_t)(a >> 32);
        uint32_t blo = (uint32_t)b, bhi = (uint32_t)(b >> 32);
        uint64_t ll = (uint64_t)alo * blo;
        uint64_t lh = (uint64_t)alo * bhi;
        uint64_t hl = (uint64_t)ahi * blo;
        uint64_t hh = (uint64_t)ahi * bhi;
        uint64_t mid = (ll >> 32) + (uint32_t)lh + (uint32_t)hl;
        return hh + (lh >> 32) + (hl >> 32) + (mid >> 32);
    }
    long apply(uint64_t b) const {
        if (range == 0) return (long)b;          // full-range interval
        return lo + (long)mulhi64(b, range);
    }
};

// ============================================================================
// Kernel bodies
//
// Layout: thread g writes elements g and g+half.
// Consecutive threads write consecutive addresses -> coalesced.
// Counter starts at 0 every call; separation between calls comes from the key.
// ============================================================================

// One word per element. Each thread emits two elements.
template <typename T, typename D>
inline void fill_1word(device T* out, uint2 key, uint n, D dist, uint g) {
    uint half_n = (n + 1) / 2;
    if (g >= half_n) return;                     // kill padding threads

    auto v = threefry2x32(key, uint2(g, g + half_n));

    out[g] = (T)dist.apply(v.val.x);
    if (g + half_n < n)                           // odd n: last word has no slot
        out[g + half_n] = (T)dist.apply(v.val.y);
}

// Two words per element (int64). One thread, one element.
template <typename T, typename D>
inline void fill_2word(device T* out, uint2 key, uint n, D dist, uint g) {
    if (g >= n) return;
    auto v = threefry2x32(key, uint2(g, g + n));
    uint64_t wide = ((uint64_t)v.val.y << 32) | (uint64_t)v.val.x;
    out[g] = (T)dist.apply(wide);
}

// Normal: consumes both words per element via Box-Muller.
template <typename T>
inline void fill_normal(device T* out, uint2 key, uint n, uint g) {
    if (g >= n) return;
    auto v = threefry2x32(key, uint2(g, g + n));
    float u1 = bits_to_unit_nz(v.val.x);          // (0,1]
    float u2 = bits_to_unit(v.val.y);             // [0,1)
    out[g] = (T)(sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI_F * u2));
}

// ============================================================================
// Instantiations
//
// Float and int are separate macros on purpose: instantiating uniform/normal
// for an integer type silently produces all-zeros (float->int truncation)
// and wrapped negatives. Only float types get them.
// ============================================================================

#define INST_FLOAT(T, tag)                                                    \
kernel void rand_##tag(device T* out          [[buffer(0)]],                  \
                       constant uint2& key    [[buffer(1)]],                  \
                       constant uint& n       [[buffer(2)]],                  \
                       uint g [[thread_position_in_grid]]) {                  \
    fill_1word<T, UniformF>(out, key, n, UniformF{}, g);                      \
}                                                                             \
kernel void randn_##tag(device T* out         [[buffer(0)]],                  \
                        constant uint2& key   [[buffer(1)]],                  \
                        constant uint& n      [[buffer(2)]],                  \
                        uint g [[thread_position_in_grid]]) {                 \
    fill_normal<T>(out, key, n, g);                                           \
}

#define INST_INT(T, tag)                                                      \
kernel void randint_##tag(device T* out        [[buffer(0)]],                 \
                          constant uint2& key  [[buffer(1)]],                 \
                          constant uint& n     [[buffer(2)]],                 \
                          constant int& lo     [[buffer(3)]],                 \
                          constant uint& range [[buffer(4)]],                 \
                          uint g [[thread_position_in_grid]]) {               \
    fill_1word<T, RandIntD>(out, key, n, RandIntD{lo, range}, g);             \
}

INST_FLOAT(float, f32)
INST_FLOAT(half,  f16)

INST_INT(int,      i32)
INST_INT(short,    i16)
INST_INT(uint8_t,  u8)
INST_INT(uint,     u32)
INST_INT(ushort,   u16)

// int64 needs the 2-word path and 64-bit bounds
kernel void randint_i64(device long* out        [[buffer(0)]],
                        constant uint2& key     [[buffer(1)]],
                        constant uint& n        [[buffer(2)]],
                        constant long& lo       [[buffer(3)]],
                        constant uint64_t& rng  [[buffer(4)]],
                        uint g [[thread_position_in_grid]]) {
    fill_2word<long, RandInt64D>(out, key, n, RandInt64D{lo, rng}, g);
}

// Raw threefry output bits, for a user-facing random_bits() op.
[[kernel]] void rbits(device uint* out      [[buffer(0)]],
                      constant uint2& key   [[buffer(1)]],
                      constant uint& n      [[buffer(2)]],
                      uint g [[thread_position_in_grid]]) {
    uint half_n = (n + 1) / 2;
    if (g >= half_n) return;
    uint2 k = key;                                // constant -> thread copy for threefry2x32
    auto v = threefry2x32(k, uint2(g, g + half_n));
    out[g] = v.val.x;
    if (g + half_n < n) out[g + half_n] = v.val.y;
}
