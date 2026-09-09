#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

// Device-side constant fill (ones(), and any future scalar-fill factory).
// Unlike zeros() there's no single-byte pattern that works for every dtype,
// so this needs a real per-type kernel instead of a blitCommandEncoder fillBuffer.
template <typename T>
inline void fill_scalar(device T* out, constant uint& n, uint g) {
    if (g >= n) return;
    out[g] = (T)1;
}

#define INST_FILL_ONES(T, tag)                                        \
kernel void fill_ones_##tag(device T* out    [[buffer(0)]],           \
                            constant uint& n  [[buffer(1)]],           \
                            uint g [[thread_position_in_grid]]) {      \
    fill_scalar<T>(out, n, g);                                        \
}

INST_FILL_ONES(float, f32)
INST_FILL_ONES(half,  f16)
INST_FILL_ONES(uint8_t, u8)
INST_FILL_ONES(int,     i32)
INST_FILL_ONES(short,   i16)
INST_FILL_ONES(uint,    u32)
INST_FILL_ONES(ushort,  u16)

// Device-side gaussian fill (matrix::gaussian()). Templated over T so Float and Float16
// output share one body - the sum accumulator stays float regardless of T since summing
// in half would lose too much precision for the normalize pass below.
//
// shape/strides are pre-padded to 3 entries by the caller: unused trailing dims get
// shape=1 and stride=0, which collapses their (i/j/k) index to 0 and their (dx/dy/dz)
// term to 0 automatically, so this one kernel body covers the 1D/2D/3D cases without
// branching on dims.
//
// sum_out accumulates the unnormalized values via a float atomic so normalize_by_sum_*
// can divide by the true sum in a second dispatch - this is what lets gaussian(..., true)
// (the default) stay fully on the GPU instead of reading the buffer back to the CPU to sum it.
template <typename T>
inline void fill_gaussian(device T* out, device atomic<float>* sum_out, constant uint& n,
                           constant uint3& shape, constant uint3& strides,
                           constant float& std_dev, uint gid) {
    if (gid >= n) return;

    uint k = gid % shape.z;
    uint rem = gid / shape.z;
    uint j = rem % shape.y;
    uint i = rem / shape.y;

    float c0 = (float(shape.x) - 1.0f) / 2.0f;
    float c1 = (float(shape.y) - 1.0f) / 2.0f;
    float c2 = (float(shape.z) - 1.0f) / 2.0f;
    float dx = float(i) - c0;
    float dy = float(j) - c1;
    float dz = float(k) - c2;

    float val = exp(-(dx * dx + dy * dy + dz * dz) / (2.0f * std_dev * std_dev));
    uint idx = i * strides.x + j * strides.y + k * strides.z;
    out[idx] = (T)val;
    atomic_fetch_add_explicit(sum_out, val, memory_order_relaxed);
}

#define INST_FILL_GAUSSIAN(T, tag)                                                     \
kernel void fill_gaussian_##tag(device T* out                  [[buffer(0)]],          \
                                 device atomic<float>* sum_out  [[buffer(1)]],          \
                                 constant uint& n                [[buffer(2)]],         \
                                 constant uint3& shape            [[buffer(3)]],        \
                                 constant uint3& strides           [[buffer(4)]],       \
                                 constant float& std_dev            [[buffer(5)]],      \
                                 uint gid [[thread_position_in_grid]]) {                \
    fill_gaussian<T>(out, sum_out, n, shape, strides, std_dev, gid);                    \
}

INST_FILL_GAUSSIAN(float, f32)
INST_FILL_GAUSSIAN(half,  f16)

template <typename T>
inline void normalize_by_sum(device T* out, constant uint& n, constant float& sum_val, uint gid) {
    if (gid >= n) return;
    if (sum_val > 0.0f) {
        out[gid] = (T)(float(out[gid]) / sum_val);
    }
}

#define INST_NORMALIZE_BY_SUM(T, tag)                                          \
kernel void normalize_by_sum_##tag(device T* out       [[buffer(0)]],          \
                                    constant uint& n     [[buffer(1)]],         \
                                    constant float& sum_val [[buffer(2)]],      \
                                    uint gid [[thread_position_in_grid]]) {     \
    normalize_by_sum<T>(out, n, sum_val, gid);                                 \
}

INST_NORMALIZE_BY_SUM(float, f32)
INST_NORMALIZE_BY_SUM(half,  f16)

// ============================================================================
// Perlin noise (matrix::perlin()) - classic improved-Perlin, ported from the
// CPU reference in MatrixH.mm's noise_texture(). Every axis is a real spatial
// axis of one 3D lattice (grad()/fade()/lerp() all take the full x/y/z), so
// cells correlate along every axis identically - a lower-dim output just
// samples a fixed-z (or fixed-y,z) slice of the same coherent 3D field via
// the same shape/stride padding trick as fill_gaussian_f32 above, rather than
// reshaping a 1D noise line into more dimensions.
// ============================================================================

static constexpr constant int kPerlinPerm[512] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
    8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
    35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,74,165,71,
    134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,132,187,208,89,
    18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,52,217,226,
    250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
    189,28,42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,43,
    172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,218,246,97,
    228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,145,235,249,14,239,
    107,49,192,214,31,181,199,106,157,184,84,204,176,115,121,50,45,127,4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

inline float perlin_fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float perlin_lerp(float t, float a, float b) {
    return a + t * (b - a);
}

inline float perlin_grad(int hash, float x, float y, float z) {
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// One lattice-cell evaluation of 3D Perlin noise, in [-1, 1].
inline float perlin3d(float x, float y, float z) {
    int X = ((int)floor(x)) & 255;
    int Y = ((int)floor(y)) & 255;
    int Z = ((int)floor(z)) & 255;

    x -= floor(x);
    y -= floor(y);
    z -= floor(z);

    float u = perlin_fade(x);
    float v = perlin_fade(y);
    float w = perlin_fade(z);

    int A  = kPerlinPerm[X]   + Y, AA = kPerlinPerm[A] + Z, AB = kPerlinPerm[A + 1] + Z;
    int B  = kPerlinPerm[X+1] + Y, BA = kPerlinPerm[B] + Z, BB = kPerlinPerm[B + 1] + Z;

    return perlin_lerp(w,
        perlin_lerp(v,
            perlin_lerp(u, perlin_grad(kPerlinPerm[AA],   x,     y,     z),
                           perlin_grad(kPerlinPerm[BA],   x - 1, y,     z)),
            perlin_lerp(u, perlin_grad(kPerlinPerm[AB],   x,     y - 1, z),
                           perlin_grad(kPerlinPerm[BB],   x - 1, y - 1, z))),
        perlin_lerp(v,
            perlin_lerp(u, perlin_grad(kPerlinPerm[AA+1], x,     y,     z - 1),
                           perlin_grad(kPerlinPerm[BA+1], x - 1, y,     z - 1)),
            perlin_lerp(u, perlin_grad(kPerlinPerm[AB+1], x,     y - 1, z - 1),
                           perlin_grad(kPerlinPerm[BB+1], x - 1, y - 1, z - 1))));
}

// shape/strides padded to 3 entries exactly like fill_gaussian_f32 (unused trailing
// axes get shape=1/stride=0, pinning that axis's coordinate to a single value - a real
// slice of the same 3D field, not a dimensionality hack). seed_offset shifts the sample
// point through the (fixed) permutation table so different calls draw different fields
// without needing to re-upload a shuffled table.
kernel void fill_perlin_f32(device float* out            [[buffer(0)]],
                            constant uint& n               [[buffer(1)]],
                            constant uint3& shape           [[buffer(2)]],
                            constant uint3& strides         [[buffer(3)]],
                            constant float& scale            [[buffer(4)]],
                            constant float3& seed_offset      [[buffer(5)]],
                            constant int& octaves              [[buffer(6)]],
                            constant float& persistence         [[buffer(7)]],
                            constant float& lacunarity           [[buffer(8)]],
                            uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;

    uint k = gid % shape.z;
    uint rem = gid / shape.z;
    uint j = rem % shape.y;
    uint i = rem / shape.y;

    float x = (float(i) / float(shape.x)) * scale + seed_offset.x;
    float y = (float(j) / float(shape.y)) * scale + seed_offset.y;
    float z = (float(k) / float(shape.z)) * scale + seed_offset.z;

    float value = 0.0f;
    float amplitude = 1.0f;
    float frequency = 1.0f;
    float max_value = 0.0f;
    for (int o = 0; o < octaves; ++o) {
        value += perlin3d(x * frequency, y * frequency, z * frequency) * amplitude;
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }

    float noise_val = (max_value > 0.0f) ? (value / max_value) : 0.0f;
    // fbm/perlin3d is signed, roughly in [-1, 1] - remap to [0, 1] for display, matching
    // the CPU reference in MatrixH.mm's noise_texture(). Without this, ~half the values
    // are negative and render as black.
    uint idx = i * strides.x + j * strides.y + k * strides.z;
    out[idx] = saturate(noise_val * 0.5f + 0.5f);
}
