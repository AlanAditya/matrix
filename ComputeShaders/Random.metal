// LEGACY: PCG-XSH-RR based RNG, superseded by the Threefry-2x32-20 kernels in
// random_kernel.metal (faster: no 64-bit multiply/modulo; unbiased randint;
// division-free float conversion). Kept for this iteration only - phase out next.

#include <metal_stdlib>
using namespace metal;

struct PCG {
    uint64_t state;
    uint64_t inc;

    PCG(uint64_t seed, uint64_t seq) {
        state = 0U;
        inc = (seq << 1u) | 1u;
        next();
        state += seed;
        next();
    }

    uint32_t next() {
        uint64_t oldstate = state;
        state = oldstate * 6364136223846793005ULL + inc;
        uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
        uint32_t rot = oldstate >> 59u;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    
    float next_float() {
        return (float)next() / (float)0xFFFFFFFF;
    }
};

template <typename T>
void fill_randint_impl(device T* output,
                         constant uint& size,
                         constant uint& seed,
                         constant int& low,
                         constant int& high,
                         uint id) {
    if (id >= size) return;
    PCG rng(seed, id);
    uint32_t range = (uint32_t)(high - low);
    output[id] = (T)(low + (int)(rng.next() % range));
}

template <typename T>
void fill_rand_impl(device T* output,
                      constant uint& size,
                      constant uint& seed,
                      uint id) {
    if (id >= size) return;
    PCG rng(seed, id);
    output[id] = (T)rng.next_float();
}

template <typename T>
void fill_randn_impl(device T* output,
                       constant uint& size,
                       constant uint& seed,
                       uint id) {
    if (id >= size) return;
    PCG rng(seed, id);
    float u1 = rng.next_float();
    float u2 = rng.next_float();
    float z0 = sqrt(-2.0 * log(u1 + 1e-7)) * cos(2.0 * M_PI_F * u2);
    output[id] = (T)z0;
}

#define INSTANTIATE_RAND_LEGACY(type, idx) \
kernel void fill_randint_legacy_##idx(device type* output [[buffer(0)]], \
                               constant uint& size [[buffer(1)]], \
                               constant uint& seed [[buffer(2)]], \
                               constant int& low [[buffer(3)]], \
                               constant int& high [[buffer(4)]], \
                               uint id [[thread_position_in_grid]]) { \
    fill_randint_impl<type>(output, size, seed, low, high, id); \
} \
kernel void fill_rand_legacy_##idx(device type* output [[buffer(0)]], \
                            constant uint& size [[buffer(1)]], \
                            constant uint& seed [[buffer(2)]], \
                            uint id [[thread_position_in_grid]]) { \
    fill_rand_impl<type>(output, size, seed, id); \
} \
kernel void fill_randn_legacy_##idx(device type* output [[buffer(0)]], \
                             constant uint& size [[buffer(1)]], \
                             constant uint& seed [[buffer(2)]], \
                             uint id [[thread_position_in_grid]]) { \
    fill_randn_impl<type>(output, size, seed, id); \
}

INSTANTIATE_RAND_LEGACY(float, 0)
INSTANTIATE_RAND_LEGACY(half, 1)
INSTANTIATE_RAND_LEGACY(uint8_t, 2)
INSTANTIATE_RAND_LEGACY(int, 3)
INSTANTIATE_RAND_LEGACY(short, 4)
INSTANTIATE_RAND_LEGACY(uint, 5)
INSTANTIATE_RAND_LEGACY(ushort, 6)
