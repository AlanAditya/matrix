#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

// Scatter: the inverse of Take. The grid runs over the updates, in take(dst, indices, axis)'s
// shape. For each update element:
//   out[out_base + indices[idx_off] * out_axis_stride]  (mode)=  updates[upd_off]
// out_strides are out's strides with 0 on the index dims, idx_strides are 0 on the non-index
// dims, and upd_strides are 0 on broadcast dims (all resolved on the CPU, see
// matrix::scatter). out already holds a copy of dst when the kernel runs.
//
// mode is ScatterMode (Mods/Utils.h): 0 SET, 1 ADD, 2 PROD, 3 MAX, 4 MIN. It's a uniform
// runtime argument rather than a template parameter so the pipeline grid stays the same
// size as Take's.

struct ScatterAddOp  { template <typename T> static T apply(T a, T b) { return T(a + b); } };
struct ScatterProdOp { template <typename T> static T apply(T a, T b) { return T(a * b); } };
struct ScatterMaxOp  { template <typename T> static T apply(T a, T b) { return a > b ? a : b; } };
struct ScatterMinOp  { template <typename T> static T apply(T a, T b) { return a < b ? a : b; } };

// Compare-and-swap fallback for every (op, dtype) pair Metal has no native atomic for.
// 8 and 16 bit types have no atomics at all, so they CAS the aligned 32-bit word that
// contains them and splice their bytes in (out is an owning buffer bound at offset 0, so
// word alignment follows from the element's byte offset; an aligned word never crosses a
// page, so the neighbouring bytes it touches are always mapped).
template <typename T, int S> struct ScatterCAS;

template <typename T> struct ScatterCAS<T, 4> {
    template <typename Op> static void apply(device T* out, size_m i, T v) {
        device atomic_uint* word = (device atomic_uint*)(out + i);
        uint expected = atomic_load_explicit(word, memory_order_relaxed);
        while (true) {
            uint desired = as_type<uint>(Op::apply(as_type<T>(expected), v));
            if (desired == expected) return;
            if (atomic_compare_exchange_weak_explicit(word, &expected, desired, memory_order_relaxed, memory_order_relaxed)) return;
        }
    }
};

template <typename T> struct ScatterCAS<T, 2> {
    template <typename Op> static void apply(device T* out, size_m i, T v) {
        size_m byte = i * 2;
        device atomic_uint* word = (device atomic_uint*)((device uchar*)out + (byte & ~3u));
        uint shift = (byte & 3u) * 8;
        uint mask = 0xFFFFu << shift;
        uint expected = atomic_load_explicit(word, memory_order_relaxed);
        while (true) {
            T cur = as_type<T>(ushort((expected & mask) >> shift));
            uint bits = uint(as_type<ushort>(Op::apply(cur, v)));
            uint desired = (expected & ~mask) | (bits << shift);
            if (desired == expected) return;
            if (atomic_compare_exchange_weak_explicit(word, &expected, desired, memory_order_relaxed, memory_order_relaxed)) return;
        }
    }
};

template <typename T> struct ScatterCAS<T, 1> {
    template <typename Op> static void apply(device T* out, size_m i, T v) {
        size_m byte = i;
        device atomic_uint* word = (device atomic_uint*)((device uchar*)out + (byte & ~3u));
        uint shift = (byte & 3u) * 8;
        uint mask = 0xFFu << shift;
        uint expected = atomic_load_explicit(word, memory_order_relaxed);
        while (true) {
            T cur = T((expected & mask) >> shift);
            uint bits = uint(uchar(Op::apply(cur, v)));
            uint desired = (expected & ~mask) | (bits << shift);
            if (desired == expected) return;
            if (atomic_compare_exchange_weak_explicit(word, &expected, desired, memory_order_relaxed, memory_order_relaxed)) return;
        }
    }
};

template <typename Op, typename T> struct ScatterReduce {
    static void apply(device T* out, size_m i, T v) { ScatterCAS<T, sizeof(T)>::template apply<Op>(out, i, v); }
};

// Native atomics where Metal has them.
template <> struct ScatterReduce<ScatterAddOp, float> {
    static void apply(device float* out, size_m i, float v) { atomic_fetch_add_explicit((device atomic_float*)(out + i), v, memory_order_relaxed); }
};
template <> struct ScatterReduce<ScatterAddOp, int> {
    static void apply(device int* out, size_m i, int v) { atomic_fetch_add_explicit((device atomic_int*)(out + i), v, memory_order_relaxed); }
};
template <> struct ScatterReduce<ScatterMaxOp, int> {
    static void apply(device int* out, size_m i, int v) { atomic_fetch_max_explicit((device atomic_int*)(out + i), v, memory_order_relaxed); }
};
template <> struct ScatterReduce<ScatterMinOp, int> {
    static void apply(device int* out, size_m i, int v) { atomic_fetch_min_explicit((device atomic_int*)(out + i), v, memory_order_relaxed); }
};
template <> struct ScatterReduce<ScatterAddOp, uint> {
    static void apply(device uint* out, size_m i, uint v) { atomic_fetch_add_explicit((device atomic_uint*)(out + i), v, memory_order_relaxed); }
};
template <> struct ScatterReduce<ScatterMaxOp, uint> {
    static void apply(device uint* out, size_m i, uint v) { atomic_fetch_max_explicit((device atomic_uint*)(out + i), v, memory_order_relaxed); }
};
template <> struct ScatterReduce<ScatterMinOp, uint> {
    static void apply(device uint* out, size_m i, uint v) { atomic_fetch_min_explicit((device atomic_uint*)(out + i), v, memory_order_relaxed); }
};

// Resolves one index and combines one update. Negative indices wrap (like MLX); indices
// still out of range after wrapping are skipped instead of writing out of bounds.
template <typename T, typename IdxT>
inline void scatter_one(device T* out, device const T* updates, device const IdxT* indices,
                        size_m out_off, size_m upd_off, size_m idx_off,
                        size_m out_axis_stride, size_m axis_size, int mode) {
    long iv = long(indices[idx_off]);
    if (iv < 0) iv += long(axis_size);
    if (iv < 0 || iv >= long(axis_size)) return;
    size_m o = out_off + size_m(iv) * out_axis_stride;
    T v = updates[upd_off];
    switch (mode) {
        case 0: out[o] = v; break;
        case 1: ScatterReduce<ScatterAddOp, T>::apply(out, o, v); break;
        case 2: ScatterReduce<ScatterProdOp, T>::apply(out, o, v); break;
        case 3: ScatterReduce<ScatterMaxOp, T>::apply(out, o, v); break;
        case 4: ScatterReduce<ScatterMinOp, T>::apply(out, o, v); break;
    }
}

// 1D Scatter
template <typename T, typename IdxT>
[[kernel]] void scatter_gg_nd1(device T* out [[buffer(0)]], device const T* updates [[buffer(1)]], device const IdxT* indices [[buffer(2)]], constant const size_m* out_strides [[buffer(3)]], constant const size_m* upd_strides [[buffer(4)]], constant const size_m* idx_strides [[buffer(5)]], constant const size_m& out_axis_stride [[buffer(6)]], constant const size_m& axis_size [[buffer(7)]], constant const int& mode [[buffer(8)]], uint index [[thread_position_in_grid]]) {
    scatter_one(out, updates, indices,
                index * out_strides[0], index * upd_strides[0], index * idx_strides[0],
                out_axis_stride, axis_size, mode);
}

// 2D Scatter
template <typename T, typename IdxT>
[[kernel]] void scatter_gg_nd2(device T* out [[buffer(0)]], device const T* updates [[buffer(1)]], device const IdxT* indices [[buffer(2)]], constant const size_m* out_strides [[buffer(3)]], constant const size_m* upd_strides [[buffer(4)]], constant const size_m* idx_strides [[buffer(5)]], constant const size_m& out_axis_stride [[buffer(6)]], constant const size_m& axis_size [[buffer(7)]], constant const int& mode [[buffer(8)]], uint2 index [[thread_position_in_grid]]) {
    scatter_one(out, updates, indices,
                index.y * out_strides[0] + index.x * out_strides[1],
                index.y * upd_strides[0] + index.x * upd_strides[1],
                index.y * idx_strides[0] + index.x * idx_strides[1],
                out_axis_stride, axis_size, mode);
}

// 3D Scatter
template <typename T, typename IdxT>
[[kernel]] void scatter_gg_nd3(device T* out [[buffer(0)]], device const T* updates [[buffer(1)]], device const IdxT* indices [[buffer(2)]], constant const size_m* out_strides [[buffer(3)]], constant const size_m* upd_strides [[buffer(4)]], constant const size_m* idx_strides [[buffer(5)]], constant const size_m& out_axis_stride [[buffer(6)]], constant const size_m& axis_size [[buffer(7)]], constant const int& mode [[buffer(8)]], uint3 index [[thread_position_in_grid]]) {
    scatter_one(out, updates, indices,
                index.z * out_strides[0] + index.y * out_strides[1] + index.x * out_strides[2],
                index.z * upd_strides[0] + index.y * upd_strides[1] + index.x * upd_strides[2],
                index.z * idx_strides[0] + index.y * idx_strides[1] + index.x * idx_strides[2],
                out_axis_stride, axis_size, mode);
}

// ND Scatter
template <typename T, typename IdxT>
[[kernel]] void scatter_gg(device T* out [[buffer(0)]], device const T* updates [[buffer(1)]], device const IdxT* indices [[buffer(2)]], constant const size_m* out_strides [[buffer(3)]], constant const size_m* upd_strides [[buffer(4)]], constant const size_m* idx_strides [[buffer(5)]], constant const size_m& out_axis_stride [[buffer(6)]], constant const size_m& axis_size [[buffer(7)]], constant const int& mode [[buffer(8)]], constant const size_m* upd_shape [[buffer(9)]], constant const int& ndim [[buffer(10)]], uint3 index [[thread_position_in_grid]]) {
    size_m out_off = index.x * out_strides[ndim-1] + index.y * out_strides[ndim-2];
    size_m upd_off = index.x * upd_strides[ndim-1] + index.y * upd_strides[ndim-2];
    size_m idx_off = index.x * idx_strides[ndim-1] + index.y * idx_strides[ndim-2];

    uint remA = index.z;
    for (int i = ndim-3; i >= 0; --i) {
        size_m mod = remA % upd_shape[i];
        out_off += mod * out_strides[i];
        upd_off += mod * upd_strides[i];
        idx_off += mod * idx_strides[i];
        remA /= upd_shape[i];
    }

    scatter_one(out, updates, indices, out_off, upd_off, idx_off, out_axis_stride, axis_size, mode);
}

// Index dtype codes follow matrix::dtype directly, same as Take.metal: only the integer
// dtypes are instantiated as indices -- 2=uint8_t, 3=int, 4=int16_t, 5=uint32_t, 6=uint16_t.
#define INSTANTIATE_SCATTER_ND_IDX(type_idx, type, idx_code, idx_type) \
    instantiate_kernel("ScatterGPU_nd_" #type_idx "_0_" #idx_code, scatter_gg_nd1, type, idx_type); \
    instantiate_kernel("ScatterGPU_nd_" #type_idx "_1_" #idx_code, scatter_gg_nd2, type, idx_type); \
    instantiate_kernel("ScatterGPU_nd_" #type_idx "_2_" #idx_code, scatter_gg_nd3, type, idx_type); \
    instantiate_kernel("ScatterGPU_nd_" #type_idx "_3_" #idx_code, scatter_gg, type, idx_type);

#define INSTANTIATE_SCATTER_ND(type_idx, type) \
    INSTANTIATE_SCATTER_ND_IDX(type_idx, type, 2, uint8_t) \
    INSTANTIATE_SCATTER_ND_IDX(type_idx, type, 3, int) \
    INSTANTIATE_SCATTER_ND_IDX(type_idx, type, 4, int16_t) \
    INSTANTIATE_SCATTER_ND_IDX(type_idx, type, 5, uint32_t) \
    INSTANTIATE_SCATTER_ND_IDX(type_idx, type, 6, uint16_t)

INSTANTIATE_SCATTER_ND(0, float)
INSTANTIATE_SCATTER_ND(1, half)
INSTANTIATE_SCATTER_ND(2, uint8_t)
INSTANTIATE_SCATTER_ND(3, int)
INSTANTIATE_SCATTER_ND(4, int16_t)
INSTANTIATE_SCATTER_ND(5, uint32_t)
INSTANTIATE_SCATTER_ND(6, uint16_t)
