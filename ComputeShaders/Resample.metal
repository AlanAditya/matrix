//
//  Resample.metal
//  AdityaIntelligenceProMax
//
//  Generic ND tensor resampling (nearest / N-linear) to an arbitrary target shape
//  of the same rank. Each output thread computes its source coordinate(s) directly
//  from the per-axis scale factor instead of reading a materialized index tensor.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

// =====================================================================
// Nearest
// =====================================================================

// 1D Nearest
template <typename T>
kernel void resample_nearest_nd1(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m& out_stride [[buffer(2)]],
    constant const size_m& in_stride [[buffer(3)]],
    constant const size_m& in_shape [[buffer(4)]],
    constant const float& scale [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    size_m out_idx = index;
    float coord = float(out_idx) * scale;
    size_m in_idx = (size_m) round(coord);
    if (in_idx >= in_shape) in_idx = in_shape - 1;
    outMat[out_idx * out_stride] = inMat[in_idx * in_stride];
}

// 2D Nearest
template <typename T>
kernel void resample_nearest_nd2(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m* out_strides [[buffer(2)]],
    constant const size_m* in_strides [[buffer(3)]],
    constant const size_m* in_shape [[buffer(4)]],
    constant const float* scale [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    size_m out_idx0 = gid.y;
    size_m out_idx1 = gid.x;
    size_m out_off = out_idx0 * out_strides[0] + out_idx1 * out_strides[1];

    size_m in_idx0 = (size_m) round(float(out_idx0) * scale[0]);
    if (in_idx0 >= in_shape[0]) in_idx0 = in_shape[0] - 1;
    size_m in_idx1 = (size_m) round(float(out_idx1) * scale[1]);
    if (in_idx1 >= in_shape[1]) in_idx1 = in_shape[1] - 1;

    outMat[out_off] = inMat[in_idx0 * in_strides[0] + in_idx1 * in_strides[1]];
}

// 3D Nearest
template <typename T>
kernel void resample_nearest_nd3(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m* out_strides [[buffer(2)]],
    constant const size_m* in_strides [[buffer(3)]],
    constant const size_m* in_shape [[buffer(4)]],
    constant const float* scale [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    size_m out_idx0 = gid.z;
    size_m out_idx1 = gid.y;
    size_m out_idx2 = gid.x;
    size_m out_off = out_idx0 * out_strides[0] + out_idx1 * out_strides[1] + out_idx2 * out_strides[2];

    size_m in_idx0 = (size_m) round(float(out_idx0) * scale[0]);
    if (in_idx0 >= in_shape[0]) in_idx0 = in_shape[0] - 1;
    size_m in_idx1 = (size_m) round(float(out_idx1) * scale[1]);
    if (in_idx1 >= in_shape[1]) in_idx1 = in_shape[1] - 1;
    size_m in_idx2 = (size_m) round(float(out_idx2) * scale[2]);
    if (in_idx2 >= in_shape[2]) in_idx2 = in_shape[2] - 1;

    outMat[out_off] = inMat[in_idx0 * in_strides[0] + in_idx1 * in_strides[1] + in_idx2 * in_strides[2]];
}

// ND Nearest (generic, arbitrary rank)
// GID: [Last out dim, Second-to-last out dim, flattened remaining leading dims]
template <typename T>
kernel void resample_nearest_gg(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m* out_strides [[buffer(2)]],
    constant const size_m* in_strides [[buffer(3)]],
    constant const size_m* in_shape [[buffer(4)]],
    constant const float* scale [[buffer(5)]],
    constant const size_m* out_shape [[buffer(6)]],
    constant const int& ndim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    size_m out_idx[MAX_TENSOR_RANK];
    out_idx[ndim-1] = gid.x;
    if (ndim >= 2) out_idx[ndim-2] = gid.y;

    uint rem = gid.z;
    for (int i = ndim-3; i >= 0; --i) {
        out_idx[i] = rem % out_shape[i];
        rem /= out_shape[i];
    }

    size_m out_off = 0;
    size_m in_off = 0;
    for (int i = 0; i < ndim; i++) {
        out_off += out_idx[i] * out_strides[i];
        size_m in_idx = (size_m) round(float(out_idx[i]) * scale[i]);
        if (in_idx >= in_shape[i]) in_idx = in_shape[i] - 1;
        in_off += in_idx * in_strides[i];
    }

    outMat[out_off] = inMat[in_off];
}

// =====================================================================
// N-Linear (bilinear for rank 2, trilinear for rank 3, generalized via a
// 2^ndim corner blend for arbitrary rank). Accumulates in float regardless
// of T, casts back to T on write -- same convention as TypeCastingGPU.
// =====================================================================

// 1D Linear
template <typename T>
kernel void resample_linear_nd1(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m& out_stride [[buffer(2)]],
    constant const size_m& in_stride [[buffer(3)]],
    constant const size_m& in_shape [[buffer(4)]],
    constant const float& scale [[buffer(5)]],
    uint index [[thread_position_in_grid]])
{
    size_m out_idx = index;
    float coord = float(out_idx) * scale;
    float f = floor(coord);
    size_m fi = (size_m)f;
    if (fi >= in_shape) fi = in_shape - 1;
    size_m ci = fi + 1;
    if (ci >= in_shape) ci = in_shape - 1;
    float frac = coord - f;

    float v0 = float(inMat[fi * in_stride]);
    float v1 = float(inMat[ci * in_stride]);
    float sum = v0 * (1.0 - frac) + v1 * frac;

    outMat[out_idx * out_stride] = (T)sum;
}

// 2D Linear (bilinear)
template <typename T>
kernel void resample_linear_nd2(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m* out_strides [[buffer(2)]],
    constant const size_m* in_strides [[buffer(3)]],
    constant const size_m* in_shape [[buffer(4)]],
    constant const float* scale [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    size_m out_idx0 = gid.y;
    size_m out_idx1 = gid.x;
    size_m out_off = out_idx0 * out_strides[0] + out_idx1 * out_strides[1];

    float coord0 = float(out_idx0) * scale[0];
    float f0 = floor(coord0);
    size_m fi0 = (size_m)f0; if (fi0 >= in_shape[0]) fi0 = in_shape[0] - 1;
    size_m ci0 = fi0 + 1;    if (ci0 >= in_shape[0]) ci0 = in_shape[0] - 1;
    float frac0 = coord0 - f0;

    float coord1 = float(out_idx1) * scale[1];
    float f1 = floor(coord1);
    size_m fi1 = (size_m)f1; if (fi1 >= in_shape[1]) fi1 = in_shape[1] - 1;
    size_m ci1 = fi1 + 1;    if (ci1 >= in_shape[1]) ci1 = in_shape[1] - 1;
    float frac1 = coord1 - f1;

    float v00 = float(inMat[fi0 * in_strides[0] + fi1 * in_strides[1]]);
    float v01 = float(inMat[fi0 * in_strides[0] + ci1 * in_strides[1]]);
    float v10 = float(inMat[ci0 * in_strides[0] + fi1 * in_strides[1]]);
    float v11 = float(inMat[ci0 * in_strides[0] + ci1 * in_strides[1]]);

    float top = v00 * (1.0 - frac1) + v01 * frac1;
    float bot = v10 * (1.0 - frac1) + v11 * frac1;
    float sum = top * (1.0 - frac0) + bot * frac0;

    outMat[out_off] = (T)sum;
}

// 3D Linear (trilinear)
template <typename T>
kernel void resample_linear_nd3(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m* out_strides [[buffer(2)]],
    constant const size_m* in_strides [[buffer(3)]],
    constant const size_m* in_shape [[buffer(4)]],
    constant const float* scale [[buffer(5)]],
    uint3 gid [[thread_position_in_grid]])
{
    size_m out_idx0 = gid.z;
    size_m out_idx1 = gid.y;
    size_m out_idx2 = gid.x;
    size_m out_off = out_idx0 * out_strides[0] + out_idx1 * out_strides[1] + out_idx2 * out_strides[2];

    float coord0 = float(out_idx0) * scale[0];
    float f0 = floor(coord0);
    size_m fi0 = (size_m)f0; if (fi0 >= in_shape[0]) fi0 = in_shape[0] - 1;
    size_m ci0 = fi0 + 1;    if (ci0 >= in_shape[0]) ci0 = in_shape[0] - 1;
    float frac0 = coord0 - f0;

    float coord1 = float(out_idx1) * scale[1];
    float f1 = floor(coord1);
    size_m fi1 = (size_m)f1; if (fi1 >= in_shape[1]) fi1 = in_shape[1] - 1;
    size_m ci1 = fi1 + 1;    if (ci1 >= in_shape[1]) ci1 = in_shape[1] - 1;
    float frac1 = coord1 - f1;

    float coord2 = float(out_idx2) * scale[2];
    float f2 = floor(coord2);
    size_m fi2 = (size_m)f2; if (fi2 >= in_shape[2]) fi2 = in_shape[2] - 1;
    size_m ci2 = fi2 + 1;    if (ci2 >= in_shape[2]) ci2 = in_shape[2] - 1;
    float frac2 = coord2 - f2;

    float v000 = float(inMat[fi0 * in_strides[0] + fi1 * in_strides[1] + fi2 * in_strides[2]]);
    float v001 = float(inMat[fi0 * in_strides[0] + fi1 * in_strides[1] + ci2 * in_strides[2]]);
    float v010 = float(inMat[fi0 * in_strides[0] + ci1 * in_strides[1] + fi2 * in_strides[2]]);
    float v011 = float(inMat[fi0 * in_strides[0] + ci1 * in_strides[1] + ci2 * in_strides[2]]);
    float v100 = float(inMat[ci0 * in_strides[0] + fi1 * in_strides[1] + fi2 * in_strides[2]]);
    float v101 = float(inMat[ci0 * in_strides[0] + fi1 * in_strides[1] + ci2 * in_strides[2]]);
    float v110 = float(inMat[ci0 * in_strides[0] + ci1 * in_strides[1] + fi2 * in_strides[2]]);
    float v111 = float(inMat[ci0 * in_strides[0] + ci1 * in_strides[1] + ci2 * in_strides[2]]);

    float c00 = v000 * (1.0 - frac2) + v001 * frac2;
    float c01 = v010 * (1.0 - frac2) + v011 * frac2;
    float c10 = v100 * (1.0 - frac2) + v101 * frac2;
    float c11 = v110 * (1.0 - frac2) + v111 * frac2;

    float c0 = c00 * (1.0 - frac1) + c01 * frac1;
    float c1 = c10 * (1.0 - frac1) + c11 * frac1;

    float sum = c0 * (1.0 - frac0) + c1 * frac0;

    outMat[out_off] = (T)sum;
}

// ND Linear (generic, arbitrary rank): blends the 2^ndim corners of the
// enclosing box via a runtime bit-loop, same shape as ConvND's kernel-tap loop.
template <typename T>
kernel void resample_linear_gg(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m* out_strides [[buffer(2)]],
    constant const size_m* in_strides [[buffer(3)]],
    constant const size_m* in_shape [[buffer(4)]],
    constant const float* scale [[buffer(5)]],
    constant const size_m* out_shape [[buffer(6)]],
    constant const int& ndim [[buffer(7)]],
    uint3 gid [[thread_position_in_grid]])
{
    size_m out_idx[MAX_TENSOR_RANK];
    out_idx[ndim-1] = gid.x;
    if (ndim >= 2) out_idx[ndim-2] = gid.y;

    uint rem = gid.z;
    for (int i = ndim-3; i >= 0; --i) {
        out_idx[i] = rem % out_shape[i];
        rem /= out_shape[i];
    }

    size_m out_off = 0;
    size_m floor_idx[MAX_TENSOR_RANK];
    float frac[MAX_TENSOR_RANK];
    for (int i = 0; i < ndim; i++) {
        out_off += out_idx[i] * out_strides[i];
        float coord = float(out_idx[i]) * scale[i];
        float f = floor(coord);
        size_m fi = (size_m)f;
        if (fi >= in_shape[i]) fi = in_shape[i] - 1;
        floor_idx[i] = fi;
        frac[i] = coord - f;
    }

    uint num_corners = 1u << ndim;
    float sum = 0.0;
    for (uint corner = 0; corner < num_corners; corner++) {
        float weight = 1.0;
        size_m in_off = 0;
        for (int i = 0; i < ndim; i++) {
            bool bit = (corner >> i) & 1u;
            size_m idx = floor_idx[i];
            if (bit) {
                size_m ci = idx + 1;
                if (ci >= in_shape[i]) ci = in_shape[i] - 1;
                idx = ci;
                weight *= frac[i];
            } else {
                weight *= (1.0 - frac[i]);
            }
            in_off += idx * in_strides[i];
        }
        if (weight != 0.0) {
            sum += weight * float(inMat[in_off]);
        }
    }

    outMat[out_off] = (T)sum;
}

// =====================================================================
// "Tail" variants: for when the trailing axis is untouched by the resample
// (in_shape == out_shape along the last axis -- e.g. an RGBA channel axis).
// The grid spans only the LEADING axes (rank = ndim-1); each thread computes
// its leading-axis interpolation coordinates/weights once and then loops the
// identity axis internally, reusing them across every element instead of
// every thread along that axis redundantly recomputing the same floor/frac
// math and (for linear) re-reading the same "ceil" sample as the "floor" one.
// =====================================================================

// 1D leading axis, identity tail -- Nearest
template <typename T>
kernel void resample_nearest_tail_nd1(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m& out_stride [[buffer(2)]],
    constant const size_m& in_stride [[buffer(3)]],
    constant const size_m& in_shape [[buffer(4)]],
    constant const float& scale [[buffer(5)]],
    constant const size_m& tail_size [[buffer(6)]],
    constant const size_m& in_tail_stride [[buffer(7)]],
    constant const size_m& out_tail_stride [[buffer(8)]],
    uint index [[thread_position_in_grid]])
{
    size_m out_idx = index;
    float coord = float(out_idx) * scale;
    size_m in_idx = (size_m) round(coord);
    if (in_idx >= in_shape) in_idx = in_shape - 1;

    size_m out_off = out_idx * out_stride;
    size_m in_off = in_idx * in_stride;
    for (size_m c = 0; c < tail_size; c++) {
        outMat[out_off + c * out_tail_stride] = inMat[in_off + c * in_tail_stride];
    }
}

// 2D leading axes, identity tail -- Nearest
template <typename T>
kernel void resample_nearest_tail_nd2(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m* out_strides [[buffer(2)]],
    constant const size_m* in_strides [[buffer(3)]],
    constant const size_m* in_shape [[buffer(4)]],
    constant const float* scale [[buffer(5)]],
    constant const size_m& tail_size [[buffer(6)]],
    constant const size_m& in_tail_stride [[buffer(7)]],
    constant const size_m& out_tail_stride [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    size_m out_idx0 = gid.y;
    size_m out_idx1 = gid.x;
    size_m out_off = out_idx0 * out_strides[0] + out_idx1 * out_strides[1];

    size_m in_idx0 = (size_m) round(float(out_idx0) * scale[0]);
    if (in_idx0 >= in_shape[0]) in_idx0 = in_shape[0] - 1;
    size_m in_idx1 = (size_m) round(float(out_idx1) * scale[1]);
    if (in_idx1 >= in_shape[1]) in_idx1 = in_shape[1] - 1;
    size_m in_off = in_idx0 * in_strides[0] + in_idx1 * in_strides[1];

    for (size_m c = 0; c < tail_size; c++) {
        outMat[out_off + c * out_tail_stride] = inMat[in_off + c * in_tail_stride];
    }
}

// 3D leading axes, identity tail -- Nearest
template <typename T>
kernel void resample_nearest_tail_nd3(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m* out_strides [[buffer(2)]],
    constant const size_m* in_strides [[buffer(3)]],
    constant const size_m* in_shape [[buffer(4)]],
    constant const float* scale [[buffer(5)]],
    constant const size_m& tail_size [[buffer(6)]],
    constant const size_m& in_tail_stride [[buffer(7)]],
    constant const size_m& out_tail_stride [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    size_m out_idx0 = gid.z;
    size_m out_idx1 = gid.y;
    size_m out_idx2 = gid.x;
    size_m out_off = out_idx0 * out_strides[0] + out_idx1 * out_strides[1] + out_idx2 * out_strides[2];

    size_m in_idx0 = (size_m) round(float(out_idx0) * scale[0]);
    if (in_idx0 >= in_shape[0]) in_idx0 = in_shape[0] - 1;
    size_m in_idx1 = (size_m) round(float(out_idx1) * scale[1]);
    if (in_idx1 >= in_shape[1]) in_idx1 = in_shape[1] - 1;
    size_m in_idx2 = (size_m) round(float(out_idx2) * scale[2]);
    if (in_idx2 >= in_shape[2]) in_idx2 = in_shape[2] - 1;
    size_m in_off = in_idx0 * in_strides[0] + in_idx1 * in_strides[1] + in_idx2 * in_strides[2];

    for (size_m c = 0; c < tail_size; c++) {
        outMat[out_off + c * out_tail_stride] = inMat[in_off + c * in_tail_stride];
    }
}

// ND leading axes (generic, arbitrary rank), identity tail -- Nearest
template <typename T>
kernel void resample_nearest_tail_gg(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m* out_strides [[buffer(2)]],
    constant const size_m* in_strides [[buffer(3)]],
    constant const size_m* in_shape [[buffer(4)]],
    constant const float* scale [[buffer(5)]],
    constant const size_m* out_shape [[buffer(6)]],
    constant const int& ndim [[buffer(7)]],
    constant const size_m& tail_size [[buffer(8)]],
    constant const size_m& in_tail_stride [[buffer(9)]],
    constant const size_m& out_tail_stride [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    size_m out_idx[MAX_TENSOR_RANK];
    out_idx[ndim-1] = gid.x;
    if (ndim >= 2) out_idx[ndim-2] = gid.y;

    uint rem = gid.z;
    for (int i = ndim-3; i >= 0; --i) {
        out_idx[i] = rem % out_shape[i];
        rem /= out_shape[i];
    }

    size_m out_off = 0;
    size_m in_off = 0;
    for (int i = 0; i < ndim; i++) {
        out_off += out_idx[i] * out_strides[i];
        size_m in_idx = (size_m) round(float(out_idx[i]) * scale[i]);
        if (in_idx >= in_shape[i]) in_idx = in_shape[i] - 1;
        in_off += in_idx * in_strides[i];
    }

    for (size_m c = 0; c < tail_size; c++) {
        outMat[out_off + c * out_tail_stride] = inMat[in_off + c * in_tail_stride];
    }
}

// 1D leading axis, identity tail -- Linear
template <typename T>
kernel void resample_linear_tail_nd1(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m& out_stride [[buffer(2)]],
    constant const size_m& in_stride [[buffer(3)]],
    constant const size_m& in_shape [[buffer(4)]],
    constant const float& scale [[buffer(5)]],
    constant const size_m& tail_size [[buffer(6)]],
    constant const size_m& in_tail_stride [[buffer(7)]],
    constant const size_m& out_tail_stride [[buffer(8)]],
    uint index [[thread_position_in_grid]])
{
    size_m out_idx = index;
    float coord = float(out_idx) * scale;
    float f = floor(coord);
    size_m fi = (size_m)f;
    if (fi >= in_shape) fi = in_shape - 1;
    size_m ci = fi + 1;
    if (ci >= in_shape) ci = in_shape - 1;
    float frac = coord - f;

    size_m out_off = out_idx * out_stride;
    size_m fi_off = fi * in_stride;
    size_m ci_off = ci * in_stride;
    float w0 = 1.0 - frac;
    float w1 = frac;

    for (size_m c = 0; c < tail_size; c++) {
        size_m tail_off = c * in_tail_stride;
        float sum = float(inMat[fi_off + tail_off]) * w0 + float(inMat[ci_off + tail_off]) * w1;
        outMat[out_off + c * out_tail_stride] = (T)sum;
    }
}

// 2D leading axes, identity tail -- Linear (bilinear weights computed once, reused)
template <typename T>
kernel void resample_linear_tail_nd2(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m* out_strides [[buffer(2)]],
    constant const size_m* in_strides [[buffer(3)]],
    constant const size_m* in_shape [[buffer(4)]],
    constant const float* scale [[buffer(5)]],
    constant const size_m& tail_size [[buffer(6)]],
    constant const size_m& in_tail_stride [[buffer(7)]],
    constant const size_m& out_tail_stride [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]])
{
    size_m out_idx0 = gid.y;
    size_m out_idx1 = gid.x;
    size_m out_off = out_idx0 * out_strides[0] + out_idx1 * out_strides[1];

    float coord0 = float(out_idx0) * scale[0];
    float f0 = floor(coord0);
    size_m fi0 = (size_m)f0; if (fi0 >= in_shape[0]) fi0 = in_shape[0] - 1;
    size_m ci0 = fi0 + 1;    if (ci0 >= in_shape[0]) ci0 = in_shape[0] - 1;
    float frac0 = coord0 - f0;

    float coord1 = float(out_idx1) * scale[1];
    float f1 = floor(coord1);
    size_m fi1 = (size_m)f1; if (fi1 >= in_shape[1]) fi1 = in_shape[1] - 1;
    size_m ci1 = fi1 + 1;    if (ci1 >= in_shape[1]) ci1 = in_shape[1] - 1;
    float frac1 = coord1 - f1;

    size_m off00 = fi0 * in_strides[0] + fi1 * in_strides[1];
    size_m off01 = fi0 * in_strides[0] + ci1 * in_strides[1];
    size_m off10 = ci0 * in_strides[0] + fi1 * in_strides[1];
    size_m off11 = ci0 * in_strides[0] + ci1 * in_strides[1];

    float w00 = (1.0 - frac0) * (1.0 - frac1);
    float w01 = (1.0 - frac0) * frac1;
    float w10 = frac0 * (1.0 - frac1);
    float w11 = frac0 * frac1;

    for (size_m c = 0; c < tail_size; c++) {
        size_m tail_off = c * in_tail_stride;
        float sum = w00 * float(inMat[off00 + tail_off])
                  + w01 * float(inMat[off01 + tail_off])
                  + w10 * float(inMat[off10 + tail_off])
                  + w11 * float(inMat[off11 + tail_off]);
        outMat[out_off + c * out_tail_stride] = (T)sum;
    }
}

// 3D leading axes, identity tail -- Linear (trilinear weights computed once, reused)
template <typename T>
kernel void resample_linear_tail_nd3(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m* out_strides [[buffer(2)]],
    constant const size_m* in_strides [[buffer(3)]],
    constant const size_m* in_shape [[buffer(4)]],
    constant const float* scale [[buffer(5)]],
    constant const size_m& tail_size [[buffer(6)]],
    constant const size_m& in_tail_stride [[buffer(7)]],
    constant const size_m& out_tail_stride [[buffer(8)]],
    uint3 gid [[thread_position_in_grid]])
{
    size_m out_idx0 = gid.z;
    size_m out_idx1 = gid.y;
    size_m out_idx2 = gid.x;
    size_m out_off = out_idx0 * out_strides[0] + out_idx1 * out_strides[1] + out_idx2 * out_strides[2];

    float coord0 = float(out_idx0) * scale[0];
    float f0 = floor(coord0);
    size_m fi0 = (size_m)f0; if (fi0 >= in_shape[0]) fi0 = in_shape[0] - 1;
    size_m ci0 = fi0 + 1;    if (ci0 >= in_shape[0]) ci0 = in_shape[0] - 1;
    float frac0 = coord0 - f0;

    float coord1 = float(out_idx1) * scale[1];
    float f1 = floor(coord1);
    size_m fi1 = (size_m)f1; if (fi1 >= in_shape[1]) fi1 = in_shape[1] - 1;
    size_m ci1 = fi1 + 1;    if (ci1 >= in_shape[1]) ci1 = in_shape[1] - 1;
    float frac1 = coord1 - f1;

    float coord2 = float(out_idx2) * scale[2];
    float f2 = floor(coord2);
    size_m fi2 = (size_m)f2; if (fi2 >= in_shape[2]) fi2 = in_shape[2] - 1;
    size_m ci2 = fi2 + 1;    if (ci2 >= in_shape[2]) ci2 = in_shape[2] - 1;
    float frac2 = coord2 - f2;

    size_m off000 = fi0 * in_strides[0] + fi1 * in_strides[1] + fi2 * in_strides[2];
    size_m off001 = fi0 * in_strides[0] + fi1 * in_strides[1] + ci2 * in_strides[2];
    size_m off010 = fi0 * in_strides[0] + ci1 * in_strides[1] + fi2 * in_strides[2];
    size_m off011 = fi0 * in_strides[0] + ci1 * in_strides[1] + ci2 * in_strides[2];
    size_m off100 = ci0 * in_strides[0] + fi1 * in_strides[1] + fi2 * in_strides[2];
    size_m off101 = ci0 * in_strides[0] + fi1 * in_strides[1] + ci2 * in_strides[2];
    size_m off110 = ci0 * in_strides[0] + ci1 * in_strides[1] + fi2 * in_strides[2];
    size_m off111 = ci0 * in_strides[0] + ci1 * in_strides[1] + ci2 * in_strides[2];

    float w000 = (1.0 - frac0) * (1.0 - frac1) * (1.0 - frac2);
    float w001 = (1.0 - frac0) * (1.0 - frac1) * frac2;
    float w010 = (1.0 - frac0) * frac1 * (1.0 - frac2);
    float w011 = (1.0 - frac0) * frac1 * frac2;
    float w100 = frac0 * (1.0 - frac1) * (1.0 - frac2);
    float w101 = frac0 * (1.0 - frac1) * frac2;
    float w110 = frac0 * frac1 * (1.0 - frac2);
    float w111 = frac0 * frac1 * frac2;

    for (size_m c = 0; c < tail_size; c++) {
        size_m tail_off = c * in_tail_stride;
        float sum = w000 * float(inMat[off000 + tail_off])
                  + w001 * float(inMat[off001 + tail_off])
                  + w010 * float(inMat[off010 + tail_off])
                  + w011 * float(inMat[off011 + tail_off])
                  + w100 * float(inMat[off100 + tail_off])
                  + w101 * float(inMat[off101 + tail_off])
                  + w110 * float(inMat[off110 + tail_off])
                  + w111 * float(inMat[off111 + tail_off]);
        outMat[out_off + c * out_tail_stride] = (T)sum;
    }
}

// ND leading axes (generic, arbitrary rank), identity tail -- Linear.
// Precomputes each of the 2^ndim corners' input offset and weight once, then
// reuses that list across every element of the identity tail axis.
template <typename T>
kernel void resample_linear_tail_gg(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    constant const size_m* out_strides [[buffer(2)]],
    constant const size_m* in_strides [[buffer(3)]],
    constant const size_m* in_shape [[buffer(4)]],
    constant const float* scale [[buffer(5)]],
    constant const size_m* out_shape [[buffer(6)]],
    constant const int& ndim [[buffer(7)]],
    constant const size_m& tail_size [[buffer(8)]],
    constant const size_m& in_tail_stride [[buffer(9)]],
    constant const size_m& out_tail_stride [[buffer(10)]],
    uint3 gid [[thread_position_in_grid]])
{
    size_m out_idx[MAX_TENSOR_RANK];
    out_idx[ndim-1] = gid.x;
    if (ndim >= 2) out_idx[ndim-2] = gid.y;

    uint rem = gid.z;
    for (int i = ndim-3; i >= 0; --i) {
        out_idx[i] = rem % out_shape[i];
        rem /= out_shape[i];
    }

    size_m out_off = 0;
    size_m floor_idx[MAX_TENSOR_RANK];
    float frac[MAX_TENSOR_RANK];
    for (int i = 0; i < ndim; i++) {
        out_off += out_idx[i] * out_strides[i];
        float coord = float(out_idx[i]) * scale[i];
        float f = floor(coord);
        size_m fi = (size_m)f;
        if (fi >= in_shape[i]) fi = in_shape[i] - 1;
        floor_idx[i] = fi;
        frac[i] = coord - f;
    }

    // MAX_TENSOR_RANK-1 leading axes max -> at most 2^(MAX_TENSOR_RANK-1) corners.
    uint num_corners = 1u << ndim;
    float corner_weight[1 << (MAX_TENSOR_RANK - 1)];
    size_m corner_off[1 << (MAX_TENSOR_RANK - 1)];
    for (uint corner = 0; corner < num_corners; corner++) {
        float weight = 1.0;
        size_m off = 0;
        for (int i = 0; i < ndim; i++) {
            bool bit = (corner >> i) & 1u;
            size_m idx = floor_idx[i];
            if (bit) {
                size_m ci = idx + 1;
                if (ci >= in_shape[i]) ci = in_shape[i] - 1;
                idx = ci;
                weight *= frac[i];
            } else {
                weight *= (1.0 - frac[i]);
            }
            off += idx * in_strides[i];
        }
        corner_weight[corner] = weight;
        corner_off[corner] = off;
    }

    for (size_m c = 0; c < tail_size; c++) {
        size_m tail_off = c * in_tail_stride;
        float sum = 0.0;
        for (uint corner = 0; corner < num_corners; corner++) {
            if (corner_weight[corner] != 0.0) {
                sum += corner_weight[corner] * float(inMat[corner_off[corner] + tail_off]);
            }
        }
        outMat[out_off + c * out_tail_stride] = (T)sum;
    }
}

// Resample keeps the input dtype exactly (nearest is a plain copy; linear casts
// the float accumulator back to T), so it's instantiated across all 7 dtype codes,
// matching matrix::dtype's ordering (see matrix.h): 0=float,1=half,2=uint8_t,
// 3=int,4=int16_t,5=uint32_t,6=uint16_t.
#define INSTANTIATE_RESAMPLE_ND(type_idx, type) \
    instantiate_kernel("ResampleNearestGPU_nd_" #type_idx "_0", resample_nearest_nd1, type); \
    instantiate_kernel("ResampleNearestGPU_nd_" #type_idx "_1", resample_nearest_nd2, type); \
    instantiate_kernel("ResampleNearestGPU_nd_" #type_idx "_2", resample_nearest_nd3, type); \
    instantiate_kernel("ResampleNearestGPU_nd_" #type_idx "_3", resample_nearest_gg, type); \
    instantiate_kernel("ResampleLinearGPU_nd_" #type_idx "_0", resample_linear_nd1, type); \
    instantiate_kernel("ResampleLinearGPU_nd_" #type_idx "_1", resample_linear_nd2, type); \
    instantiate_kernel("ResampleLinearGPU_nd_" #type_idx "_2", resample_linear_nd3, type); \
    instantiate_kernel("ResampleLinearGPU_nd_" #type_idx "_3", resample_linear_gg, type); \
    instantiate_kernel("ResampleNearestTailGPU_nd_" #type_idx "_0", resample_nearest_tail_nd1, type); \
    instantiate_kernel("ResampleNearestTailGPU_nd_" #type_idx "_1", resample_nearest_tail_nd2, type); \
    instantiate_kernel("ResampleNearestTailGPU_nd_" #type_idx "_2", resample_nearest_tail_nd3, type); \
    instantiate_kernel("ResampleNearestTailGPU_nd_" #type_idx "_3", resample_nearest_tail_gg, type); \
    instantiate_kernel("ResampleLinearTailGPU_nd_" #type_idx "_0", resample_linear_tail_nd1, type); \
    instantiate_kernel("ResampleLinearTailGPU_nd_" #type_idx "_1", resample_linear_tail_nd2, type); \
    instantiate_kernel("ResampleLinearTailGPU_nd_" #type_idx "_2", resample_linear_tail_nd3, type); \
    instantiate_kernel("ResampleLinearTailGPU_nd_" #type_idx "_3", resample_linear_tail_gg, type);

INSTANTIATE_RESAMPLE_ND(0, float)
INSTANTIATE_RESAMPLE_ND(1, half)
INSTANTIATE_RESAMPLE_ND(2, uint8_t)
INSTANTIATE_RESAMPLE_ND(3, int)
INSTANTIATE_RESAMPLE_ND(4, int16_t)
INSTANTIATE_RESAMPLE_ND(5, uint32_t)
INSTANTIATE_RESAMPLE_ND(6, uint16_t)
