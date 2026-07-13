////
////  Matrix.m
////  AdityaIntelligenceProMax
////
////  Created by Aditya Dudeja on 22/10/25.
////
//
// #import <Foundation/Foundation.h>
// #import <Metal/Metal.h>
// #import <simd/simd.h>
#import <iostream>
@import Utils;
@import GPUManager;
#include "matrix.h"
#include "primitives.cpp"
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <arm_fp16.h>
#include <arm_neon.h>
#include <atomic>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#if !TARGET_OS_IPHONE
    #import <CoreServices/CoreServices.h>
#else
    #import <MobileCoreServices/MobileCoreServices.h>
#endif

//using R = AxisRange;

// enum class dtype {
//     Float = 0,
//     Float16 = 1,
//     UInt8 = 2,
//     Int32 = 3,
//     Int16 = 4,
//     UInt32 = 5,
//     UInt16 = 6
//     // Add more as needed
// };

// template<dtype code> struct type_from_dtype;
//
// template<> struct type_from_dtype<dtype::Float>   { using type = float; };
// template<> struct type_from_dtype<dtype::Float16>  { using type = float16_t;
// }; template<> struct type_from_dtype<dtype::UInt8>   { using type = uint8_t;
// }; template<> struct type_from_dtype<dtype::Int32>   { using type = int; };
// template<> struct type_from_dtype<dtype::Int16>   { using type = int16_t; };
// template<> struct type_from_dtype<dtype::UInt32>  { using type = uint32_t; };
// template<> struct type_from_dtype<dtype::UInt16>  { using type = uint16_t; };
//
//
// template<typename T> dtype dtype_from_type();
//
// template<> constexpr inline dtype dtype_from_type<float>()           { return
// dtype::Float; } template<> constexpr inline dtype
// dtype_from_type<float16_t>()       { return dtype::Float16; } template<>
// constexpr inline dtype dtype_from_type<uint8_t>()         { return
// dtype::UInt8; } template<> constexpr inline dtype dtype_from_type<int>() {
// return dtype::Int32; }
//
// template<> constexpr inline dtype dtype_from_type<int16_t>()           {
// return dtype::Int16; } template<> constexpr inline dtype
// dtype_from_type<uint32_t>()        { return dtype::UInt32; } template<>
// constexpr inline dtype dtype_from_type<uint16_t>()        { return
// dtype::UInt16; }
//
//
// constexpr size_t dtype_size(dtype d) {
//     //     switch (d) {
//     //     case dtype::Float:   return sizeof(float);
//     //     case dtype::Float16: return sizeof(float16_t);
//     //     case dtype::UInt8:   return sizeof(uint8_t);
//     //     case dtype::Int32:   return sizeof(int32_t);
//     //     case dtype::Int16:   return sizeof(int16_t);
//     //     case dtype::UInt16:  return sizeof(uint16_t);
//     //     case dtype::UInt32:  return sizeof(uint32_t);
//     // }
//     static constexpr size_t sizes[] = {
//         sizeof(float),     // 0: Float
//         sizeof(float16_t), // 1: Float16
//         sizeof(uint8_t),   // 2: UInt8
//         sizeof(int32_t),   // 3: Int32
//         sizeof(int16_t),   // 4: Int16
//         sizeof(uint32_t),  // 5: UInt32
//         sizeof(uint16_t)   // 6: UInt16
//     };
//     return sizes[static_cast<size_t>(d)];
// }

dtype promote_types(dtype a, dtype b) {
    return type_rules[static_cast<int>(a)][static_cast<int>(b)];
}
//
// struct SharedArrayDescriptor {
//     std::atomic<uint32_t> refCount;
//inline size_m *SharedArrayDescriptor::shape() {
//    return reinterpret_cast<size_m *>(this + 1);
//}
//inline size_m *SharedArrayDescriptor::strides(int dims) {
//    return shape() + dims;
//}
//SharedArrayDescriptor* SharedArrayDescriptor::retain() {
//    refCount.fetch_add(1, std::memory_order_relaxed);
//    return this;
//}

//    static SharedArrayDescriptor* create(uint32_t dims) {
//        size_t total_bytes = sizeof(SharedArrayDescriptor) + (dims * 2 *
//        sizeof(size_m)); void* mem = ::operator new(total_bytes);
//        SharedArrayDescriptor* shared = new (mem) SharedArrayDescriptor();
//        shared->refCount.store(1, std::memory_order_relaxed);
//        return shared;
//    }

void SharedArrayDescriptor::release() {
    if (refCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        this->~SharedArrayDescriptor();
        ::operator delete(this);
    }
}
//};

inline size_m *BroadcastDescriptor::shape() {
    return reinterpret_cast<size_m *>(this + 1);
}
inline size_m *BroadcastDescriptor::strides(int dims) {
    return reinterpret_cast<size_m *>(this + 1) + dims;
}

void broadcast_shapes(const array_descriptor &arr_desc1,
                      const array_descriptor &arr_desc2,
                      array_descriptor &out_shape,
                      BroadcastDescriptor *new_desc1,
                      BroadcastDescriptor *new_desc2, int dim1, int dim2) {
    int out_dim = std::max(dim1, dim2);
    
    if (out_dim > SBO_MAX_DIMS) {
        out_shape.shared_arr_desc = SharedArrayDescriptor::create(out_dim);
    }
    assert(new_desc1 && "broadcast_shapes: new_desc1 must not be null");
    assert(new_desc2 && "broadcast_shapes: new_desc2 must not be null");
    
    const size_m *shape1 = (dim1 > SBO_MAX_DIMS)
    ? arr_desc1.shared_arr_desc->shape()
    : arr_desc1.inline_buffer;
    const size_m *shape2 = (dim2 > SBO_MAX_DIMS)
    ? arr_desc2.shared_arr_desc->shape()
    : arr_desc2.inline_buffer;
    size_m *shape_out = (out_dim > SBO_MAX_DIMS)
    ? out_shape.shared_arr_desc->shape()
    : out_shape.inline_buffer;
    size_m *strides_out = (out_dim > SBO_MAX_DIMS)
    ? out_shape.shared_arr_desc->strides(out_dim)
    : out_shape.inline_buffer + SBO_MAX_DIMS;
    const size_m *strides1 = (dim1 > SBO_MAX_DIMS)
    ? arr_desc1.shared_arr_desc->strides(dim1)
    : arr_desc1.inline_buffer + SBO_MAX_DIMS;
    const size_m *strides2 = (dim2 > SBO_MAX_DIMS)
    ? arr_desc2.shared_arr_desc->strides(dim2)
    : arr_desc2.inline_buffer + SBO_MAX_DIMS;
    
    memcpy(new_desc1->strides(out_dim) + (out_dim - dim1), strides1,
           dim1 * sizeof(size_m));
    memcpy(new_desc2->strides(out_dim) + (out_dim - dim2), strides2,
           dim2 * sizeof(size_m));
    memset(new_desc1->strides(out_dim), 0, (out_dim - dim1) * sizeof(size_m));
    memset(new_desc2->strides(out_dim), 0, (out_dim - dim2) * sizeof(size_m));
    memcpy(new_desc1->shape() + (out_dim - dim1), shape1, dim1 * sizeof(size_m));
    memcpy(new_desc2->shape() + (out_dim - dim2), shape2, dim2 * sizeof(size_m));
    
    for (int i = 0; i < out_dim; i++) {
        // dims - i-1 < 0
        if (dim1 < i + 1) {
            new_desc1->shape()[out_dim - i - 1] = shape2[dim2 - i - 1];
            shape_out[out_dim - i - 1] = shape2[dim2 - i - 1];
        } else if (dim2 < i + 1) {
            new_desc2->shape()[out_dim - i - 1] = shape1[dim1 - i - 1];
            shape_out[out_dim - i - 1] = shape1[dim1 - i - 1];
        } else if (shape1[dim1 - i - 1] != shape2[dim2 - i - 1]) {
            if (shape1[dim1 - i - 1] == 1) {
                new_desc1->shape()[out_dim - i - 1] = shape2[dim2 - i - 1];
                shape_out[out_dim - i - 1] = shape2[dim2 - i - 1];
                new_desc1->strides(out_dim)[out_dim - i - 1] = 0;
            } else if (shape2[dim2 - i - 1] == 1) {
                new_desc2->shape()[out_dim - i - 1] = shape1[dim1 - i - 1];
                shape_out[out_dim - i - 1] = shape1[dim1 - i - 1];
                new_desc2->strides(out_dim)[out_dim - i - 1] = 0;
            } else {
                throw std::invalid_argument(
                                            "MatrixH: Incompatible shapes for broadcasting");
            }
        } else {
            shape_out[out_dim - i - 1] = shape1[dim1 - i - 1];
        }
    }
    size_m acc = 1;
    for (int i = out_dim - 1; i >= 0; i--) {
        strides_out[i] = acc;
        acc *= shape_out[i];
    }
}

void broadcast_shapes_matmul(const array_descriptor &arr_desc1,
                             const array_descriptor &arr_desc2,
                             array_descriptor &out_shape,
                             BroadcastDescriptor *new_desc1,
                             BroadcastDescriptor *new_desc2, int dim1, int dim2) {
    int out_dim = std::max(dim1, dim2);
    
    if (out_dim > SBO_MAX_DIMS) {
        out_shape.shared_arr_desc = SharedArrayDescriptor::create(out_dim);
    }
    assert(new_desc1 && "broadcast_shapes: new_desc1 must not be null");
    assert(new_desc2 && "broadcast_shapes: new_desc2 must not be null");
    
    const size_m *shape1 = (dim1 > SBO_MAX_DIMS)
    ? arr_desc1.shared_arr_desc->shape()
    : arr_desc1.inline_buffer;
    const size_m *shape2 = (dim2 > SBO_MAX_DIMS)
    ? arr_desc2.shared_arr_desc->shape()
    : arr_desc2.inline_buffer;
    size_m *shape_out = (out_dim > SBO_MAX_DIMS)
    ? out_shape.shared_arr_desc->shape()
    : out_shape.inline_buffer;
    size_m *strides_out = (out_dim > SBO_MAX_DIMS)
    ? out_shape.shared_arr_desc->strides(out_dim)
    : out_shape.inline_buffer + SBO_MAX_DIMS;
    const size_m *strides1 = (dim1 > SBO_MAX_DIMS)
    ? arr_desc1.shared_arr_desc->strides(dim1)
    : arr_desc1.inline_buffer + SBO_MAX_DIMS;
    const size_m *strides2 = (dim2 > SBO_MAX_DIMS)
    ? arr_desc2.shared_arr_desc->strides(dim2)
    : arr_desc2.inline_buffer + SBO_MAX_DIMS;
    
//    array_desc1 => [....., M, K]
//    array_desc2 => [....., N, K] from B.T
    shape_out[out_dim - 2] = shape1[dim1 - 2]; // M
    shape_out[out_dim - 1] = shape2[dim2 - 2]; // N
    
    memcpy(new_desc1->strides(out_dim) + (out_dim - dim1), strides1,
           dim1 * sizeof(size_m));
    memcpy(new_desc2->strides(out_dim) + (out_dim - dim2), strides2,
           dim2 * sizeof(size_m));
    memset(new_desc1->strides(out_dim), 0, (out_dim - dim1) * sizeof(size_m));
    memset(new_desc2->strides(out_dim), 0, (out_dim - dim2) * sizeof(size_m));
    memcpy(new_desc1->shape() + (out_dim - dim1), shape1, dim1 * sizeof(size_m));
    memcpy(new_desc2->shape() + (out_dim - dim2), shape2, dim2 * sizeof(size_m));
    
    for (int i = 2; i < out_dim; i++) {
        // dims - i-1 < 0
        if (dim1 < i + 1) {
            new_desc1->shape()[out_dim - i - 1] = shape2[dim2 - i - 1];
            shape_out[out_dim - i - 1] = shape2[dim2 - i - 1];
        } else if (dim2 < i + 1) {
            new_desc2->shape()[out_dim - i - 1] = shape1[dim1 - i - 1];
            shape_out[out_dim - i - 1] = shape1[dim1 - i - 1];
        } else if (shape1[dim1 - i - 1] != shape2[dim2 - i - 1]) {
            if (shape1[dim1 - i - 1] == 1) {
                new_desc1->shape()[out_dim - i - 1] = shape2[dim2 - i - 1];
                shape_out[out_dim - i - 1] = shape2[dim2 - i - 1];
                new_desc1->strides(out_dim)[out_dim - i - 1] = 0;
            } else if (shape2[dim2 - i - 1] == 1) {
                new_desc2->shape()[out_dim - i - 1] = shape1[dim1 - i - 1];
                shape_out[out_dim - i - 1] = shape1[dim1 - i - 1];
                new_desc2->strides(out_dim)[out_dim - i - 1] = 0;
            } else {
                throw std::invalid_argument(
                                            "MatrixH: Incompatible shapes for broadcasting");
            }
        } else {
            shape_out[out_dim - i - 1] = shape1[dim1 - i - 1];
        }
    }
    
    size_m acc = 1;
    for (int i = out_dim - 1; i >= 0; i--) {
        strides_out[i] = acc;
        acc *= shape_out[i];
    }
}




matrix matrix::broadcast_to_impl(const size_m* broadcasted_shape, int broadcasted_dim) const {
    matrix out_mat(broadcasted_dim, type);

    const size_m* src_shape   = shape();
    const size_m* src_strides = strides();
    size_m* dst_strides       = out_mat.strides();

    for (int i = 0; i < broadcasted_dim; i++) {
        int src_i = dims - i - 1;
        int dst_i = broadcasted_dim - i - 1;

        if (dims < i + 1) {
            dst_strides[dst_i] = 0;
        } else if (src_shape[src_i] == broadcasted_shape[dst_i]) {
            dst_strides[dst_i] = src_strides[src_i];
        } else if (src_shape[src_i] == 1) {
            dst_strides[dst_i] = 0;
        } else {
            throw std::invalid_argument("matrix: Incompatible shapes for broadcasting");
        }
    }
    memcpy(out_mat.shape(), broadcasted_shape, broadcasted_dim * sizeof(size_m));
    
    
//    if (tape) {
    out_mat.tape = new BrodcastPrimitive(const_cast<matrix&>(*this), out_mat.array_desc, broadcasted_dim);
//    }
    out_mat.total_size = out_mat.accumul(0, broadcasted_dim);
    out_mat.flags = flags;
    out_mat.flags |= NON_CONTIGUOUS_FLAG;
    return out_mat;
}

matrix matrix::broadcast_toV2(size_m* broadcasted_shape, int broadcasted_dim) const {
    return broadcast_to_impl(broadcasted_shape, broadcasted_dim);
}

matrix matrix::broadcast_toV2(std::initializer_list<size_m> shape) const {
    return broadcast_to_impl(shape.begin(), shape.size());
}
template <typename... Args> requires (std::convertible_to<Args, size_m> && ...)
matrix matrix::reshape(Args... newShape) {
    constexpr size_t newDims = sizeof...(newShape);
    matrix output(newDims, type);
    memcpy(output.shape(), (size_m[]){static_cast<size_m>(newShape)...}, sizeof(size_m) * newDims);
    output.total_size = output.accumul(0, newDims);
    if (output.total_size != total_size) {
        std::cerr << "Matrix: Reshape total size mismatch. Original size: " << total_size << ", New size: " << output.total_size << "\n";
        throw;
    }
    output.calcStrides();
    output.flags = flags;
    output.flags &= ~NON_CONTIGUOUS_FLAG;
    
    output.total_size = total_size;

    bool REQUIRES_NEW_BUFFER = (flags & NON_CONTIGUOUS_FLAG) != 0;
    if (REQUIRES_NEW_BUFFER) {
        output.flags &= ~NON_OWNERSHIP_FLAG;
    }
    output.tape = new ReshapePrimitive(*this, output.array_desc, output.dims, REQUIRES_NEW_BUFFER);
    return output;
}

template matrix matrix::reshape(size_m);
template matrix matrix::reshape(size_m, size_m);
template matrix matrix::reshape(size_m, size_m, size_m);
template matrix matrix::reshape(size_m, size_m, size_m, size_m);
template matrix matrix::reshape(size_m, size_m, size_m, size_m, size_m);
template matrix matrix::reshape(size_m, size_m, size_m, size_m, size_m, size_m);

template matrix matrix::reshape(int);
template matrix matrix::reshape(int, int);
template matrix matrix::reshape(int, int, int);
template matrix matrix::reshape(int, int, int, int);
template matrix matrix::reshape(int, int, int, int, int);

void matrix::reshape_eval(matrix& output, array_descriptor reshape_desc, size_t reshape_dims, ExecutionDevice execDev) {

    // Create a temporary view of output that perfectly matches the input's shape
    // This is necessary because copyGPUinplace and copyCPUinplace rely on collapse_dims
    // which assumes the number of dimensions in inMat and outMat match.
    matrix output_view(dims, type);
    memcpy(output_view.shape(), shape(), dims * sizeof(size_m));
    output_view.calcStrides();
    output.shareBuffer(output_view);
    output_view.metalBuffer = output.metalBuffer;
    output_view.total_size = total_size;
    output_view.flags = output.flags;
    
    if (execDev == ExecutionDevice::AUTO) {
        execDev = (total_size > 10) ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    if (execDev == ExecutionDevice::METAL) {
        copyGPUinplace(output_view, *this, 0, Execution::Encode);
    } else {
        copyCPUinplace(output_view, *this, 0);
    }
    
}

matrix matrix::reshape(array_descriptor reshape_desc, size_t reshape_dims) {
    matrix output(reshape_dims, type);
    output.set_array_desc(reshape_desc, reshape_dims);
    output.dims = reshape_dims;
    output.total_size = output.accumul(0, reshape_dims);
    
    if (output.total_size != total_size) {
        throw std::runtime_error(
            "Matrix: Reshape total size mismatch. Original: " +
            std::to_string(total_size) + ", New: " +
            std::to_string(output.total_size)
        );
    }
    
    output.calcStrides();
    
    output.flags = flags;
    output.flags &= ~NON_CONTIGUOUS_FLAG;
    

//    if (!(flags & NON_CONTIGUOUS_FLAG)) {
//        output.buffer = buffer;
//        output.metalBuffer = metalBuffer;
//        if (refCount) {
//            refCount->fetch_add(1);
//        }
//    }
    bool REQUIRES_NEW_BUFFER = (flags & NON_CONTIGUOUS_FLAG) != 0;
    output.tape = new ReshapePrimitive(*this, output.array_desc, output.dims, REQUIRES_NEW_BUFFER);
    
    return output;
}


matrix matrix::unsqueeze(int axis) const {
    if (axis < 0) axis += dims+1;
    matrix output(dims+1, type);
    memcpy(output.shape(), shape(), axis * sizeof(size_m));
    output.shape()[axis]=1;
    memcpy(output.shape() + axis + 1, shape() + axis, (dims-axis) * sizeof(size_m));
    if (flags & NON_CONTIGUOUS_FLAG) {
        memcpy(output.strides(), strides(), axis * sizeof(size_m));
        output.strides()[axis] = (axis < dims) ? strides()[axis] : 1; 
        memcpy(output.strides() + axis + 1, strides() + axis, (dims-axis) * sizeof(size_m));
        output.flags = flags;
        output.flags &= ~NON_OWNERSHIP_FLAG;
    } else {
        output.calcStrides();
        output.flags = flags;
        output.flags &= ~NON_CONTIGUOUS_FLAG;
        output.flags &= ~NON_OWNERSHIP_FLAG;
    }
    output.total_size = total_size;
//    output.refCount = refCount;

//    if (!(flags & NON_CONTIGUOUS_FLAG)) {
//        output.buffer = buffer;
//        output.metalBuffer = metalBuffer;
//        if (refCount) {
//            refCount->fetch_add(1);
//        }
//    }
    output.tape = new ReshapePrimitive(const_cast<matrix&>(*this), output.array_desc, output.dims, false);
    return output;
}

matrix matrix::squeeze(int axis) const {
    if (axis < 0) axis += dims;
    if (shape()[axis] != 1) {
        matrix output(dims, type);
        memcpy(output.shape(), shape(), dims * sizeof(size_m));
        if (flags & NON_CONTIGUOUS_FLAG) {
            memcpy(output.strides(), strides(), dims * sizeof(size_m));
            output.flags = flags;
            output.flags &= ~NON_OWNERSHIP_FLAG;
        } else {
            output.calcStrides();
            output.flags = flags;
            output.flags &= ~NON_CONTIGUOUS_FLAG;
            output.flags &= ~NON_OWNERSHIP_FLAG;
        }
        output.total_size = total_size;
        output.tape = new ReshapePrimitive(const_cast<matrix&>(*this), output.array_desc, output.dims, false);
        return output;
    }
    
    matrix output(dims - 1, type);
    if (axis > 0) {
        memcpy(output.shape(), shape(), axis * sizeof(size_m));
    }
    if (axis < dims - 1) {
        memcpy(output.shape() + axis, shape() + axis + 1, (dims - axis - 1) * sizeof(size_m));
    }
    if (flags & NON_CONTIGUOUS_FLAG) {
        if (axis > 0) {
            memcpy(output.strides(), strides(), axis * sizeof(size_m));
        }
        if (axis < dims - 1) {
            memcpy(output.strides() + axis, strides() + axis + 1, (dims - axis - 1) * sizeof(size_m));
        }
        output.flags = flags;
        output.flags &= ~NON_OWNERSHIP_FLAG;
    } else {
        output.calcStrides();
        output.flags = flags;
        output.flags &= ~NON_CONTIGUOUS_FLAG;
        output.flags &= ~NON_OWNERSHIP_FLAG;
    }
    output.total_size = total_size;
    output.tape = new ReshapePrimitive(const_cast<matrix&>(*this), output.array_desc, output.dims, false);
    return output;
}

matrix matrix::squeeze() const {
    int new_dims = 0;
    for (int i = 0; i < dims; i++) {
        if (shape()[i] != 1) new_dims++;
    }
    if (new_dims == dims) {
        return *this;
    }
    
    // In edge case where matrix is literally size 1 (e.g. {1,1}) we squeeze to 0D? Or 1D?
    // Let's assume minimum dimensionality might be allowed as 0D (if supported),
    // but if not supported, we should at least leave 1 dim.
    // Assuming 0D is supported in this codebase.
    matrix output(new_dims, type);
    int curr = 0;
    for (int i = 0; i < dims; i++) {
        if (shape()[i] != 1) {
            output.shape()[curr] = shape()[i];
//            if (flags & NON_CONTIGUOUS_FLAG) {
                output.strides()[curr] = strides()[i];
//            }
            curr++;
        }
    }
    
//    if (flags & NON_CONTIGUOUS_FLAG) {
        output.flags = flags;
////        output.flags &= ~NON_OWNERSHIP_FLAG;
//    } else {
//        output.calcStrides();
//        output.flags = flags;
//        output.flags &= ~NON_CONTIGUOUS_FLAG;
//        output.flags &= ~NON_OWNERSHIP_FLAG;
//    }
    output.total_size = total_size;
    output.tape = new ReshapePrimitive(const_cast<matrix&>(*this), output.array_desc, output.dims, false);
    return output;
}

matrix matrix::flatten(int start_dim, int end_dim) const {
    // Handle negative indexing
    if (end_dim < 0) end_dim += dims;
    if (start_dim < 0) start_dim += dims;
    
    int new_dims = dims - (end_dim - start_dim);
    matrix output(new_dims, type);
    
    // 1. Copy dimensions before start_dim
    if (start_dim > 0) {
        memcpy(output.shape(), shape(), start_dim * sizeof(size_m));
    }
    
    // 2. Collapse the specified range into a single dimension
    size_m collapsed_size = 1;
    for (int i = start_dim; i <= end_dim; i++) {
        collapsed_size *= shape()[i];
    }
    output.shape()[start_dim] = collapsed_size;
    
    // 3. Copy dimensions after end_dim
    if (end_dim < dims - 1) {
        memcpy(output.shape() + start_dim + 1, shape() + end_dim + 1, (dims - 1 - end_dim) * sizeof(size_m));
    }
    
    output.calcStrides();
    output.flags = flags;
    output.total_size = total_size;
    output.flags &= ~NON_CONTIGUOUS_FLAG;
    output.flags &= ~NON_OWNERSHIP_FLAG;
    bool REQUIRES_NEW_BUFFER = (flags & NON_CONTIGUOUS_FLAG) != 0;
    output.tape = new ReshapePrimitive(const_cast<matrix&>(*this), output.array_desc, output.dims, REQUIRES_NEW_BUFFER);
    
    return output;
}


matrix matrix::unsqueeze(uint32_t* axes, uint32_t num_axes) const {
    if (num_axes == 0) return *this;
    // normalize and sort axes (negative axis support optional)
    uint32_t new_dims = dims + num_axes;
    
    // sort axes (need them in order to insert correctly)
    uint32_t sorted_axes[num_axes];
    memcpy(sorted_axes, axes, num_axes * sizeof(uint32_t));
    std::sort(sorted_axes, sorted_axes + num_axes);
    
    matrix output(new_dims, type);
    
    // build new shape by walking both old shape and insertion points
    uint32_t old_i = 0;
    uint32_t new_i = 0;
    uint32_t ax_i = 0;
    
    while (new_i < new_dims) {
        if (ax_i < num_axes && sorted_axes[ax_i] == new_i) {
            output.shape()[new_i] = 1;
            ax_i++;
        } else {
            output.shape()[new_i] = shape()[old_i];
            old_i++;
        }
        new_i++;
    }
    
    
    old_i = 0;
    new_i = 0;
    ax_i = 0;
    while (new_i < new_dims) {
        if (ax_i < num_axes && sorted_axes[ax_i] == new_i) {
            output.strides()[new_i] = (old_i < dims) ? strides()[old_i] : 1;
            ax_i++;
        } else {
            output.strides()[new_i] = strides()[old_i];
            old_i++;
        }
        new_i++;
    }
    output.flags = flags;

    output.total_size = total_size;
    output.tape = new ReshapePrimitive(const_cast<matrix&>(*this), output.array_desc, output.dims, false);
    return output;
}

//template <typename Func>
//inline void dispatch_type(dtype type, void *buffer, Func &&function_to_run) {
//    if (!buffer)
//        return;
//    
//    // The switch statement happens exactly ONCE here.
//    // It casts the void* to the strict C++ type, and passes it into your lambda.
//    switch (type) {
//        case dtype::Float:
//            function_to_run(static_cast<float *>(buffer));
//            break;
//        case dtype::Float16:
//            // Standard C++ doesn't have a native 16-bit float yet on all compilers.
//            // We cast it to uint16_t for storage/printing purposes.
//            function_to_run(static_cast<uint16_t *>(buffer));
//            break;
//        case dtype::Int32:
//            function_to_run(static_cast<int32_t *>(buffer));
//            break;
//        case dtype::UInt32:
//            function_to_run(static_cast<uint32_t *>(buffer));
//            break;
//        case dtype::UInt8:
//            function_to_run(static_cast<uint8_t *>(buffer));
//            break;
//        case dtype::Int16:
//            function_to_run(static_cast<int16_t *>(buffer));
//            break;
//        case dtype::UInt16:
//            function_to_run(static_cast<uint16_t *>(buffer));
//            break;
//        default:
//            throw std::runtime_error("Unsupported dtype during dispatch");
//    }
//}

//// Detect if a type is an initializer_list
// template <typename T> struct is_init_list : std::false_type {};
// template <typename T> struct is_init_list<std::initializer_list<T>> :
// std::true_type {};
//
//// Count the depth of the nested initializer lists (Compile-time)
// template <typename T> struct init_list_depth { static constexpr int value =
// 0; }; template <typename T> struct init_list_depth<std::initializer_list<T>>
// {
//     static constexpr int value = 1 + init_list_depth<T>::value;
// };
//
//// Find the lowest-level base type (e.g., float, int) (Compile-time)
// template <typename T> struct init_list_base { using type = T; };
// template <typename T> struct init_list_base<std::initializer_list<T>> {
//     using type = typename init_list_base<T>::type;
// };

// class matrix {
// public:
//     static constexpr int SBO_MAX_DIMS = 3;
//
//     union array_descriptor {
//         size_m inline_buffer[SBO_MAX_DIMS * 2]; // For dims <= 3
//         SharedArrayDescriptor* shared_arr_desc;  // For dims > 3
//     } array_desc;
//
//
//     void* buffer = nullptr;
//     size_t total_size;
//     id<MTLBuffer> metalBuffer = nil;
//     std::atomic<uint32_t>* refCount = nil;
//     primitive* tape = nullptr;
//
//     dtype type;
//     uint32_t dims = 0;
//     uint8_t flags = 0;

//    template <typename Type>
//        dims = 1;
//        if (total_size > 10) {
//            buildMetalBuffer();
//        }
//    }

matrix::matrix(uint32_t rank, size_t total_size_inp, dtype type_inp) {
    dims = rank;
    total_size = total_size_inp;
    type = type_inp;
    
    if (dims > SBO_MAX_DIMS) {
        array_desc.shared_arr_desc = SharedArrayDescriptor::create(dims);
    }
    buffer = new uint8_t[total_size * dtype_size(type)];
    if (total_size_inp > 10) {
        buildMetalBuffer();
    }
}

matrix::matrix(uint32_t rank, dtype type_inp) {
    dims = rank;
    type = type_inp;
    
    if (dims > SBO_MAX_DIMS) {
        array_desc.shared_arr_desc = SharedArrayDescriptor::create(dims);
    }
}

//    template <typename Type>
//    matrix(std::initializer_list<std::initializer_list<Type>> list) {
//        setup_from_list<2, Type>(list);
//    }
//    template <typename Type>
//    matrix(std::initializer_list<std::initializer_list<std::initializer_list<Type>>>
//    list) {
//        setup_from_list<3, Type>(list);
//    }
//    template <typename Type>
//    matrix(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<Type>>>>
//    list) {
//        setup_from_list<4, Type>(list);
//    }

//// MARK: // --- Unified Initialization Logic ---
//template <int Dims, typename BaseType, typename NestedList>
//void matrix::setup_from_list(const NestedList &list) {
//    this->dims = Dims;
//    
//    type = dtype_from_type<BaseType>();
//    
//    // 2. Trigger SBO logic if dimensions exceed threshold
//    if (this->dims > SBO_MAX_DIMS) {
//        array_desc.shared_arr_desc = SharedArrayDescriptor::create(this->dims);
//    }
//    
//    // 3. Extract the shape dynamically
//    extract_shape(this->shape(), list, 0);
//    
//    // 4. Calculate strides and total size
//    this->total_size = 1;
//    for (int i = 0; i < this->dims; ++i) {
//        this->total_size *= this->shape()[i];
//    }
//    
//    size_m current_stride = 1;
//    for (int i = this->dims - 1; i >= 0; --i) {
//        this->strides()[i] = current_stride;
//        current_stride *= this->shape()[i];
//    }
//    
//    // 5. Allocate raw memory
//    this->buffer = new uint8_t[this->total_size * sizeof(BaseType)];
//    
//    // 6. Flatten and copy data
//    BaseType *raw_ptr = static_cast<BaseType *>(this->buffer);
//    size_m offset = 0;
//    copy_data(raw_ptr, list, offset);
//}
//
//// --- Recursive Helpers (from previous step) ---
//template <typename ListType>
//void matrix::extract_shape(size_m *shape_arr,
//                           const std::initializer_list<ListType> &list,
//                           int current_dim) {
//    shape_arr[current_dim] = static_cast<size_m>(list.size());
//    if constexpr (is_init_list<ListType>::value) {
//        if (list.size() > 0) {
//            extract_shape(shape_arr, *list.begin(), current_dim + 1);
//        }
//    }
//}
//
//template <typename ListType, typename BaseType>
//void matrix::copy_data(BaseType *dest,
//                       const std::initializer_list<ListType> &list,
//                       size_m &offset) {
//    if constexpr (is_init_list<ListType>::value) {
//        for (const auto &sub_list : list) {
//            copy_data(dest, sub_list, offset);
//        }
//    } else {
//        for (const auto &item : list) {
//            dest[offset++] = item;
//        }
//    }
//}


void matrix::shareBuffer(matrix &mat) const {
    if (mat.tape) {
        mat.tape->update_cache((uint8_t*)buffer, metalBuffer, refCount);
    }
    mat.releaseBuffer();
    mat.buffer = buffer;
    mat.metalBuffer = metalBuffer;
    if (refCount) {
        mat.refCount = refCount;
        refCount->fetch_add(1, std::memory_order_relaxed);
    } else {
        mat.flags |= NON_OWNERSHIP_FLAG;
    }
}

inline void matrix::update_from_trace() {
    if (tape && tape->out_buffer != buffer) {
        releaseBuffer();
        buffer = tape->out_buffer;
        metalBuffer = tape->out_metal_buffer;
        refCount = tape->out_refcount;
//        if (refCount) {
            refCount->fetch_add(1);
//        }
    }
}
inline void matrix::beginReferenceCounting() {
    refCount = new std::atomic<uint32_t>(1);
}
void matrix::calcStrides() {
    size_m *l_strides = strides();
    size_m *l_shape = shape();
    size_m acc = 1;
    for (int i = dims - 1; i >= 0; i--) {
        l_strides[i] = acc;
        acc *= l_shape[i];
    }
}

size_t matrix::accumul(uint32_t start, uint32_t end) const {
    const size_m *shape_arr = shape();
    size_t acc = 1;
    for (uint32_t i = start; i < end; i++) {
        acc *= shape_arr[i];
    }
    return acc;
}
size_t matrix::effectiveBufferSize() const {
    if (flags & NON_CONTIGUOUS_FLAG) {
        // CORRECT: Calculate the maximum memory offset generated by the strides
        size_t max_offset = 0;
        for (int d = 0; d < dims; ++d) {
            if (shape()[d] > 0) {
                // The max index for this dimension is (count - 1) * stride
                max_offset += (shape()[d] - 1) * strides()[d];
            }
        }
        // Buffer must be large enough to hold the last element
        return (max_offset + 1);
    } else {
        return total_size;
    }
}

inline void matrix::set_array_desc(const array_descriptor& new_array_desc) {
    const_cast<array_descriptor&>(new_array_desc).retain(dims);
    array_desc.release(dims);
    array_desc = new_array_desc;
}

inline void matrix::set_array_desc(const array_descriptor& new_array_desc, uint32_t new_dims) {
    const_cast<array_descriptor&>(new_array_desc).retain(new_dims);
    array_desc.release(dims);
    array_desc = new_array_desc;
}

matrix matrix::transpose(std::initializer_list<size_m> new_axis_order) const {
    assert(new_axis_order.size() == dims && "matrix: transpose => error: new_axis_order size must match matrix dimensions!");
    matrix output_view(dims, type);
    const size_m* input_shape = shape();
    const size_m* input_strides = strides();
    size_m* output_shape = output_view.shape();
    size_m* output_strides = output_view.strides();
//    shareBuffer(output_view);
    for (int i = 0; i < dims; i++) {
        int new_axii = *(new_axis_order.begin() + i);
        output_shape[i] = input_shape[new_axii];
        output_strides[i] = input_strides[new_axii];
    }
    output_view.total_size = total_size;
    output_view.flags = flags;
    output_view.flags |= NON_CONTIGUOUS_FLAG;
    output_view.tape = new TransposePrimitive(const_cast<matrix&>(*this), output_view.array_desc);
    return output_view;
}

matrix matrix::transpose(const std::vector<size_m>& new_axis_order) const {
    assert(new_axis_order.size() == dims && "matrix: transpose => error: new_axis_order size must match matrix dimensions!");
    matrix output_view(dims, type);
    const size_m* input_shape = shape();
    const size_m* input_strides = strides();
    size_m* output_shape = output_view.shape();
    size_m* output_strides = output_view.strides();
    for (int i = 0; i < dims; i++) {
        int new_axii = new_axis_order[i];
        output_shape[i] = input_shape[new_axii];
        output_strides[i] = input_strides[new_axii];
    }
    output_view.total_size = total_size;
    output_view.flags = flags | NON_CONTIGUOUS_FLAG;
    output_view.tape = new TransposePrimitive(const_cast<matrix&>(*this), output_view.array_desc);
    return output_view;
}


matrix matrix::T() const {
    matrix output_view(dims, type);
    const size_m* input_shape   = shape();
    const size_m* input_strides = strides();
    size_m* output_shape   = output_view.shape();
    size_m* output_strides = output_view.strides();
    for (int i = 0; i < dims; i++) {
        output_shape[i]   = input_shape[dims - 1 - i];
        output_strides[i] = input_strides[dims - 1 - i];
    }
    output_view.total_size = total_size;
    output_view.flags      = flags;
    output_view.flags     |= NON_CONTIGUOUS_FLAG;
    output_view.tape       = new TransposePrimitive(const_cast<matrix&>(*this), output_view.array_desc);
    return output_view;
}

matrix matrix::transpose(array_descriptor transpose_desc) const {
    matrix output_view(dims, type);
    output_view.set_array_desc(transpose_desc);
    output_view.total_size = total_size;
    output_view.flags = flags;
    output_view.flags |= NON_CONTIGUOUS_FLAG;
    output_view.tape = new TransposePrimitive(const_cast<matrix&>(*this), output_view.array_desc);
    return output_view;
}

std::function<matrix(matrix&)> matrix::jit_gpu(std::function<matrix(matrix&)> func, matrix& sample) {
    sample.begin_refcount();
    sample.tape = new SwapLeafPrimitive(sample);
    matrix output = func(sample);
    output.compile_metal();
    sample.releaseBuffer();
    return [&sample, output](matrix& input) mutable -> matrix {
        input.shareBuffer(sample);
        output.execute_metal();
        uint32_t val = output.refCount->load(std::memory_order_relaxed);

        printf("%u\n", val);
        matrix returnable_output(output.dims, output.type);
        returnable_output.set_array_desc(output.array_desc);
        output.shareBuffer(returnable_output);
        output.releaseBuffer();
        output.tape->out_buffer = nullptr;
        output.tape->out_refcount = nullptr;
        output.tape->out_metal_buffer = nullptr;
        output.clear_trace_checks();
        returnable_output.tape = nullptr;
        return returnable_output;
    };
}




std::function<std::vector<matrix>(const std::vector<matrix>&)> matrix::multi_jit_graph_gpu(std::function<std::vector<matrix>(std::vector<matrix>)> func, const std::vector<matrix>& sample_inputs) {
    std::vector<matrix> inner_samples = sample_inputs;

    for (auto& sample : inner_samples) {
        if (sample.tape) {
            sample.eval_metal();
        }
        sample.begin_refcount();
        sample.tape = new SwapLeafPrimitive(sample);
    }
    
    std::vector<matrix> inner_outputs = func(inner_samples);
    
    for (auto& o : inner_outputs) {
        o.compile_metal();
    }
    for (auto& sample : inner_samples) {
        sample.releaseBuffer();
    }
    
    return [inner_samples, inner_outputs](const std::vector<matrix>& inputs) mutable -> std::vector<matrix> {
        std::vector<matrix> mutable_inputs = const_cast<std::vector<matrix>&>(inputs);
        
        std::vector<matrix> outer_outputs;
        for (auto& o : inner_outputs) {
            matrix out(o.dims, o.type);
            out.set_array_desc(o.array_desc);
            out.total_size = o.total_size;
            out.flags = o.flags;
            outer_outputs.push_back(out);
        }
        
        std::vector<MultiInputCompilePrimitive*> prims;
        for (size_t i = 0; i < outer_outputs.size(); i++) {
            prims.push_back(new MultiInputCompilePrimitive(mutable_inputs, inner_samples, inner_outputs, outer_outputs, i));
        }
        
        // Patch the tape pointers on the local outer_outputs and the copies stored inside the primitives
        for (size_t i = 0; i < outer_outputs.size(); i++) {
            outer_outputs[i].tape = prims[i];
            for (auto p : prims) {
                p->outer_outputs[i].tape = prims[i];
                p->siblings = prims;
            }
        }
        
        return outer_outputs;
    };
}




std::function<matrix(const matrix&)> matrix::jit_graph_gpu(std::function<matrix(matrix)> func, matrix sample) {
    if (sample.tape) {
        sample.eval_metal();
    }
    sample.begin_refcount();
    sample.tape = new SwapLeafPrimitive(sample);
    matrix output = func(sample);
    
    output.compile_metal();
    sample.releaseBuffer();
    
    return [sample, output](const matrix& input) mutable -> matrix {
        matrix returnable_output(output.dims, output.type);
        returnable_output.set_array_desc(output.array_desc);
        returnable_output.total_size = output.total_size;
        returnable_output.flags = output.flags;
        matrix& mut_input = const_cast<matrix&>(input);
        returnable_output.tape = new CompiledNodePrimitive(mut_input, sample, output);
        return returnable_output;
    };
}


std::function<matrix(matrix)> matrix::grad_gpu(std::function<matrix(matrix)> func, matrix sample) {
    sample.begin_refcount();
    sample.tape = new SwapLeafPrimitive(sample);
    matrix output = func(sample);
    matrix grad_output = matrix::build_grad_graph(output, sample);
    grad_output.compile_metal();
    sample.releaseBuffer();
    
    return [sample, grad_output](matrix input) mutable -> matrix {
        input.shareBuffer(sample); // Swaps the buffer in SwapLeafPrimitive
        
        grad_output.execute_metal();
        matrix returnable_output(grad_output.dims, grad_output.type);
        returnable_output.set_array_desc(grad_output.array_desc);
        returnable_output.shareBuffer(grad_output);
        grad_output.releaseBuffer();
        grad_output.tape->update_cache(nullptr, nil, nullptr);
        returnable_output.tape = nullptr;
        
        grad_output.clear_trace_checks();
        
        return returnable_output;
    };
}

std::function<matrix(matrix)> matrix::grad_graph_gpu(std::function<matrix(matrix)> func, matrix sample) {
    if (sample.tape) {
        sample.eval_metal();
    }
    sample.begin_refcount();
    sample.tape = new SwapLeafPrimitive(sample);
    matrix output = func(sample);
    
    matrix grad_output = matrix::build_grad_graph(output, sample);
    grad_output.compile_metal();
    
    return [sample, grad_output](matrix input) mutable -> matrix {
        if (!input.tape) {
            input.begin_refcount();
        }
        matrix returnable_output(grad_output.dims, grad_output.type);
        returnable_output.set_array_desc(grad_output.array_desc);
        returnable_output.total_size = grad_output.total_size;
        returnable_output.flags = grad_output.flags;
        returnable_output.tape = new CompiledNodePrimitive(input, sample, grad_output);
        return returnable_output;
    };
}

matrix matrix::build_grad_graph(matrix& output_node, matrix& sample_input_node) {
    std::vector<matrix> topo;
    std::unordered_set<void*> visited;
    std::unordered_map<void*, matrix> grads;
    
    auto get_id = [](const matrix& m) -> void* {
        return m.tape ? m.tape : m.buffer;
    };
    
    std::function<void(matrix&)> dfs = [&](matrix& m) {
        void* uuid = get_id(m);
        if (visited.count(uuid)) return;
        visited.insert(uuid);
        if (m.tape) {
            for (auto& inp : m.tape->get_inputs()) {
                dfs(inp);
            }
        }
        topo.push_back(m);
    };
    
    dfs(output_node);
    
    grads.insert({get_id(output_node), output_node.ones()});
    
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        matrix& m = *it;
        if (!m.tape) { continue; }
        matrix current_grad = grads.at(get_id(m));
        
        auto input_grads = m.tape->vjp(current_grad);
        auto inputs = m.tape->get_inputs();
        
        for (int i = 0; i < inputs.size(); i++) {
            void* in_id = get_id(inputs[i]);
            auto it = grads.find(in_id);
            if (it != grads.end()) {
                // Already exists, accumulate!
                it->second = it->second + input_grads[i];
            } else {
                // Doesn't exist, safely insert it!
                grads.insert({in_id, input_grads[i]});
            }
        }
    }
    auto it = grads.find(get_id(sample_input_node));
    if (it != grads.end()) {
        return it->second;
    }
    return sample_input_node.zeros();
}

std::vector<matrix> matrix::build_grad_graph(std::vector<matrix>& output_nodes, std::vector<matrix>& sample_input_nodes) {
    std::vector<matrix> topo;
    std::unordered_set<void*> visited;
    std::unordered_map<void*, matrix> grads;
    
    auto get_id = [](const matrix& m) -> void* {
        return m.tape ? m.tape : m.buffer;
    };
    
    std::function<void(matrix&)> dfs = [&](matrix& m) {
        void* uuid = get_id(m);
        if (visited.count(uuid)) return;
        visited.insert(uuid);
        if (m.tape) {
            for (auto& inp : m.tape->get_inputs()) {
                dfs(inp);
            }
        }
        topo.push_back(m);
    };
    
    for (auto& output_node : output_nodes) {
        dfs(output_node);
        grads.insert({get_id(output_node), output_node.ones()});
    }
    
    
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        matrix& m = *it;
        if (!m.tape) { continue; }
        matrix current_grad = grads.at(get_id(m));
        
        auto input_grads = m.tape->vjp(current_grad);
        auto inputs = m.tape->get_inputs();
        
        for (int i = 0; i < inputs.size(); i++) {
            void* in_id = get_id(inputs[i]);
            auto it = grads.find(in_id);
            if (it != grads.end()) {
                // Already exists, accumulate!
                it->second = it->second + input_grads[i];
            } else {
                // Doesn't exist, safely insert it!
                grads.insert({in_id, input_grads[i]});
            }
        }
    }
    std::vector<matrix> grad_out;
    
    for (auto& sample_input_node: sample_input_nodes) {
        auto it = grads.find(get_id(sample_input_node));
        if (it != grads.end()) {
            grad_out.push_back(it->second);
        } else {
            // If it wasn't used in the graph, the gradient is just 0 but was part of the function to be derivated like f(x,y)=x^2 ; y isnt used so not part of graph but part of function
            grad_out.push_back(sample_input_node.zeros());
        }
    }
    return grad_out;
}


std::function<std::vector<matrix>(const std::vector<matrix>)> matrix::grad_graph_gpu(std::function<std::vector<matrix>(std::vector<matrix>)> func, const std::vector<matrix>& sample_inputs) {
    
    std::vector<matrix> inner_samples = sample_inputs;

    for (auto& sample : inner_samples) {
        if (sample.tape) {
            sample.eval_metal();
        }
        sample.begin_refcount();
        sample.tape = new SwapLeafPrimitive(sample);
    }
    
    std::vector<matrix> inner_outputs = func(inner_samples);
    std::vector<matrix> grad_outputs = build_grad_graph(inner_outputs, inner_samples);
    
    for (auto& o : grad_outputs) {
        o.compile_metal();
    }
    for (auto& sample : inner_samples) {
        sample.releaseBuffer();
    }
    
    return [inner_samples, grad_outputs](const std::vector<matrix>& inputs) mutable -> std::vector<matrix> {
        std::vector<matrix> mutable_inputs = inputs;
        for (auto& i: mutable_inputs){
            if (i.buffer) {
                i.begin_refcount();
            }
        }
        
        std::vector<matrix> grad_outer_outputs;
        for (auto& o : grad_outputs) {
            matrix out(o.dims, o.type);
            out.set_array_desc(o.array_desc);
            out.total_size = o.total_size;
            out.flags = o.flags;
            grad_outer_outputs.push_back(out);
        }
        
        std::vector<MultiInputCompilePrimitive*> prims;
        for (size_t i = 0; i < grad_outer_outputs.size(); i++) {
            prims.push_back(new MultiInputCompilePrimitive(mutable_inputs, inner_samples, grad_outputs, grad_outer_outputs, i));
        }
        
        // Patch the tape pointers on the local outer_outputs and the copies stored inside the primitives
        for (size_t i = 0; i < grad_outer_outputs.size(); i++) {
            grad_outer_outputs[i].tape = prims[i];
            for (auto p : prims) {
                p->outer_outputs[i].tape = prims[i];
                p->siblings = prims;
            }
        }
        
        return grad_outer_outputs;
    };
}

//    static matrix matrix::withShape(std::initializer_list<size_m> shape, dtype
//    type) {
//        matrix output(shape.size(), type);
//        memcpy(output.shape(), shape.begin(), output.dims * sizeof(size_m));
//        output.calcStrides();
//        output.total_size = output.accumul(0, output.dims);
//        output.buffer = new uint8_t[output.total_size * dtype_size(type)];
//        if (output.total_size > 10) {
//            output.buildMetalBuffer();
//        }
//        return output;
//    }

void matrix::print() const {
    const_cast<matrix&>(*this).update_from_trace();
    const_cast<matrix&>(*this).eval();
    if (!buffer)
        return;
    
    
    // The single dispatch call that handles the type erasure
    dispatch_type(this->type, this->buffer, [&](auto *typed_data) {
        // Extract the exact C++ type we are currently working with
        using T = std::decay_t<decltype(*typed_data)>;
        const size_m *s = shape();
        const size_m *st = strides();
        // 1. PRE-PASS: Find the maximum character width for alignment
        uint32_t max_width = 0;
        if (!(flags & NON_CONTIGUOUS_FLAG)) {
            for (size_t i = 0; i < total_size; ++i) {
                std::ostringstream oss;
                // Treat chars as numbers for printing
                if constexpr (std::is_same_v<T, char> ||
                              std::is_same_v<T, unsigned char> ||
                              std::is_same_v<T, int8_t>) {
                    oss << static_cast<int>(typed_data[i]);
                } else {
                    oss << typed_data[i];
                }
                
                uint32_t current_length = (uint32_t)oss.str().length();
                if (current_length > max_width)
                    max_width = current_length;
            }
            
        } else {
            
            if (dims == 1) {
                for (size_t i = 0; i < s[0]; i++) {
                    std::ostringstream oss;
                    if constexpr (std::is_same_v<T, char> ||
                                  std::is_same_v<T, unsigned char> ||
                                  std::is_same_v<T, int8_t>) {
                        oss << static_cast<int>(typed_data[st[0]*i]);
                    } else {
                        oss << typed_data[st[0]*i];
                    }
                    uint32_t l = (uint32_t)oss.str().length();
                    if (l > max_width) max_width = l;
                }

            } else if (dims == 2) {
                for (size_t i = 0; i < s[0]; i++)
                for (size_t j = 0; j < s[1]; j++) {
                    std::ostringstream oss;
                    if constexpr (std::is_same_v<T, char> ||
                                  std::is_same_v<T, unsigned char> ||
                                  std::is_same_v<T, int8_t>) {
                        oss << static_cast<int>(typed_data[st[0]*i + st[1]*j]);
                    } else {
                        oss << typed_data[st[0]*i + st[1]*j];
                    }
                    uint32_t l = (uint32_t)oss.str().length();
                    if (l > max_width) max_width = l;
                }

            } else if (dims == 3) {
                for (size_t i = 0; i < s[0]; i++)
                for (size_t j = 0; j < s[1]; j++)
                for (size_t k = 0; k < s[2]; k++) {
                    std::ostringstream oss;
                    if constexpr (std::is_same_v<T, char> ||
                                  std::is_same_v<T, unsigned char> ||
                                  std::is_same_v<T, int8_t>) {
                        oss << static_cast<int>(typed_data[st[0]*i + st[1]*j + st[2]*k]);
                    } else {
                        oss << typed_data[st[0]*i + st[1]*j + st[2]*k];
                    }
                    uint32_t l = (uint32_t)oss.str().length();
                    if (l > max_width) max_width = l;
                }

            } else if (dims == 4) {
                for (size_t l = 0; l < s[0]; l++)
                for (size_t i = 0; i < s[1]; i++)
                for (size_t j = 0; j < s[2]; j++)
                for (size_t k = 0; k < s[3]; k++) {
                    std::ostringstream oss;
                    if constexpr (std::is_same_v<T, char> ||
                                  std::is_same_v<T, unsigned char> ||
                                  std::is_same_v<T, int8_t>) {
                        oss << static_cast<int>(typed_data[st[0]*l + st[1]*i + st[2]*j + st[3]*k]);
                    } else {
                        oss << typed_data[st[0]*l + st[1]*i + st[2]*j + st[3]*k];
                    }
                    uint32_t len = (uint32_t)oss.str().length();
                    if (len > max_width) max_width = len;
                }
            }
        }
        

        
        // Tiny helper to print a single element with correct width and casting
        auto print_elem = [&](size_t flat_index) {
            std::cout << std::setw(max_width);
            if constexpr (std::is_same_v<T, char> ||
                          std::is_same_v<T, unsigned char> ||
                          std::is_same_v<T, int8_t>) {
                std::cout << static_cast<int>(typed_data[flat_index]);
            } else {
                std::cout << typed_data[flat_index];
            }
        };
        
        // 2. YOUR HARDCODED LOOPS (Strictly typed, zero branching!)
        if (dims == 0) {
            std::cout << "matrix: 0D \n";
            print_elem(0);
            std::cout << " \n";
        } else if (dims == 1) {
            std::cout << "{ ";
            for (size_t i = 0; i < s[0]; i++) {
                print_elem(st[0] * i);
                std::cout << " ";
            }
            std::cout << "}\n";
        } else if (dims == 2) {
            std::cout << "{\n";
            for (size_t i = 0; i < s[0]; i++) {
                std::cout << "  { ";
                for (size_t j = 0; j < s[1]; j++) {
                    print_elem(st[0] * i + st[1] * j);
                    if (j < s[1] - 1)
                        std::cout << ", ";
                }
                std::cout << " }\n";
            }
            std::cout << "}\n";
        } else if (dims == 3) {
            std::cout << "{\n";
            for (size_t i = 0; i < s[0]; i++) {
                std::cout << "  {\n";
                for (size_t j = 0; j < s[1]; j++) {
                    std::cout << "    { ";
                    for (size_t k = 0; k < s[2]; k++) {
                        print_elem(st[0] * i + st[1] * j + st[2] * k);
                        if (k < s[2] - 1)
                            std::cout << ", ";
                    }
                    std::cout << " }";
                    if (j < s[1] - 1)
                        std::cout << ",\n";
                }
                std::cout << "\n  }";
                if (i < s[0] - 1)
                    std::cout << ",\n";
            }
            std::cout << "\n}\n";
        } else if (dims == 4) {
            std::cout << "{ \n";
            for (size_t l = 0; l < s[0]; l++) {
                std::cout << "  { \n";
                for (size_t i = 0; i < s[1]; i++) {
                    std::cout << "      { ";
                    for (size_t j = 0; j < s[2]; j++) {
                        std::cout << "{ ";
                        for (size_t k = 0; k < s[3]; k++) {
                            print_elem(st[0] * l + st[1] * i + st[2] * j + st[3] * k);
                            std::cout << " ";
                        }
                        std::cout << "} ";
                    }
                    std::cout << "} ";
                    std::cout << "\n";
                }
                std::cout << "  } \n";
                
            }
            std::cout << "}";
        } else {
            std::cerr << "Printing only supported up to 4D matrices.\n";
        }
    }); // End of lambda & dispatcher
}

inline void setBufferOrBytes(id<MTLComputeCommandEncoder> commandEncoder,
                             const matrix &tensor, NSUInteger index) {
    if (tensor.metalBuffer) {
        NSUInteger offset = (uint8_t*)tensor.buffer - (uint8_t*)[tensor.metalBuffer contents];
        [commandEncoder setBuffer:tensor.metalBuffer offset:offset atIndex:index];
    } else {
        [commandEncoder setBytes:tensor.buffer length:tensor.effectiveBufferSize() * dtype_size(tensor.type) atIndex:index];
    }
}

//    static matrix matrix::zeros(std::initializer_list<size_m> shapeI, dtype
//    type = dtype::Float) {
//        matrix output((uint32_t)shapeI.size(), type);
//
//        if (output.dims > SBO_MAX_DIMS) {
//            output.array_desc.shared_arr_desc =
//            SharedArrayDescriptor::create(output.dims);
//        }
//        memcpy(output.shape(), shapeI.begin(), output.dims * sizeof(size_m));
//        output.calcStrides();
//        output.total_size = output.accumul(0, output.dims);
//        output.buffer = new uint8_t[output.total_size * dtype_size(type)];
//        if (output.total_size > 10) {
//            output.buildMetalBuffer();
//        }
//        memset(output.buffer, 0, output.total_size * dtype_size(type));
//        return output;
//    }
//
//    static matrix matrix::ones(std::initializer_list<size_m> shapeI, dtype
//    type = dtype::Float) {
//        matrix output((uint32_t)shapeI.size(), type);
//
//        if (output.dims > SBO_MAX_DIMS) {
//            output.array_desc.shared_arr_desc =
//            SharedArrayDescriptor::create(output.dims);
//        }
//        memcpy(output.shape(), shapeI.begin(), output.dims * sizeof(size_m));
//        output.calcStrides();
//        output.total_size = output.accumul(0, output.dims);
//        output.buffer = new uint8_t[output.total_size * dtype_size(type)];
//        if (output.total_size > 10) {
//            output.buildMetalBuffer();
//        }
//        dispatch_type(type, output.buffer, [&](auto* typed_ptr) {
//            std::fill(typed_ptr, typed_ptr + output.total_size, 1);
//        });
//        return output;
//    }

matrix matrix::ones() const {
    matrix output(dims, type);
    memcpy(output.shape(), shape(), dims * sizeof(size_m));
    output.calcStrides();
    output.total_size = output.accumul(0, dims);
    output.buffer = new uint8_t[output.total_size * dtype_size(type)];
    if (total_size > 10) {
        output.buildMetalBuffer();
    }
    dispatch_type(type, output.buffer, [&](auto *typed_ptr) {
        std::fill(typed_ptr, typed_ptr + output.total_size, 1);
    });
    return output;
}

matrix matrix::zeros() const {
    matrix output(dims, type);
    memcpy(output.shape(), shape(), dims * sizeof(size_m));
    output.calcStrides();
    output.total_size = output.accumul(0, dims);
    output.buffer = new uint8_t[output.total_size * dtype_size(type)];
    if (total_size > 10) {
        output.buildMetalBuffer();
    }
    memset(output.buffer, 0, output.total_size * dtype_size(type));
    return output;
}

matrix matrix::eye(uint m, uint n, int k, dtype type ) {
    matrix output = matrix::zeros({ m, n }, type);
    uint iteration = MIN(m, n-std::abs(k));
    dispatch_type(type, output.buffer, [&](auto *outBuff) {
        
        if (0 <= k) {
            for (int i = 0; i < iteration; i++) {
                // [i, j+k]
                outBuff[i * output.shape()[1] + i + k] = 1;
            }

        } else {
            for (int i = 0; i < iteration; i++) {
                // [i, i-abs(k)] => same as shifting it down => [i+abs(k), i]
                // Since k is -ve => [i-k, i]
                // Moving the Diagnol Left is Same as moving it above as y = (x + k) ==> (y - k) = x
                outBuff[(i - k) * output.shape()[1] + i] = 1;
            }
        }
    });

    return output;
}

matrix matrix::eye(uint m, dtype type ) {
    return matrix::eye(m, m, 0, type);
}


//    static matrix repeating(std::initializer_list<size_m> shapeI, const
//    matrix& pattern) {
//        #ifdef SAFE_MODE
//        if (shapeI.size() + dimsI != dims) {
//            std::cerr << "Dimensions Dont Add up, Pattern: " << dimsI << " +
//            Repeat:" << shapeI.size() << " != Total Dim" << dims << "\n";
//            throw std::invalid_argument("MatrixH: Repeating shape dimensions
//            mismatch.");
//        }
//        #endif
//        matrix output(pattern.dims + shapeI.size(), pattern.type);
//
//        memcpy(output.shape(), shapeI.begin(), shapeI.size() *
//        sizeof(size_m)); memcpy(output.shape() + shapeI.size(),
//        pattern.shape(), pattern.dims * sizeof(size_m)); output.calcStrides();
//        output.total_size = output.accumul(0, output.dims);
//        output.buffer = new uint8_t[output.total_size *
//        dtype_size(pattern.type)]; PatternFill(output.buffer, pattern.buffer,
//        pattern.total_size * dtype_size(pattern.type), output.accumul(0,
//        shapeI.size())); if (output.total_size > 10 || pattern.metalBuffer)
//            output.buildMetalBuffer();
//        return output;
//    }
//
//    static matrix repeatingGPU(std::initializer_list<size_m> shapeI, const
//    matrix& pattern) {
//        #ifdef SAFE_MODE
//        if (shapeI.size() + dimsI != dims) {
//            std::cerr << "Dimensions Dont Add up, Pattern: " << dimsI << " +
//            Repeat:" << shapeI.size() << " != Total Dim" << dims << "\n";
//            throw std::invalid_argument("Repeating shape dimensions
//            mismatch."); // FIXED
//        }
//        #endif
//        matrix output(pattern.dims + shapeI.size(), pattern.type);
//        matrix patternView(pattern.dims + shapeI.size(), pattern.type);
//
//        pattern.shareBuffer(patternView);
//        memcpy(patternView.shape(), shapeI.begin(), shapeI.size() *
//        sizeof(size_m)); memcpy(patternView.shape() + shapeI.size(),
//        pattern.shape(), pattern.dims * sizeof(size_m));
//        memset(patternView.strides(), 0, shapeI.size() * sizeof(size_m));
//        memcpy(patternView.strides() + shapeI.size(), pattern.strides(),
//        pattern.dims * sizeof(size_m)); patternView.total_size =
//        patternView.accumul(0, patternView.dims); patternView.metalBuffer =
//        pattern.metalBuffer; patternView.flags |= NON_CONTIGUOUS_FLAG;
//
//        memcpy(output.shape(), patternView.shape(), patternView.dims *
//        sizeof(size_m)); output.calcStrides(); output.total_size =
//        patternView.total_size; output.buffer = new uint8_t[output.total_size
//        * dtype_size(pattern.type)]; output.buildMetalBuffer();
//
//        copyGPUinplace(output, patternView, 0);
//        return output;
//    }
//
//    static matrix fromImage(std::string path_str =
//    std::string("/Users/adityadude/Documents/TUSHU.HEIC"), CFDictionaryRef*
//    meta_out = nullptr) {
//        #if !TARGET_OS_IPHONE
//        CFStringRef path = CFStringCreateWithCString(NULL, path_str.c_str(),
//        kCFStringEncodingUTF8); CFURLRef url =
//        CFURLCreateWithFileSystemPath(NULL, path, kCFURLPOSIXPathStyle,
//        false); CGImageSourceRef source; CGImageRef cgImage; for (int i = 0; i
//        < 3; i++) {
//            source = CGImageSourceCreateWithURL(url, NULL);
//            cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
//            if (cgImage) {break;}
//        }
//        CFRelease(url);
//        CFRelease(path);
//        #endif
//
//        #if TARGET_OS_IPHONE
//        UIImage *image = [UIImage imageNamed:@"IMG_1278"];
//        CGImageRef cgImage = image.CGImage;
//        #endif
//
//        if (meta_out && source)
//            *meta_out = CGImageSourceCopyPropertiesAtIndex(source, 0,
//            nullptr);
//
//        if (!cgImage) {
//            std::cerr << "Failed to create CGImage" << std::endl;
//        }
//        size_t Imgwidth = CGImageGetWidth(cgImage);
//        size_t Imgheight = CGImageGetHeight(cgImage);
//        CGBitmapInfo bitmapInfo = CGImageGetBitmapInfo(cgImage);
//        bool isFloat = (bitmapInfo & kCGBitmapFloatComponents) != 0;
//        if (isFloat) {
//            size_t bytesPerRow = 4 * sizeof(float) * Imgwidth;
//            float* data = static_cast<float*>(malloc(bytesPerRow *
//            Imgheight));
//
//            CGColorSpaceRef space =
//            CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
//            CGContextRef ctx = CGBitmapContextCreate(data, Imgwidth,
//            Imgheight, 32, bytesPerRow, space,
//                                                     kCGBitmapFloatComponents
//                                                     |
//                                                     kCGImageAlphaPremultipliedLast);
//            CGColorSpaceRelease(space);
//            CGContextDrawImage(ctx, CGRectMake(0, 0, Imgwidth, Imgheight),
//            cgImage); CGContextRelease(ctx); CGImageRelease(cgImage);
//
//            matrix result(3, dtype::Float);
//            result.buffer = data;
//            result.shape()[0] = Imgheight;
//            result.shape()[1] = Imgwidth;
//            result.shape()[2] = 4;
//            result.calcStrides();
//            result.total_size = Imgwidth * Imgheight * 4;
//            result.buildMetalBuffer();  // MTLPixelFormatRGBA32Float
//            return result;
//        }
//        std::cout << "Img of Width: " <<Imgwidth<<"and Height: " << Imgheight
//        << "Loaded \n"; size_t bytesPerRow = 4 * Imgwidth; void *data =
//        malloc(bytesPerRow * Imgheight); CGContextRef context =
//        CGBitmapContextCreate(data, Imgwidth, Imgheight, 8, bytesPerRow,
//                                                     CGImageGetColorSpace(cgImage),
//                                                     kCGImageAlphaPremultipliedLast
//                                                     |
//                                                     kCGBitmapByteOrder32Big);
//        CGContextDrawImage(context, CGRectMake(0, 0, Imgwidth, Imgheight),
//        cgImage); CGContextRelease(context); CGImageRelease(cgImage);
//
//        uint8_t* pixelData = static_cast<uint8_t*>(data);
//
//        matrix result(3, dtype::UInt8);
//        result.buffer = pixelData;
//        result.shape()[0] = Imgheight;
//        result.shape()[1] = Imgwidth;
//        result.shape()[2] = 4;
//        result.calcStrides();
//        result.total_size = Imgwidth * Imgheight * 4;
//        result.buildMetalBuffer();
//        return result;
//    }
matrix matrix::slice(R slice_range, int slice_axis) {
    update_from_trace();
    uint32_t out_dims = slice_range.is_index ? dims - 1 : dims;
    
    matrix slicedMat = matrix(out_dims, type);
    const size_m *src_shape = this->shape();
    const size_m *src_strides = this->strides();
    size_m *dst_shape = slicedMat.shape();
    size_m *dst_strides = slicedMat.strides();
    
    size_t offsets = 0;
    
    size_t actual_start = slice_range.start < 0 ? src_shape[slice_axis] + slice_range.start : slice_range.start;
    size_t actual_end;
    if (slice_range.is_all()) {
        actual_end = src_shape[slice_axis];
    } else {
        actual_end = slice_range.end < 0 ? src_shape[slice_axis] + slice_range.end : slice_range.end;
    }
    std::vector<size_m> indices(2 * dims, 0);
    size_t dst_index = 0;
    for (int i = 0; i < dims; i++) {
        if (i == slice_axis) {
            indices[2*i]   = actual_start;
            indices[2*i+1] = actual_end;
            offsets += actual_start * src_strides[i];
            if (slice_range.is_index) {
                continue;  // squeeze: no dst write, axis dropped, dst_index does NOT advance
            }
            dst_shape[dst_index]   = (uint32_t)(actual_end - actual_start);
            dst_strides[dst_index] = src_strides[i];
            dst_index++;
        } else {
            indices[2*i]   = 0;
            indices[2*i+1] = src_shape[i];   // full range, not {0,0}
            dst_shape[dst_index]   = src_shape[i];
            dst_strides[dst_index] = src_strides[i];
            dst_index++;
        }
    }
    
    slicedMat.total_size = slicedMat.accumul(0, dims);
    slicedMat.flags |= NON_OWNERSHIP_FLAG;
    slicedMat.flags |= NON_CONTIGUOUS_FLAG;
    std::vector<size_m> unsqueze_mask = slice_range.is_index ? std::vector<size_m>{ (size_m)slice_axis } : std::vector<size_m>{};
    slicedMat.tape = new SlicePrimitive(*this, slicedMat.array_desc, indices, unsqueze_mask, slicedMat.total_size, offsets);
    
    return slicedMat;
}


matrix matrix::slice(array_descriptor slice_range, std::vector<size_m>& slice_indices, const std::vector<size_m>& unsqueeze_axis, size_t offset) {
    update_from_trace();
    matrix slicedMat = matrix(dims, type);

    slicedMat.array_desc = slice_range;
    if (dims > SBO_MAX_DIMS) { slicedMat.array_desc.shared_arr_desc->refCount.fetch_add(1); }

    slicedMat.total_size = slicedMat.accumul(0, dims);
    slicedMat.flags |= NON_OWNERSHIP_FLAG;
    slicedMat.flags |= NON_CONTIGUOUS_FLAG;
    slicedMat.tape = new SlicePrimitive(*this, slicedMat.array_desc, slice_indices, unsqueeze_axis, slicedMat.total_size, offset);

    return slicedMat;
}


matrix matrix::slice(std::initializer_list<std::optional<std::pair<size_m, size_m>>> slice_range) {
    update_from_trace();
    matrix slicedMat = matrix(dims, type);
    const size_m *src_shape = this->shape();
    const size_m *src_strides = this->strides();
    size_m *dst_shape = slicedMat.shape();
    size_m *dst_strides = slicedMat.strides();
    std::vector<size_m> indices(2 * dims, 0);
    size_t index = 0;
    size_t offsets = 0;
    for (auto i : slice_range) {
        if (i.has_value()) {
#ifdef SAFE_MODE
            if (i->second > src_shape[index]) {
                std::cerr << "matrix: index " << i->second << " excedes the shape "
                << src_shape[index] << "of axis " << index << "\n";
            }
            if (i->first > src_shape[index]) {
                std::cerr << "matrix: index " << i->second << " excedes the shape "
                << src_shape[index] << "of axis " << index << "\n";
            }
#endif
            indices[2 * index] = i->first;
            indices[2 * index + 1] = i->second;
            dst_shape[index] = (i->second - i->first);
            offsets += i->first * src_strides[index];
        } else {
            dst_shape[index] = src_shape[index];
            indices[2 * index + 1] = src_shape[index];
        }

        index++;
    }
    if (slice_range.size() < dims) {
        memcpy(dst_shape + slice_range.size(), src_shape + slice_range.size(),
               (dims - slice_range.size()) * sizeof(size_m));
    }
    memcpy(dst_strides, src_strides, dims * sizeof(size_m));
    slicedMat.total_size = slicedMat.accumul(0, dims);
    slicedMat.flags |= NON_OWNERSHIP_FLAG;
    slicedMat.flags |= NON_CONTIGUOUS_FLAG;
    slicedMat.tape = new SlicePrimitive(*this, slicedMat.array_desc, indices, {}, slicedMat.total_size, offsets);
    return slicedMat;
}

matrix matrix::slice(std::initializer_list<R> slice_range) {
    update_from_trace();
    const size_m *src_shape = this->shape();
    const size_m *src_strides = this->strides();

    // 1. FAST FIRST PASS: Count dimensions after slicing
    size_t out_dims = dims;
    for (const auto& i : slice_range) {
        if (i.is_index) out_dims--;
    }

    matrix slicedMat(out_dims, type);
    size_m *dst_shape = slicedMat.shape();
    size_m *dst_strides = slicedMat.strides();

    std::vector<size_m> indices(2 * dims, 0);
    // Renamed to unsqueeze_mask: tracks dimensions that need to be restored during backprop
    std::vector<size_m> unsqueeze_mask(dims, 0);

    uint32_t src_axis = 0;
    uint32_t dst_axis = 0;
    size_t offsets = 0;

    for (const auto& i : slice_range) {
        size_t actual_start = (i.start < 0) ? (src_shape[src_axis] + i.start) : i.start;
        size_t actual_end;

        if (i.is_all()) {
            actual_end = src_shape[src_axis];
        } else if (i.is_index) {
            actual_end = actual_start + 1;
        } else {
            actual_end = (i.end < 0) ? (src_shape[src_axis] + i.end) : i.end;
        }

#ifdef SAFE_MODE
        if (!i.is_all()) {
            if (actual_end > src_shape[src_axis] || actual_start > src_shape[src_axis]) {
                std::cerr << "matrix: index out of bounds on axis " << src_axis
                          << ". Shape is " << src_shape[src_axis] << "\n";
            }
        }
#endif

        indices[2 * src_axis] = actual_start;
        indices[2 * src_axis + 1] = actual_end;
        offsets += actual_start * src_strides[src_axis];

        if (i.is_index) {
            // Mark axis to be unsqueezed when routing gradients back to the original shape
            unsqueeze_mask[src_axis] = 1;
        } else {
            dst_shape[dst_axis] = actual_end - actual_start;
            dst_strides[dst_axis] = src_strides[src_axis];
            dst_axis++;
        }
        src_axis++;
    }

    // Fill trailing dimensions natively
    while (src_axis < dims) {
        indices[2 * src_axis] = 0;
        indices[2 * src_axis + 1] = src_shape[src_axis];
        
        dst_shape[dst_axis] = src_shape[src_axis];
        dst_strides[dst_axis] = src_strides[src_axis];
        
        src_axis++;
        dst_axis++;
    }

    slicedMat.total_size = slicedMat.accumul(0, out_dims);
    slicedMat.flags |= NON_OWNERSHIP_FLAG;
    slicedMat.flags |= NON_CONTIGUOUS_FLAG;
    
    // Pass the unsqueeze_mask to the autograd tape
    slicedMat.tape = new SlicePrimitive(*this, slicedMat.array_desc, indices, unsqueeze_mask, slicedMat.total_size, offsets);

    return slicedMat;
}

matrix matrix::slice_assign(R slice_range, int slice_axis, const matrix& rhs) {
    update_from_trace();
    matrix outMat = matrix(dims, type);
    memcpy(outMat.shape(), this->shape(), dims * sizeof(size_m));
    memcpy(outMat.strides(), this->strides(), dims * sizeof(size_m));
    outMat.total_size = this->total_size;

    matrix virtual_slice = this->slice(slice_range, slice_axis);
    SlicePrimitive* sp = static_cast<SlicePrimitive*>(virtual_slice.tape);
    outMat.tape = new SliceAssignPrimitive(*this, rhs, virtual_slice.array_desc, sp->slice_indices, virtual_slice.total_size, sp->offset);
    
    return outMat;
}

matrix matrix::slice_assign(array_descriptor slice_range, std::vector<size_m>& slice_indices, size_t offset, const matrix& rhs) {
    update_from_trace();
    matrix outMat = matrix(dims, type);
    memcpy(outMat.shape(), this->shape(), dims * sizeof(size_m));
    memcpy(outMat.strides(), this->strides(), dims * sizeof(size_m));
    outMat.total_size = this->total_size;

    matrix virtual_slice = this->slice(slice_range, slice_indices, {},  offset);
    SlicePrimitive* sp = static_cast<SlicePrimitive*>(virtual_slice.tape);
    outMat.tape = new SliceAssignPrimitive(*this, rhs, virtual_slice.array_desc, sp->slice_indices, virtual_slice.total_size, sp->offset);
    
    return outMat;
}

matrix matrix::slice_assign(std::initializer_list<std::optional<std::pair<size_m, size_m>>> slice_range, const matrix& rhs) {
    update_from_trace();
    matrix outMat = matrix(dims, type);
    memcpy(outMat.shape(), this->shape(), dims * sizeof(size_m));
    memcpy(outMat.strides(), this->strides(), dims * sizeof(size_m));
    outMat.total_size = this->total_size;

    matrix virtual_slice = this->slice(slice_range);
    SlicePrimitive* sp = static_cast<SlicePrimitive*>(virtual_slice.tape);
    outMat.tape = new SliceAssignPrimitive(*this, rhs, virtual_slice.array_desc, sp->slice_indices, virtual_slice.total_size, sp->offset);
    
    return outMat;
}

matrix matrix::slice_assign(std::initializer_list<R> slice_range, const matrix& rhs) {
    update_from_trace();
    matrix outMat = matrix(dims, type);
    memcpy(outMat.shape(), this->shape(), dims * sizeof(size_m));
    memcpy(outMat.strides(), this->strides(), dims * sizeof(size_m));
    outMat.total_size = this->total_size;

    matrix virtual_slice = this->slice(slice_range);
    SlicePrimitive* sp = static_cast<SlicePrimitive*>(virtual_slice.tape);
    outMat.tape = new SliceAssignPrimitive(*this, rhs, virtual_slice.array_desc, sp->slice_indices, virtual_slice.total_size, sp->offset);
    
    return outMat;
}

void matrix::padding(std::initializer_list<std::pair<size_m, size_m>> padding_range, matrix& padded_mat, const matrix& value, EvalType eval_type) {
    update_from_trace();
    const size_m *src_shape = this->shape();
    const size_m *src_strides = this->strides();
    size_m *dst_shape = padded_mat.shape();
    size_m *dst_strides = padded_mat.strides();
    if (eval_type == EvalType::EVAL_AUTO) {
        for (int i = 0; i < padding_range.size(); i++) {
            dst_shape[i] = src_shape[i] + (padding_range.begin() + i)->first + (padding_range.begin() + i)->second;
        }
        memcpy(dst_shape + padding_range.size(), src_shape + padding_range.size(), (dims - padding_range.size()) * sizeof(size_m));
        padded_mat.calcStrides();
        size_t offset = 0;
        for (int i = 0; i < padding_range.size(); i++) {
            offset += (padding_range.begin() + i)->first * src_strides[i];
        }
        padded_mat.total_size = padded_mat.accumul(0, dims);
        
        std::vector<size_m> pad_flat(dims * 2, 0);
        for (int i = 0; i < dims; i++ ) {
            pad_flat[2 * i] = (padding_range.begin() + i)->first;
            pad_flat[2 * i + 1] = (padding_range.begin() + i)->first + src_shape[i];
        }
        memcpy(pad_flat.data() + padding_range.size(), src_shape + padding_range.size(), (dims - padding_range.size()) * sizeof(size_m));
//        memcpy(pad_flat.data(), padding_range.begin(), padding_range.size() * sizeof(std::pair<size_m, size_m>));
        padded_mat.tape = new PaddingPrimitive(*this, const_cast<matrix&>(value), offset, pad_flat);
        return;
    }
    
    PaddingPrimitive* prim = (PaddingPrimitive*)padded_mat.tape;
    
    
    size_t elem_size = dtype_size(padded_mat.type);
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();

    uint8_t typeCode = static_cast<int>(padded_mat.type);
    
    uint32_t cdims = padded_mat.dims;
    
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
    
    auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
    auto _dispatchExecutionSize = MTLSizeMake(padded_mat.total_size, 1, 1);
    setBufferOrBytes(commandEncoder, padded_mat, 0);
    setBufferOrBytes(commandEncoder, *this, 1);
    [commandEncoder setBytes:dst_strides length:cdims * sizeof(size_m) atIndex:2];
    [commandEncoder setBytes:src_strides length:cdims * sizeof(size_m) atIndex:3];
    [commandEncoder setBytes:prim->padding_range.data() length: 2 * cdims * sizeof(size_m) atIndex:4];
    
    [commandEncoder setBytes:&(prim->offset) length:sizeof(int) atIndex:5];
    [commandEncoder setBytes:value.buffer length:dtype_size(value.type) atIndex:5];
    if (cdims == 1) {
        if (!GlobalGPUManager.CopyInplace[typeCode][0]) {
            GlobalGPUManager.initPadding(typeCode, 0);
        }
        [commandEncoder setComputePipelineState:GlobalGPUManager.Padding_ComputeState[typeCode][0]];
    } else if (cdims == 2) {
        if (!GlobalGPUManager.CopyInplace[typeCode][1]) {
            GlobalGPUManager.initPadding(typeCode, 1);
        }
        [commandEncoder setComputePipelineState:GlobalGPUManager.Padding_ComputeState[typeCode][1]];
        _dispatchExecutionSize = MTLSizeMake(dst_shape[1], dst_shape[0], 1);

    } else if (cdims == 3) {
        if (!GlobalGPUManager.CopyInplace[typeCode][2]) {
            GlobalGPUManager.initPadding(typeCode, 2);
        }
        [commandEncoder setComputePipelineState:GlobalGPUManager.Padding_ComputeState[typeCode][2]];
        _dispatchExecutionSize = MTLSizeMake(dst_shape[2], dst_shape[1], dst_shape[0]);
        
    } else {
        if (!GlobalGPUManager.CopyInplace[typeCode][3]) {
            GlobalGPUManager.initPadding(typeCode, 3);
        }
        size_m acc = 1;
        for (int i = 0; i < cdims - 2; i++) {
            acc *= dst_shape[i];
        }
        _dispatchExecutionSize = MTLSizeMake(dst_shape[cdims - 1], dst_shape[cdims - 2], acc);
        [commandEncoder setBytes:dst_shape length:cdims * sizeof(size_m) atIndex:5];
        [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:6];
        [commandEncoder setComputePipelineState:GlobalGPUManager.Padding_ComputeState[typeCode][2]];
    }
    
    [commandEncoder dispatchThreads:_dispatchExecutionSize
              threadsPerThreadgroup:_threadsPerThreadgroup];
    
}

void matrix::padding(std::vector<size_m>& padding_range, matrix& padded_mat, const matrix& value, EvalType eval_type) {
    update_from_trace();
    const size_m *src_shape = this->shape();
    const size_m *src_strides = this->strides();
    size_m *dst_shape = padded_mat.shape();
    size_m *dst_strides = padded_mat.strides();
    if (eval_type == EvalType::EVAL_AUTO) {
        for (int i = 0; i < padding_range.size(); i+=2) {
            dst_shape[i] = src_shape[i] + padding_range[i] + padding_range[i+1];
        }
        memcpy(dst_shape + padding_range.size(), src_shape + padding_range.size(), (dims - padding_range.size()) * sizeof(size_m));
        padded_mat.calcStrides();
        size_t offset = 0;
        for (int i = 0; i < padding_range.size(); i+=2) {
            offset += padding_range[i] * src_strides[i];
        }
        padded_mat.total_size = padded_mat.accumul(0, dims);
        padded_mat.tape = new PaddingPrimitive(*this, const_cast<matrix&>(value), offset, padding_range);
        return;
    }
    
    PaddingPrimitive* prim = (PaddingPrimitive*)padded_mat.tape;
    

    
    size_t elem_size = dtype_size(padded_mat.type);
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();

    uint8_t typeCode = static_cast<int>(padded_mat.type);
    
    uint32_t cdims = padded_mat.dims;
    
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
    
    auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
    auto _dispatchExecutionSize = MTLSizeMake(padded_mat.total_size, 1, 1);
    setBufferOrBytes(commandEncoder, padded_mat, 0);
    setBufferOrBytes(commandEncoder, *this, 1);
    [commandEncoder setBytes:dst_strides length:cdims * sizeof(size_m) atIndex:2];
    [commandEncoder setBytes:src_strides length:cdims * sizeof(size_m) atIndex:3];
    [commandEncoder setBytes:prim->padding_range.data() length: 2 * cdims * sizeof(size_m) atIndex:4];
    
    [commandEncoder setBytes:&(prim->offset) length:sizeof(int) atIndex:5];
    [commandEncoder setBytes:value.buffer length:dtype_size(value.type) atIndex:6];
    if (cdims == 1) {
        if (!GlobalGPUManager.PaddingInit[typeCode][0]) {
            GlobalGPUManager.initPadding(typeCode, 0);
        }
        [commandEncoder setComputePipelineState:GlobalGPUManager.Padding_ComputeState[typeCode][0]];
    } else if (cdims == 2) {
        if (!GlobalGPUManager.PaddingInit[typeCode][1]) {
            GlobalGPUManager.initPadding(typeCode, 1);
        }
        [commandEncoder setComputePipelineState:GlobalGPUManager.Padding_ComputeState[typeCode][1]];
        _dispatchExecutionSize = MTLSizeMake(dst_shape[1], dst_shape[0], 1);

    } else if (cdims == 3) {
        if (!GlobalGPUManager.PaddingInit[typeCode][2]) {
            GlobalGPUManager.initPadding(typeCode, 2);
        }
        [commandEncoder setComputePipelineState:GlobalGPUManager.Padding_ComputeState[typeCode][2]];
        _dispatchExecutionSize = MTLSizeMake(dst_shape[2], dst_shape[1], dst_shape[0]);
        
    } else {
        if (!GlobalGPUManager.PaddingInit[typeCode][3]) {
            GlobalGPUManager.initPadding(typeCode, 3);
        }
        size_m acc = 1;
        for (int i = 0; i < cdims - 2; i++) {
            acc *= dst_shape[i];
        }
        _dispatchExecutionSize = MTLSizeMake(dst_shape[cdims - 1], dst_shape[cdims - 2], acc);
        [commandEncoder setBytes:dst_shape length:cdims * sizeof(size_m) atIndex:5];
        [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:6];
        [commandEncoder setComputePipelineState:GlobalGPUManager.Padding_ComputeState[typeCode][2]];
    }
    
    [commandEncoder dispatchThreads:_dispatchExecutionSize
              threadsPerThreadgroup:_threadsPerThreadgroup];
    
}

matrix matrix::padding(std::vector<size_m>& padding_range, const matrix& value) {
    matrix padded_mat(dims, type);
    padding(padding_range, padded_mat, value);
    return padded_mat;
}

matrix matrix::broadcast_to(const size_m *target_shape, int target_dims) const {
    throw std::runtime_error("matrix: depreceated function use broadcast_toV2");
    if (target_dims < dims)
        throw std::runtime_error("matrix: Cannot broadcast to fewer dimensions.");
    
    matrix view(target_dims, type);
    view.tape = tape;
    if (view.tape) view.tape->primitive_refCount.fetch_add(1, std::memory_order_relaxed);
    view.flags = flags | NON_OWNERSHIP_FLAG | NON_CONTIGUOUS_FLAG;
    view.buffer = buffer;
    view.metalBuffer = metalBuffer;
    
    // Safely share the data
    if (refCount) {
        view.refCount = refCount;
        view.refCount->fetch_add(1, std::memory_order_relaxed);
    } else if (!(flags & NON_OWNERSHIP_FLAG)) {
        const_cast<matrix *>(this)->refCount = new std::atomic<uint32_t>(2);
        view.refCount = this->refCount;
    }
    
    if (view.dims > SBO_MAX_DIMS) {
        view.array_desc.shared_arr_desc = SharedArrayDescriptor::create(view.dims);
    }
    
    // ==========================================
    // THE BROADCASTING LOGIC (Right-to-Left)
    // ==========================================
    int src_d = dims - 1;
    view.total_size = 1;
    
    for (int dst_d = target_dims - 1; dst_d >= 0; --dst_d) {
        size_m dst_size = target_shape[dst_d];
        view.shape()[dst_d] = dst_size;
        view.total_size *= dst_size;
        
        if (src_d >= 0) {
            size_m src_size = shape()[src_d];
            if (src_size == dst_size) {
                // Dimension matches perfectly, copy the stride
                view.strides()[dst_d] = strides()[src_d];
            } else if (src_size == 1) {
                // Source dimension is 1, stretch it by setting stride to 0!
                view.strides()[dst_d] = 0;
            } else {
                throw std::runtime_error("MatrixH: Shapes are not broadcastable.");
            }
        } else {
            // We ran out of source dimensions (e.g. promoting [4] to [10, 4])
            // Pretend the missing dimension was a 1, so set stride to 0!
            view.strides()[dst_d] = 0;
        }
        src_d--;
    }
    
    return view;
}

std::pair<matrix, matrix> matrix::meshgrid(const matrix& x, const matrix& y, bool sparse) {
    int num = 2;
    
    size_m out_shape[num * x.dims];
//    out_shape = [Y.shape | X.shape]
    
    memcpy(out_shape, y.shape(), y.dims * sizeof(size_m));
    memcpy(out_shape + y.dims, x.shape(), x.dims * sizeof(size_m));
    
    matrix X = x.broadcast_toV2(out_shape, num * x.dims);
    uint32_t axii[x.dims];
    for (uint32_t i = 0; i < x.dims; i++) {
        axii[i] = i + y.dims;
    }
    matrix Y_unsq = y.unsqueeze(axii, x.dims);
    if (sparse) return {x, Y_unsq};
    
    matrix Y = Y_unsq.broadcast_toV2(out_shape, num * x.dims);
    return {X, Y};
}

std::tuple<matrix, matrix, matrix> matrix::meshgrid(const matrix& x, const matrix& y, const matrix& z, bool sparse) {
    int total_dims = x.dims + y.dims + z.dims;
    
    size_m out_shape[total_dims];
    memcpy(out_shape, z.shape(), z.dims * sizeof(size_m));
    memcpy(out_shape + z.dims, y.shape(), y.dims * sizeof(size_m));
    memcpy(out_shape + z.dims + y.dims, x.shape(), x.dims * sizeof(size_m));
    // out_shape = [Z.shape | Y.shape | X.shape]

    
    // careful: unsqueeze axes are in the NEW tensor's index space
    // Y starts as (m,), we want (m, 1) so axis = y.dims = 1
    uint32_t y_axes[x.dims];
    for (uint32_t i = 0; i < x.dims; i++)
        y_axes[i] = y.dims + i;          // (m,) → (m, 1)
    matrix Y_unsq = y.unsqueeze(y_axes, x.dims);

    // Z: append y.dims + x.dims axes
    uint32_t z_axes[y.dims + x.dims];
    for (uint32_t i = 0; i < y.dims + x.dims; i++)
        z_axes[i] = z.dims + i;          // (p,) → (p, 1, 1)
    matrix Z_unsq = z.unsqueeze(z_axes, y.dims + x.dims);

    if (sparse) return {x, Y_unsq, Z_unsq};

    matrix X_out = x.broadcast_toV2(out_shape, total_dims);
    matrix Y_out = Y_unsq.broadcast_toV2(out_shape, total_dims);
    matrix Z_out = Z_unsq.broadcast_toV2(out_shape, total_dims);
    return {X_out, Y_out, Z_out};
}

matrix matrix::operator[](AxisRange range) {
    return slice({range});
}
matrix matrix::operator[](AxisRange range1, AxisRange range2) {
    return slice({range1, range2});
}
matrix matrix::operator[](AxisRange range1, AxisRange range2, AxisRange range3) {
    return slice({range1, range2, range3});
}

//matrix matrix::operator[](int i, int j) const {
//#ifdef SAFE_MODE
//    if (i < 0) {
//        i = shape[0] + i;
//    }
//    if (i >= shape[0]) {
//        throw std::invalid_argument("Index Out Of range");
//    }
//    
//    if (j < 0) {
//        j = shape[1] + j;
//    }
//    if (j >= shape[1]) {
//        throw std::invalid_argument("Index Out Of range");
//    }
//#endif
//    matrix slicedMat(dims - 2, type);
//    const size_m *curr_stride = strides();
//    
//    slicedMat.buffer =
//    (uint8_t *)buffer +
//    (curr_stride[0] * i + curr_stride[1] * j) * dtype_size(type);
//    
//    int sub_dims = dims - 2;
//    if (sub_dims == 1) {
//        // Highly optimized fast-path for 3D tensors returning a 1D pixel natively
//        slicedMat.strides()[0] = curr_stride[2];
//        slicedMat.shape()[0] = shape()[2];
//        slicedMat.total_size = shape()[2];
//    } else {
//        memcpy(slicedMat.strides(), curr_stride + 2, sub_dims * sizeof(size_m));
//        memcpy(slicedMat.shape(), shape() + 2, sub_dims * sizeof(size_m));
//        slicedMat.total_size = accumul(2, dims);
//    }
//    
//    if (slicedMat.total_size > 10) {
//        slicedMat.buildMetalBuffer();
//    }
//    slicedMat.flags |= NON_OWNERSHIP_FLAG;
//    return slicedMat;
//}

void matrix::CopyToTexture(id<MTLTexture> texture, Execution exec) {
#ifdef SAFE_MODE
    if (dims < 2) {
        throw std::runtime_error(
                                 "Matrix must be at least 2D to blit to a texture.");
    }
#endif
    NSUInteger width = (NSUInteger)shape()[1];
    NSUInteger height = (NSUInteger)shape()[0];
    MTLRegion region = MTLRegionMake2D(0, 0, width, height);
    
    NSUInteger bytesPerRow = strides()[0] * dtype_size(type);
    size_t source_offset = 0;
    if (flags & NON_OWNERSHIP_FLAG) {
        source_offset = (uint8_t *)buffer - (uint8_t *)metalBuffer.contents;
    }
    
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    GlobalGPUManager.endCommandEncoding();
    id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
    
    [blitEncoder copyFromBuffer:metalBuffer
                   sourceOffset:source_offset
              sourceBytesPerRow:bytesPerRow
            sourceBytesPerImage:bytesPerRow * height
                     sourceSize:region.size
                      toTexture:texture
               destinationSlice:0
               destinationLevel:0
              destinationOrigin:region.origin];
    [blitEncoder endEncoding];
    if (exec == Execution::EncodeAndExecute) {
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        GlobalGPUManager.setCommandBuffer(nil);
    }
}

id<MTLTexture> matrix::ToMTLTexture(Execution exec) {
    id<MTLTexture> resultTexture;
    MTLTextureDescriptor *drawableDesc = [MTLTextureDescriptor
                                          texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB
                                          width:(NSUInteger)shape()[1]
                                          height:(NSUInteger)shape()[0]
                                          mipmapped:NO];
    drawableDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    // Use shared storage so the CPU can read the texture data.
    drawableDesc.storageMode = MTLStorageModeShared;
    
    resultTexture = [GlobalGPUManager.metalDevice newTextureWithDescriptor:drawableDesc];
    
    CopyToTexture(resultTexture, exec);
    return resultTexture;
}

void matrix::save_as_image(std::string path, ImgType img_type) {
    int width = shape()[1];
    int height = shape()[0];
    int channels = shape()[2]; // 3 = RGB, 4 = RGBA
    uint8_t *src = reinterpret_cast<uint8_t *>(buffer);
    
    CFStringRef pathStr = CFStringCreateWithCString(
                                                    kCFAllocatorDefault, path.c_str(), kCFStringEncodingUTF8);
    CFURLRef url = CFURLCreateWithFileSystemPath(kCFAllocatorDefault, pathStr,
                                                 kCFURLPOSIXPathStyle, false);
    CFRelease(pathStr);
    
    CFStringRef uti;
    switch (img_type) {
        case ImgType::PNG:
            uti = kUTTypePNG;
            break;
        case ImgType::JPG:
            uti = kUTTypeJPEG;
            break;
        case ImgType::EXR:
            uti = CFSTR("com.ilm.openexr-image");
            break;
    }
    
    CGColorSpaceRef colorSpace = (channels == 1) ? CGColorSpaceCreateDeviceGray()
    : CGColorSpaceCreateDeviceRGB();
    
    if (!colorSpace) {
        fprintf(stderr, "Failed to create color space!\n");
        CFRelease(url);
        return;
    }
    
    // EXR: upcaste uint8 → float16, write as HDR
    // PNG/JPG: use uint8 directly
    void *bufPtr;
    CGBitmapInfo bitmapInfo;
    size_t bitsPerComponent;
    size_t bytesPerRow;
    std::vector<uint16_t> half_buf; // owns EXR conversion buffer
    
    if (img_type == ImgType::EXR) {
        CGColorSpaceRelease(colorSpace);
        colorSpace = CGColorSpaceCreateWithName(kCGColorSpaceLinearSRGB);
        
        size_t n = (size_t)width * height * channels;
        half_buf.resize(n);
        
        // uint8 [0,255] → float [0,1] → float16
        for (size_t i = 0; i < n; i++) {
            half_buf[i] = (__fp16)((float)src[i] / 255.0f);
            // stored into uint16_t via memcpy to avoid type-punning UB:
            __fp16 h = (__fp16)((float)src[i] / 255.0f);
            memcpy(&half_buf[i], &h, sizeof(uint16_t));
        }
        bufPtr = half_buf.data();
        bitsPerComponent = 16;
        bytesPerRow = (size_t)width * channels * sizeof(uint16_t);
        bitmapInfo =
        kCGBitmapByteOrder16Host | kCGBitmapFloatComponents |
        (channels == 4 ? kCGImageAlphaPremultipliedLast : kCGImageAlphaNone);
    } else {
        bufPtr = src;
        bitsPerComponent = 8;
        bytesPerRow = (size_t)width * channels;
        bitmapInfo =
        (channels == 4 ? kCGImageAlphaPremultipliedLast : kCGImageAlphaNone);
    }
    
    CGContextRef context =
    CGBitmapContextCreate(bufPtr, width, height, bitsPerComponent,
                          bytesPerRow, colorSpace, bitmapInfo);
    if (!context) {
        fprintf(stderr, "Failed to create bitmap context!\n");
        CGColorSpaceRelease(colorSpace);
        CFRelease(url);
        return;
    }
    
    CGImageRef image = CGBitmapContextCreateImage(context);
    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    if (!image) {
        fprintf(stderr, "Failed to create CGImage!\n");
        CFRelease(url);
        return;
    }
    
    CGImageDestinationRef dest =
    CGImageDestinationCreateWithURL(url, uti, 1, NULL);
    CFRelease(url);
    
    if (!dest) {
        fprintf(stderr, "Failed to create image destination!\n");
        CGImageRelease(image);
        return;
    }
    
    CGImageDestinationAddImage(dest, image, NULL);
    if (!CGImageDestinationFinalize(dest))
        fprintf(stderr, "Failed to write: %s\n", path.c_str());
    
    CGImageRelease(image);
    CFRelease(dest);
}

//    static void copyGPUinplace( matrix& outMat, const matrix& inMat, int
//    offset, Execution exec = Execution::EncodeAndExecute) {
// #ifdef SAFE_MODE
//        if (inMat.total_size > outMat.total_size) {
//            std::cerr << "MatrixH: CopyInplace operation requires both mats to
//            be of same size." << "\n"; throw;
//        }
//        if (inMat.type != outMat.type) {
//            throw std::runtime_error("Type mismatch in copy");
//        }
// #endif
//        if (inMat.type != outMat.type) {
//            copyGPUinplaceTypeCasted(outMat, inMat, offset, exec);
//            return;
//        }
//        size_t elem_size = dtype_size(inMat.type);
//        id<MTLCommandBuffer> commandBuffer =
//        GlobalGPUManager.getCommandBuffer();
////        Blit fast path for the GPU. A compute shader (even a 1D one)
/// requires the GPU's ALU execution units to run the copy loop. A Blit command
/// skips the ALUs entirely and uses the GPU's direct memory access (DMA)
/// engines to blast the bytes across VRAM. It is drastically faster.
//        if (!(inMat.flags & NON_CONTIGUOUS_FLAG) && !(outMat.flags &
//        NON_CONTIGUOUS_FLAG)) {
//            id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer
//            blitCommandEncoder]; [blitEncoder copyFromBuffer:inMat.metalBuffer
//                           sourceOffset:offset * elem_size
//                               toBuffer:outMat.metalBuffer
//                      destinationOffset:offset * elem_size
//                                   size:inMat.total_size * elem_size];
//            [blitEncoder endEncoding];
//            if (exec == Execution::EncodeAndExecute) {
//                [commandBuffer commit];
//                [commandBuffer waitUntilCompleted];
//                GlobalGPUManager.gCommandBuffer = nil;
//            }
//            return;
//        }
//        uint8_t typeCode = static_cast<int>(inMat.type);
//
//        auto res = collapse_dims(inMat.shape(), outMat.strides(),
//        inMat.strides(), inMat.dims, INT32_MAX); uint32_t cdims =
//        res.out_dims;
//
//
//        id<MTLComputeCommandEncoder> commandEncoder =
//        GlobalGPUManager.getCommandEncoder();
//
//        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(inMat.total_size, 1, 1);
//        setBufferOrBytes(commandEncoder, outMat, 0);
//        setBufferOrBytes(commandEncoder, inMat, 1);
//        [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m)
//        atIndex:2]; [commandEncoder setBytes:res.stridesB  length:cdims *
//        sizeof(size_m) atIndex:3]; [commandEncoder setBytes:&offset
//        length:sizeof(int) atIndex:4]; if (cdims == 1) {
//            if (!GlobalGPUManager.CopyInplace[typeCode][typeCode][0]) {
//                GlobalGPUManager.initCopyInplace(typeCode, typeCode, 0);
//            }
//            [commandEncoder
//            setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[typeCode][typeCode][0]];
//        } else if (cdims == 2) {
//            if (!GlobalGPUManager.CopyInplace[typeCode][typeCode][1]) {
//                GlobalGPUManager.initCopyInplace(typeCode, typeCode, 1);
//            }
//            _dispatchExecutionSize =  MTLSizeMake(res.shape[1], res.shape[0],
//            1); [commandEncoder
//            setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[typeCode][typeCode][1]];
//        } else if (cdims == 3) {
//            if (!GlobalGPUManager.CopyInplace[typeCode][typeCode][2]) {
//                GlobalGPUManager.initCopyInplace(typeCode, typeCode, 2);
//            }
//            _dispatchExecutionSize =  MTLSizeMake(res.shape[2], res.shape[1],
//            res.shape[0]); [commandEncoder
//            setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[typeCode][typeCode][2]];
//
//        } else {
//            if (!GlobalGPUManager.CopyInplace[typeCode][typeCode][3]) {
//                GlobalGPUManager.initCopyInplace(typeCode, typeCode, 3);
//            }
//            size_m acc = 1;
//            for (int i = 0; i < cdims-2; i++) {acc *= res.shape[i]; }
//            _dispatchExecutionSize =  MTLSizeMake(res.shape[cdims-1],
//            res.shape[cdims-2], acc); [commandEncoder setBytes:res.shape
//            length:cdims * sizeof(size_m) atIndex:5]; [commandEncoder
//            setBytes:&cdims length:sizeof(uint32_t) atIndex:6];
//            [commandEncoder
//            setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[typeCode][typeCode][3]];
//        }
//
//
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//        if (exec == Execution::EncodeAndExecute) {
//            [commandEncoder endEncoding];
//            [commandBuffer commit];
//            [commandBuffer waitUntilCompleted];
//            GlobalGPUManager.gCommandBuffer = nil;
//            GlobalGPUManager.gCommandEncoder=nil;
//        }
//    }
//
//    static void copyGPUinplaceTypeCasted( matrix& outMat, const matrix& inMat,
//    int offset, Execution exec = Execution::EncodeAndExecute) {
// #ifdef SAFE_MODE
//        if (inMat.total_size > outMat.total_size) {
//            std::cerr << "MatrixH: CopyInplace operation requires both mats to
//            be of same size." << "\n"; throw;
//        }
//        if (inMat.type != outMat.type) {
//            throw std::runtime_error("Type mismatch in copy");
//        }
// #endif
//        uint8_t typeCode = static_cast<int>(inMat.type);
//        uint8_t dstTypeCode = static_cast<int>(outMat.type);
//
//        auto res = collapse_dims(inMat.shape(), outMat.strides(),
//        inMat.strides(), inMat.dims, INT32_MAX); uint32_t cdims =
//        res.out_dims;
//
//        id<MTLCommandBuffer> commandBuffer =
//        GlobalGPUManager.getCommandBuffer(); id<MTLComputeCommandEncoder>
//        commandEncoder = GlobalGPUManager.getCommandEncoder();
//
//        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(inMat.total_size, 1, 1);
//        setBufferOrBytes(commandEncoder, outMat, 0);
//        setBufferOrBytes(commandEncoder, inMat, 1);
//        [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m)
//        atIndex:2]; [commandEncoder setBytes:res.stridesB  length:cdims *
//        sizeof(size_m) atIndex:3]; [commandEncoder setBytes:&offset
//        length:sizeof(int) atIndex:4]; if (cdims == 1) {
//            if (!GlobalGPUManager.CopyInplace[dstTypeCode][typeCode][0]) {
//                GlobalGPUManager.initCopyInplace(dstTypeCode, typeCode, 0);
//            }
//            [commandEncoder
//            setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[dstTypeCode][typeCode][0]];
//        } else if (cdims == 2) {
//            if (!GlobalGPUManager.CopyInplace[dstTypeCode][typeCode][1]) {
//                GlobalGPUManager.initCopyInplace(dstTypeCode, typeCode, 1);
//            }
//            _dispatchExecutionSize =  MTLSizeMake(res.shape[1], res.shape[0],
//            1); [commandEncoder
//            setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[dstTypeCode][typeCode][1]];
//        } else if (cdims == 3) {
//            if (!GlobalGPUManager.CopyInplace[dstTypeCode][typeCode][2]) {
//                GlobalGPUManager.initCopyInplace(dstTypeCode, typeCode, 2);
//            }
//            _dispatchExecutionSize =  MTLSizeMake(res.shape[2], res.shape[1],
//            res.shape[0]); [commandEncoder
//            setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[dstTypeCode][typeCode][2]];
//
//        } else {
//            if (!GlobalGPUManager.CopyInplace[dstTypeCode][typeCode][3]) {
//                GlobalGPUManager.initCopyInplace(dstTypeCode, typeCode, 3);
//            }
//            size_m acc = 1;
//            for (int i = 0; i < cdims-2; i++) {acc *= res.shape[i]; }
//            _dispatchExecutionSize =  MTLSizeMake(res.shape[cdims-1],
//            res.shape[cdims-2], acc); [commandEncoder setBytes:res.shape
//            length:cdims * sizeof(size_m) atIndex:5]; [commandEncoder
//            setBytes:&cdims length:sizeof(uint32_t) atIndex:6];
//            [commandEncoder
//            setComputePipelineState:GlobalGPUManager.CopyInplace_ComputeState[dstTypeCode][typeCode][3]];
//        }
//
//
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//        if (exec == Execution::EncodeAndExecute) {
//            [commandEncoder endEncoding];
//            [commandBuffer commit];
//            [commandBuffer waitUntilCompleted];
//            GlobalGPUManager.gCommandBuffer = nil;
//            GlobalGPUManager.gCommandEncoder=nil;
//        }
//    }
//
//    static void copyCPUinplace( matrix& outMat, const matrix& inMat, int
//    offset) {
// #ifdef SAFE_MODE
//        if (inMat.total_size > outMat.total_size) {
//            std::cerr << "MatrixH: CopyInplace operation requires both mats to
//            be of same size." << "\n"; throw;
//        }
//        if (inMat.dims != outMat.dims) {
//            throw std::runtime_error("Type mismatch in copy");
//        }
// #endif
//        if (inMat.type != outMat.type) {
//            copyCPUinplaceTypeCasted(outMat, inMat, offset);
//            return;
//        }
//        size_t elem_size = dtype_size(inMat.type);
//        if (!(inMat.flags & NON_CONTIGUOUS_FLAG) && !(outMat.flags &
//        NON_CONTIGUOUS_FLAG)) {
//            size_t bytes = inMat.total_size * elem_size;
//            // // Native CPU fast-paths skipping dynamic memcpy linkage
//            (Critical for nested loops plotting 32-bit floats or 4-channel
//            uint8!) if (bytes == 4) {
//                *reinterpret_cast<uint32_t*>(outMat.buffer) =
//                *reinterpret_cast<const uint32_t*>(inMat.buffer);
//            } else if (bytes == 1) {
//                *reinterpret_cast<uint8_t*>(outMat.buffer)  =
//                *reinterpret_cast<const uint8_t*>(inMat.buffer);
//            } else if (bytes == 8) {
//                *reinterpret_cast<uint64_t*>(outMat.buffer) =
//                *reinterpret_cast<const uint64_t*>(inMat.buffer);
//            } else {
//                memcpy(outMat.buffer, inMat.buffer, bytes);
//            }
//            return;
//        }
//
//        auto res = collapse_dims(inMat.shape(), outMat.strides(),
//        inMat.strides(), inMat.dims, INT32_MAX); auto cdims = res.out_dims;
//        dispatch_type(inMat.type, inMat.buffer, [&](auto* in_data) {
//            using T = std::decay_t<decltype(*in_data)>;
//            T* out_data = static_cast<T*>(outMat.buffer);
//            // us stands for unsafe and fast subscripting so it doesnt suppor
//            negative indices and is super fast. if (cdims == 1) {
//                for (uint32_t i = 0; i < inMat.total_size; i++) {
//                    out_data[i * res.stridesA[0]] = in_data[i *
//                    res.stridesB[0]];
//                }
//            } else if (cdims == 2) {
//                if (res.stridesA[cdims-1] == 1 && res.stridesB[cdims-1] == 1)
//                {
//                    for (uint32_t i = 0; i < res.shape[0]; i++) {
//                    memcpy(out_data + res.stridesA[0] * i, in_data +
//                    res.stridesB[0] * i, res.shape[1] * elem_size); } return;
//                }
//                for (uint32_t i = 0; i < res.shape[0]; i++) {
//                    for (uint32_t j = 0; j < res.shape[1]; j++) {
//                        out_data[i * res.stridesA[0] + j * res.stridesA[1]] =
//                        in_data[i * res.stridesB[0] + j * res.stridesB[1]]; }
//                }
//            } else if (cdims == 3) {
//                if (res.stridesA[cdims-1] == 1 && res.stridesB[cdims-1] == 1)
//                {
//                    for (uint32_t i = 0; i < res.shape[0]; i++) {
//                        for (uint32_t j = 0; j < res.shape[1]; j++) {
//                        memcpy(out_data + res.stridesA[0] * i +
//                        res.stridesA[1] * j, in_data + res.stridesB[0] * i +
//                        res.stridesB[1] * j, res.shape[2] * elem_size); }
//                    }
//                    return;
//                }
//                for (uint32_t i = 0; i < res.shape[0]; i++) {
//                    for (uint32_t j = 0; j < res.shape[1]; j++) {
//                        for (uint32_t k = 0; k < res.shape[2]; k++) {
//                            out_data[i * res.stridesA[0] + j * res.stridesA[1]
//                            + k * res.stridesA[2]] = in_data[i *
//                            res.stridesB[0] + j * res.stridesB[1] + k *
//                            res.stridesB[2]];
//                        }
//                    }
//                }
//
//            } else {
//                uint32_t outer_iterations = 1;
//                for (uint32_t o = 0; o <= cdims - 4; o++) {
//                    outer_iterations *= res.shape[o];
//                }
//                for (uint32_t o = 0; o < outer_iterations; o++) {
//                    uint32_t inMatIndex = 0;
//                    uint32_t outMatIndex = 0;
//                    uint32_t rem = o;
//                    for (int i = cdims-4; i >=0; i--) {
//                        inMatIndex  += (rem % res.shape[i]) * res.stridesB[i];
//                        outMatIndex += (rem % res.shape[i]) * res.stridesA[i];
//                        rem /= res.shape[i];
//                    }
//                    if (res.stridesA[cdims-1] == 1 && res.stridesB[cdims-1] ==
//                    1) {
//                        for (uint32_t i = 0; i < res.shape[0]; i++) {
//                            for (uint32_t j = 0; j < res.shape[1]; j++) {
//                            memcpy(out_data + res.stridesA[cdims-3] * i +
//                            res.stridesA[cdims-2] * j, in_data +
//                            res.stridesB[cdims-3] * i + res.stridesB[cdims-2]
//                            * j, res.shape[cdims-1] * elem_size); }
//                        }
//                        break;
//                    }
//                    for (uint32_t i = 0; i < res.shape[cdims-3]; i++) {
//                        for (uint32_t j = 0; j < res.shape[cdims-2]; j++) {
//                            for (uint32_t k = 0; k < res.shape[cdims-1]; k++)
//                            {
//                                out_data[outMatIndex + i *
//                                res.stridesA[cdims-3] + j *
//                                res.stridesA[cdims-2] + k *
//                                res.stridesA[cdims-1]] = in_data[inMatIndex +
//                                i * res.stridesB[cdims-3] + j *
//                                res.stridesB[cdims-2] + k *
//                                res.stridesB[cdims-1]];
//                            }
//                        }
//                    }
//                }
//            }
//        });
//    }
//
//
//    static void copyCPUinplaceTypeCasted(matrix& outMat, const matrix& inMat,
//    int offset) {
//        auto res = collapse_dims(inMat.shape(), outMat.strides(),
//        inMat.strides(), inMat.dims, INT32_MAX); auto cdims = res.out_dims;
//        dispatch_type(outMat.type, outMat.buffer, [&](auto* out_data) {
//            dispatch_type(inMat.type, inMat.buffer, [&](auto* in_data) {
//                using DstT = std::decay_t<decltype(*out_data)>;
//                // us stands for unsafe and fast subscripting so it doesnt
//                suppor negative indices and is super fast. if (cdims == 1) {
//                    for (uint32_t i = 0; i < inMat.total_size; i++) {
//                        out_data[i * res.stridesA[0]] = static_cast<DstT>(
//                        in_data[i * res.stridesB[0]] );
//                    }
//                } else if (cdims == 2) {
//                    for (uint32_t i = 0; i < res.shape[0]; i++) {
//                        for (uint32_t j = 0; j < res.shape[1]; j++) {
//                            out_data[i * res.stridesA[0] + j *
//                            res.stridesA[1]] = static_cast<DstT>( in_data[i *
//                            res.stridesB[0] + j * res.stridesB[1]] ); }
//                    }
//                } else if (cdims == 3) {
//                    for (uint32_t i = 0; i < res.shape[0]; i++) {
//                        for (uint32_t j = 0; j < res.shape[1]; j++) {
//                            for (uint32_t k = 0; k < res.shape[2]; k++) {
//                                out_data[i * res.stridesA[0] + j *
//                                res.stridesA[1] + k * res.stridesA[2]] =
//                                static_cast<DstT>( in_data[i * res.stridesB[0]
//                                + j * res.stridesB[1] + k * res.stridesB[2]]
//                                );
//                            }
//                        }
//                    }
//
//                } else {
//                    uint32_t outer_iterations = 1;
//                    for (uint32_t o = 0; o <= cdims - 4; o++) {
//                        outer_iterations *= res.shape[o];
//                    }
//                    for (uint32_t o = 0; o < outer_iterations; o++) {
//                        uint32_t inMatIndex = 0;
//                        uint32_t outMatIndex = 0;
//                        uint32_t rem = o;
//                        for (int i = cdims-4; i >=0; i--) {
//                            inMatIndex  += (rem % res.shape[i]) *
//                            res.stridesB[i]; outMatIndex += (rem %
//                            res.shape[i]) * res.stridesA[i]; rem /=
//                            res.shape[i];
//                        }
//                        for (uint32_t i = 0; i < res.shape[cdims-3]; i++) {
//                            for (uint32_t j = 0; j < res.shape[cdims-2]; j++)
//                            {
//                                for (uint32_t k = 0; k < res.shape[cdims-1];
//                                k++) {
//                                    out_data[outMatIndex + i *
//                                    res.stridesA[cdims-3] + j *
//                                    res.stridesA[cdims-2] + k *
//                                    res.stridesA[cdims-1]] =
//                                    static_cast<DstT>( in_data[inMatIndex + i
//                                    * res.stridesB[cdims-3] + j *
//                                    res.stridesB[cdims-2] + k *
//                                    res.stridesB[cdims-1]] );
//                                }
//                            }
//                        }
//                    }
//                }
//            });
//        });
//    }

void matrix::begin_refcount() {
    if (flags & NON_OWNERSHIP_FLAG)
        throw std::runtime_error(
                                 "matrix: cannot begin refcount on non-owning matrix");
    if (refCount != nullptr)
        return;
    refCount = new std::atomic<uint32_t>(1);
}

matrix matrix::stack(const std::vector<matrix>& mats, int axis) {
    if (axis < 0) axis += mats[0].dims + 1;
#ifdef SAFE_MODE
    if (output.dims != mats[0].dims + 1) {
        throw std::invalid_argument("matrix::stack: output dims must be input dims + 1");
    }
    for (size_t i = 1; i < mats.size(); i++) {
        if (mats[i].dims != mats[0].dims) {
            throw std::invalid_argument("matrix::stack: all input matrices must have same dims");
        }
        if (mats[i].type != mats[0].type) {
            throw std::invalid_argument("matrix::stack: all input matrices must have same type");
        }
        for (uint32_t d = 0; d < mats[0].dims; d++) {
            if (mats[i].shape()[d] != mats[0].shape()[d]) {
                throw std::invalid_argument("matrix::stack: all input matrices must have same shape");
            }
        }
    }
    if (axis < 0 || (uint32_t)axis > mats[0].dims) {
        throw std::invalid_argument("matrix::stack: axis out of range");
    }
#endif
    
    matrix output(mats[0].dims + 1, mats[0].type);
    
    size_m* out_shape = output.shape();
    memcpy(out_shape, mats[0].shape(), axis * sizeof(size_m));
    memcpy(out_shape + (axis + 1), mats[0].shape() + (axis), (mats[0].dims - axis) * sizeof(size_m));
    out_shape[axis] = (size_m)mats.size();
    output.total_size = mats[0].total_size * mats.size();
    output.calcStrides();
    
    output.tape = new StackPrimitive(mats, axis);
    
    return output;
}
void matrix::stack(const std::vector<matrix>& mats, matrix& output, int axis, ExecutionDevice exec_device) {
    if (axis < 0) axis += mats[0].dims;
    
    size_m* out_shape = output.shape();
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = output.total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    output.flags |= NON_CONTIGUOUS_FLAG;
    out_shape[axis] = 1;
    output.total_size = mats[0].total_size;
    
    matrix View(output.dims, output.type);
    memcpy(View.shape(), out_shape, (output.dims+1) * sizeof(size_m));
    View.shape()[axis] = 1;
    View.calcStrides();
    View.total_size = mats[0].total_size;
    View.flags |= NON_OWNERSHIP_FLAG;
    
    size_m offset = output.strides()[axis];
    if (exec_device == ExecutionDevice::METAL) {
        for (int i = 0; i < mats.size(); i++) {
            View.buffer = mats[i].buffer;
            View.metalBuffer = mats[i].metalBuffer;
            copyGPUinplace(output, View, i * offset, Execution::Encode);
        }
    } else {
        for (int i = 0; i < mats.size(); i++) {
            View.buffer = mats[i].buffer;
            View.metalBuffer = mats[i].metalBuffer;
            copyCPUinplace(output, View, i * offset);
        }
    }
    
    output.shape()[axis] = (size_m)mats.size();
    output.total_size = mats[0].total_size * mats.size();
    output.flags &= ~NON_CONTIGUOUS_FLAG;
}

matrix matrix::concat(const std::vector<matrix>& mats, int axis) {
    if (axis < 0) axis+= mats[0].dims;
    matrix output(mats[0].dims, mats[0].type);
    memcpy(output.shape(), mats[0].shape(), sizeof(size_m) * output.dims);
    for (int i = 1; i < mats.size(); i++) {
        output.shape()[axis] += mats[i].shape()[axis];
    }
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);
    output.tape = new ConcatPrimitive(mats, axis);
    return output;
}

void matrix::concat(const std::vector<matrix>& mats, matrix& output, int axis, ExecutionDevice exec_device) {
#ifdef SAFE_MODE
    assert(mats.size() > 0 && "concat: empty input vector");
    assert(axis >= 0 && axis < mats[0].dims && "concat: axis out of bounds");
    
    for (int i = 1; i < mats.size(); i++) {
        assert(mats[i].dims == mats[0].dims && "concat: all inputs must have same ndim");
        assert(mats[i].type == mats[0].type && "concat: all inputs must have same dtype");
        for (int d = 0; d < mats[0].dims; d++) {
            if (d == axis) continue;
            assert(mats[i].shape()[d] == mats[0].shape()[d] && "concat: shape mismatch on non-axis dim");
        }
    }
#endif
    
    
    matrix View(output.dims, output.type);
    View.flags |= NON_OWNERSHIP_FLAG;
    View.flags |= NON_CONTIGUOUS_FLAG;
    View.flags |= NON_CONTIGUOUS_FLAG;
    size_m prev_outaxis_shape = output.shape()[axis];
    size_m prev_out_size = output.total_size;
    size_m offset = 0;
    if (exec_device == ExecutionDevice::METAL) {
        for (int i = 0; i < mats.size(); i++) {
            View.set_array_desc(mats[i].array_desc);
            View.total_size = mats[i].total_size;
            View.buffer = mats[i].buffer;
            View.metalBuffer = mats[i].metalBuffer;
            output.shape()[axis] = View.shape()[axis];
            output.total_size = View.total_size;
            copyGPUinplace(output, View, offset, Execution::Encode);
            offset += View.accumul(axis, output.dims);
        }
    } else {
        for (int i = 0; i < mats.size(); i++) {
            View.set_array_desc(mats[i].array_desc);
            View.total_size = mats[i].total_size;
            View.buffer = mats[i].buffer;
            View.metalBuffer = mats[i].metalBuffer;
            output.shape()[axis] = View.shape()[axis];
            output.total_size = View.total_size;
            copyCPUinplace(output, View, offset);
            offset += View.accumul(axis, output.dims);
        }
    }
    
    output.shape()[axis] = prev_outaxis_shape;
    output.total_size = prev_out_size;
    output.flags &= ~NON_CONTIGUOUS_FLAG;
}

void matrix::SumNoRed(matrix& output, int axis, EvalType eval_type) {
    if (dims == 0) {
        if (eval_type == EvalType::EVAL_AUTO) {
            output.total_size = 1;
            // Reusing SumPrimitive is perfect here so the autograd tape still connects
            output.tape = new SumPrimitive(*this, axis, true);
        } else {
            // The mathematical sum of a scalar is just the scalar itself.
            // Copy the single element natively on the GPU without a sum kernel.
            copyGPUinplace(output, *this, 0, Execution::Encode);
        }
        return;
    }
    if (eval_type == EvalType::EVAL_AUTO) {
        if (axis < 0){
            axis += dims;
        }
        if ((dims-1 < axis)) {
            std::cerr << "MatrixH: Axis should not excede Dims of " << dims << "\n";
            throw;
        }
        
        
        if (!GlobalGPUManager.SumInit[(int)type]) {
            GlobalGPUManager.initSum_All((int)type);
        }
        
        memcpy(output.shape(), shape(), dims * sizeof(size_m));
        output.shape()[axis] = 1;
        output.total_size = output.accumul(0, dims);
        output.calcStrides();
        output.tape = new SumPrimitive(*this, axis, true);
        return;
    }
        
    size_m ElStride = accumul(axis+1, dims);

    size_m noOfOpp = shape()[axis];
    
    size_m axisStride;
    if (axis != dims-1) {
        axisStride = 1;
    }
    else {
        axisStride = shape()[axis];
    }
    

//    std::cout << "AxStride: " << axisStride << " ElStride: " << ElStride << " noOfOpp: " << noOfOpp << "\n";
//    printArray(inputStrides, dims-1);
//    printArray(maskedStrides, dims-1);

    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();

    auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
    auto _dispatchExecutionSize =  MTLSizeMake(output.total_size, 1, 1);

    size_t outputDims = dims -1;
    [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
    setBufferOrBytes(commandEncoder, *this, 1);
    
    [commandEncoder setBytes:&axisStride length: sizeof(size_m) atIndex:2];
    [commandEncoder setBytes:&ElStride length: sizeof(size_m) atIndex:3];
    [commandEncoder setBytes:&noOfOpp length: sizeof(size_m) atIndex:4];
    if (dims > 1) {
        size_m inputStrides[dims-1];
        size_t acc = 1;
        for (int i = dims-2; i >= 0; i--) {
            inputStrides[i] = acc;
            acc *= output.shape()[i];
        }

        size_m maskedStrides[dims-1];
        memcpy(maskedStrides, inputStrides, sizeof(size_m) * (dims-1));

        acc = 1;
        for (int i = 0; i < axis; i++) {
            maskedStrides[i] *= shape()[axis];
        }
        [commandEncoder setBytes:&inputStrides length: (dims-1) * sizeof(size_m) atIndex:5];
        [commandEncoder setBytes:&maskedStrides length: (dims-1)* sizeof(size_m) atIndex:6];
    } else {
        size_m inpandmaskedStrides = 1;
        [commandEncoder setBytes:&inpandmaskedStrides length: sizeof(size_m) atIndex:5];
        [commandEncoder setBytes:&inpandmaskedStrides length:sizeof(size_m) atIndex:6];
    }
    [commandEncoder setBytes:&outputDims length:  sizeof(size_m) atIndex:7];
    [commandEncoder setComputePipelineState:GlobalGPUManager.SumComputeState[(int)type]];
    [commandEncoder dispatchThreads:_dispatchExecutionSize
              threadsPerThreadgroup:_threadsPerThreadgroup];

}

void matrix::sum_legacy(matrix& output, int axis, bool keepdims, EvalType eval_type) {
    if (!GlobalGPUManager.SumInit[(int)type]) {
        GlobalGPUManager.initSum_All((int)type);
    }
    
    if (dims == 0) {
        if (eval_type == EvalType::EVAL_AUTO) {
            output.total_size = 1;
            // Reusing SumPrimitive is perfect here so the autograd tape still connects
            output.tape = new SumPrimitive(*this, axis, keepdims);
        } else {
            // The mathematical sum of a scalar is just the scalar itself.
            // Copy the single element natively on the GPU without a sum kernel.
            copyGPUinplace(output, *this, 0, Execution::Encode);
        }
        return;
    }
    
    if (eval_type == EvalType::EVAL_AUTO) {
        if (axis < 0){
            axis += dims;
        }
        if ((dims-1 < axis)) {
            std::cerr << "MatrixH: Axis should not excede Dims of " << dims << "\n";
            throw;
        }
        
        

        if (!keepdims) {
            memcpy(output.shape(), shape(), axis * sizeof(size_m));
            memcpy(output.shape() + axis, shape() + axis + 1, (dims-axis) * sizeof(size_m));
            output.total_size = output.accumul(0, dims-1);
        } else {
            memcpy(output.shape(), shape(), dims * sizeof(size_m));
            output.shape()[axis] = 1;
            output.total_size = output.accumul(0, dims);
        }
        output.calcStrides();
        output.tape = new SumPrimitive(*this, axis, false);
        return;
    }
    
    size_m ElStride = (size_m)accumul(axis+1, dims);

    size_m noOfOpp = shape()[axis];
    size_m axisStride;
    if (axis != dims-1) {
        axisStride = 1;
    }
    else {
        axisStride = shape()[axis];
    }


//    std::cout << "AxStride: " << axisStride << " ElStride: " << ElStride << " noOfOpp: " << noOfOpp << "\n";
//    printArray(inputStrides, dims-1);
//    printArray(maskedStrides, dims-1);

    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();

    auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
    auto _dispatchExecutionSize =  MTLSizeMake(output.total_size, 1, 1);

    size_t outputDims = dims - 1;
    [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
    setBufferOrBytes(commandEncoder, *this, 1);
    
    [commandEncoder setBytes:&axisStride length: sizeof(size_m) atIndex:2];
    [commandEncoder setBytes:&ElStride length: sizeof(size_m) atIndex:3];
    [commandEncoder setBytes:&noOfOpp length: sizeof(size_m) atIndex:4];
    if (dims > 1) {
        size_m inputStrides[dims-1];
        size_t acc = 1;
        for (int i = dims-2; i >= 0; i--) {
            inputStrides[i] = acc;
            acc *= output.shape()[i];
        }

        size_m maskedStrides[dims-1];
        memcpy(maskedStrides, inputStrides, sizeof(size_m) * (dims-1));

        acc = 1;
        for (int i = 0; i < axis; i++) {
            maskedStrides[i] *= shape()[axis];
        }
        [commandEncoder setBytes:&inputStrides length: (dims-1) * sizeof(size_m) atIndex:5];
        [commandEncoder setBytes:&maskedStrides length: (dims-1)* sizeof(size_m) atIndex:6];
    } else {
        size_m inpandmaskedStrides = 1;
        [commandEncoder setBytes:&inpandmaskedStrides length: sizeof(size_m) atIndex:5];
        [commandEncoder setBytes:&inpandmaskedStrides length:sizeof(size_m) atIndex:6];
    }
    [commandEncoder setBytes:&outputDims length:  sizeof(size_m) atIndex:7];
    [commandEncoder setComputePipelineState:GlobalGPUManager.SumComputeState[(int)type]];
    [commandEncoder dispatchThreads:_dispatchExecutionSize
              threadsPerThreadgroup:_threadsPerThreadgroup];

}

matrix matrix::sum_legacy(int axis, bool keepdims) {
    if (dims == 0) {
        matrix output(0, type); // Summing a scalar yields a scalar
        sum_legacy(output, axis, keepdims);
        return output;
    }
    int outputDims = keepdims ? dims : dims-1;
    matrix output(outputDims, type);
    sum_legacy(output, axis, keepdims);
    return output;
}

matrix matrix::unbroadcast_shape(const size_m* target_shape, int target_dims) const {
    matrix result = *this;
    int rank_diff = dims - target_dims;
    
    if (rank_diff > 0) {
        result = result.sum(0, rank_diff - 1, false);
    }
    
    for (int i = 0; i < target_dims; i++) {
        if (target_shape[i] == 1 && result.shape()[i] != 1) {
            matrix summed(result.dims, result.type);
            result.SumNoRed(summed, i);
            result = summed;
        }
    }
    
    return result;
}


matrix matrix::conv1d(const matrix& input, const matrix& kernel, int padding, int stride, int dilation, int groups) {
    // input shape [batch, X, channels]
    // kernel shape [out_channel, x, in_channel]
    // output shape [batch, X' , out_channel]
    // in_channel = channels/groups
    matrix output(input.dims, input.type);
    output.shape()[0] = input.shape()[0];
    output.shape()[1] = (input.shape()[1] + 2 * padding - dilation * (kernel.shape()[1] - 1) - 1) / stride + 1;
    output.shape()[2] = kernel.shape()[0];
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);
    output.tape = new ConvolvePrimitive(input, kernel, {padding}, {stride}, {dilation}, groups);
    return output;
}

matrix matrix::conv2d(const matrix& input, const matrix& kernel, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w, int groups) {
    // input shape [batch, H, W, channels]
    // kernel shape [out_channel, h, w, in_channel]
    // output shape [batch, H', W', out_channel]
    // in_channel = channels/groups
    matrix output(input.dims, input.type);
    output.shape()[0] = input.shape()[0];
    output.shape()[1] = (input.shape()[1] + 2 * pad_h - dilation_h * (kernel.shape()[1] - 1) - 1) / stride_h + 1;
    output.shape()[2] = (input.shape()[2] + 2 * pad_w - dilation_w * (kernel.shape()[2] - 1) - 1) / stride_w + 1;
    output.shape()[3] = kernel.shape()[0];
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);
    output.tape = new ConvolvePrimitive(input, kernel, {pad_h, pad_w}, {stride_h, stride_w}, {dilation_h, dilation_w}, groups);
    return output;
}

matrix matrix::conv3d(const matrix& input, const matrix& kernel, int pad_d, int pad_h, int pad_w, int stride_d, int stride_h, int stride_w, int dilation_d, int dilation_h, int dilation_w, int groups) {
    // input shape [batch, D, H, W, channels]
    // kernel shape [out_channel, d, h, w, in_channel]
    // output shape [batch, D', H', W', out_channel]
    // in_channel = channels/groups
    matrix output(input.dims, input.type);
    output.shape()[0] = input.shape()[0];
    output.shape()[1] = (input.shape()[1] + 2 * pad_d - dilation_d * (kernel.shape()[1] - 1) - 1) / stride_d + 1;
    output.shape()[2] = (input.shape()[2] + 2 * pad_h - dilation_h * (kernel.shape()[2] - 1) - 1) / stride_h + 1;
    output.shape()[3] = (input.shape()[3] + 2 * pad_w - dilation_w * (kernel.shape()[3] - 1) - 1) / stride_w + 1;
    output.shape()[4] = kernel.shape()[0];
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);
    output.tape = new ConvolvePrimitive(input, kernel, {pad_d, pad_h, pad_w}, {stride_d, stride_h, stride_w}, {dilation_d, dilation_h, dilation_w}, groups);
    return output;
}

void matrix::conv1d_gpu(const matrix& kernel, matrix& output) {
    
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
    
    int typeCode = (int)type;
    if (!GlobalGPUManager.Conv1dInit[typeCode]) {
        GlobalGPUManager.initConv1d(typeCode);
    }
    [commandEncoder setComputePipelineState:GlobalGPUManager.Conv1dComputeState[typeCode]];
    
    [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
    setBufferOrBytes(commandEncoder, *this, 1);
    setBufferOrBytes(commandEncoder, kernel, 2);
    
    [commandEncoder setBytes:this->shape() length:this->dims * sizeof(size_m) atIndex:3];
    [commandEncoder setBytes:kernel.shape() length:kernel.dims * sizeof(size_m) atIndex:4];
    [commandEncoder setBytes:output.shape() length:output.dims * sizeof(size_m) atIndex:5];
    
    ConvolvePrimitive* prim = (ConvolvePrimitive*)output.tape;
    [commandEncoder setBytes:prim->padding.data() length:prim->padding.size() * sizeof(int) atIndex:6];
    [commandEncoder setBytes:prim->stride.data() length:prim->stride.size() * sizeof(int) atIndex:7];
    [commandEncoder setBytes:prim->dilation.data() length:prim->dilation.size() * sizeof(int) atIndex:8];
    [commandEncoder setBytes:&prim->groups length:sizeof(int) atIndex:9];
    
    [commandEncoder setBytes:this->strides() length:this->dims * sizeof(size_m) atIndex:10];
    [commandEncoder setBytes:kernel.strides() length:kernel.dims * sizeof(size_m) atIndex:11];
    [commandEncoder setBytes:output.strides() length:output.dims * sizeof(size_m) atIndex:12];
    
    auto max_threads = [GlobalGPUManager.Conv1dComputeState[typeCode] maxTotalThreadsPerThreadgroup];
    NSUInteger w = MIN(16, max_threads);
    NSUInteger h = MIN(16, max_threads / w);
    auto _dispatchExecutionSize = MTLSizeMake(output.shape()[2], output.shape()[1], output.shape()[0]);
    auto _threadsPerThreadgroup = MTLSizeMake(w, h, 1);
    
    [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
}

void matrix::conv2d_gpu(const matrix& kernel, matrix& output) {
    
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
    
    int typeCode = (int)type;
    if (!GlobalGPUManager.Conv2dInit[typeCode]) {
        GlobalGPUManager.initConv2d(typeCode);
    }
    [commandEncoder setComputePipelineState:GlobalGPUManager.Conv2dComputeState[typeCode]];
    
    [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
    setBufferOrBytes(commandEncoder, *this, 1);
    setBufferOrBytes(commandEncoder, kernel, 2);
    
    [commandEncoder setBytes:this->shape() length:this->dims * sizeof(size_m) atIndex:3];
    [commandEncoder setBytes:kernel.shape() length:kernel.dims * sizeof(size_m) atIndex:4];
    [commandEncoder setBytes:output.shape() length:output.dims * sizeof(size_m) atIndex:5];
    
    ConvolvePrimitive* prim = (ConvolvePrimitive*)output.tape;
    [commandEncoder setBytes:prim->padding.data() length:prim->padding.size() * sizeof(int) atIndex:6];
    [commandEncoder setBytes:prim->stride.data() length:prim->stride.size() * sizeof(int) atIndex:7];
    [commandEncoder setBytes:prim->dilation.data() length:prim->dilation.size() * sizeof(int) atIndex:8];
    [commandEncoder setBytes:&prim->groups length:sizeof(int) atIndex:9];
    
    [commandEncoder setBytes:this->strides() length:this->dims * sizeof(size_m) atIndex:10];
    [commandEncoder setBytes:kernel.strides() length:kernel.dims * sizeof(size_m) atIndex:11];
    [commandEncoder setBytes:output.strides() length:output.dims * sizeof(size_m) atIndex:12];
    
    auto max_threads = [GlobalGPUManager.Conv2dComputeState[typeCode] maxTotalThreadsPerThreadgroup];
    NSUInteger w = MIN(16, max_threads);
    NSUInteger h = MIN(16, max_threads / w);
    auto _dispatchExecutionSize = MTLSizeMake(output.shape()[3], output.shape()[2], output.shape()[1] * output.shape()[0]);
    auto _threadsPerThreadgroup = MTLSizeMake(w, h, 1);
    
    [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
}

void matrix::conv3d_gpu(const matrix& kernel, matrix& output) {
    
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
    
    int typeCode = (int)type;
    if (!GlobalGPUManager.Conv3dInit[typeCode]) {
        GlobalGPUManager.initConv3d(typeCode);
    }
    [commandEncoder setComputePipelineState:GlobalGPUManager.Conv3dComputeState[typeCode]];
    
    [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
    setBufferOrBytes(commandEncoder, *this, 1);
    setBufferOrBytes(commandEncoder, kernel, 2);
    
    [commandEncoder setBytes:this->shape() length:this->dims * sizeof(size_m) atIndex:3];
    [commandEncoder setBytes:kernel.shape() length:kernel.dims * sizeof(size_m) atIndex:4];
    [commandEncoder setBytes:output.shape() length:output.dims * sizeof(size_m) atIndex:5];
    
    ConvolvePrimitive* prim = (ConvolvePrimitive*)output.tape;
    [commandEncoder setBytes:prim->padding.data() length:prim->padding.size() * sizeof(int) atIndex:6];
    [commandEncoder setBytes:prim->stride.data() length:prim->stride.size() * sizeof(int) atIndex:7];
    [commandEncoder setBytes:prim->dilation.data() length:prim->dilation.size() * sizeof(int) atIndex:8];
    [commandEncoder setBytes:&prim->groups length:sizeof(int) atIndex:9];
    
    [commandEncoder setBytes:this->strides() length:this->dims * sizeof(size_m) atIndex:10];
    [commandEncoder setBytes:kernel.strides() length:kernel.dims * sizeof(size_m) atIndex:11];
    [commandEncoder setBytes:output.strides() length:output.dims * sizeof(size_m) atIndex:12];
    
    auto max_threads = [GlobalGPUManager.Conv3dComputeState[typeCode] maxTotalThreadsPerThreadgroup];
    auto _dispatchExecutionSize = MTLSizeMake(output.total_size, 1, 1);
    auto _threadsPerThreadgroup = MTLSizeMake(max_threads, 1, 1);
    
    [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
}

void matrix::unbrodcast(matrix& output, matrix& target) {
    output = unbroadcast_shape(target.shape(), target.dims);
}

matrix matrix::clamp(double min_val, double max_val) const {
    dtype out_type = this->type;
    matrix output(this->dims, out_type);
    memcpy(output.shape(), this->shape(), this->dims * sizeof(size_m));
    output.total_size = this->total_size;
    output.calcStrides();
    
    auto res = collapse_dims(this->shape(), output.strides(), this->strides(), this->dims, INT32_MAX);
    ClampPrimitive* prim = new ClampPrimitive(*this, min_val, max_val);
    prim->collapsed_dims = res;
    output.tape = prim;
    return output;
}

void matrix::clamp(matrix& output, double min_val, double max_val, ExecutionDevice exec_device) const {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    ClampPrimitive* primit = static_cast<ClampPrimitive*>(output.tape);
    auto res = primit->collapsed_dims;
    uint32_t cdims = res.out_dims;

    if (exec_device == ExecutionDevice::METAL) {
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        int typeCode = (int)type;
        int kernel_code = cdims > 3 ? 3 : (cdims == 0 ? 0 : cdims - 1);

        if (!GlobalGPUManager.ClampInit_nd[typeCode][kernel_code]) {
            GlobalGPUManager.initClamp_nd(typeCode, kernel_code);
        }

        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0]; // Output destination
        setBufferOrBytes(commandEncoder, *this, 1); // Input is the normal contiguous/strided source
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);

        if (cdims == 0) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.ClampComputeState_nd[typeCode][0]];
            size_m one = 1;
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:3];
            if (type == dtype::Float) {
                float min_val_f = (float)min_val;
                float max_val_f = (float)max_val;
                [commandEncoder setBytes:&min_val_f length:sizeof(float) atIndex:4];
                [commandEncoder setBytes:&max_val_f length:sizeof(float) atIndex:5];
            } else if (type == dtype::Int32) {
                int min_val_i = (int)min_val;
                int max_val_i = (int)max_val;
                [commandEncoder setBytes:&min_val_i length:sizeof(int) atIndex:4];
                [commandEncoder setBytes:&max_val_i length:sizeof(int) atIndex:5];
            }
            _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        } else if (cdims == 1) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.ClampComputeState_nd[typeCode][0]];
            [commandEncoder setBytes:res.stridesA length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[0], 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.ClampComputeState_nd[typeCode][1]];
            [commandEncoder setBytes:res.stridesA length:2 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:2 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[1], res.shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.ClampComputeState_nd[typeCode][2]];
            [commandEncoder setBytes:res.stridesA length:3 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:3 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[2], res.shape[1], res.shape[0]);
        } else {
            [commandEncoder setComputePipelineState:GlobalGPUManager.ClampComputeState_nd[typeCode][3]];
            [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:cdims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:res.shape length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:5];
            
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= res.shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(res.shape[cdims-1], res.shape[cdims-2], acc);
        }
        
        dispatch_type(this->type, (void*)1, [&](auto* dummy) {
            using T = std::decay_t<decltype(*dummy)>;
            T min_t = static_cast<T>(min_val);
            T max_t = static_cast<T>(max_val);
            [commandEncoder setBytes:&min_t length:sizeof(T) atIndex:6];
            [commandEncoder setBytes:&max_t length:sizeof(T) atIndex:7];
        });

        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        dispatch_type(this->type, output.buffer, [&](auto* out_data) {
            using T = std::decay_t<decltype(*out_data)>;
            T* in_data = (T*)this->buffer;
            T min_t = static_cast<T>(min_val);
            T max_t = static_cast<T>(max_val);

            if (cdims == 1) {
                for (size_m i = 0; i < res.shape[0]; ++i) {
                    T val = in_data[i * res.stridesB[0]];
                    out_data[i * res.stridesA[0]] = std::max(min_t, std::min(val, max_t));
                }
            } else if (cdims == 2) {
                for (size_m i = 0; i < res.shape[0]; ++i) {
                    for (size_m j = 0; j < res.shape[1]; ++j) {
                        T val = in_data[i * res.stridesB[0] + j * res.stridesB[1]];
                        out_data[i * res.stridesA[0] + j * res.stridesA[1]] = std::max(min_t, std::min(val, max_t));
                    }
                }
            } else if (cdims == 3) {
                for (size_m i = 0; i < res.shape[0]; ++i) {
                    for (size_m j = 0; j < res.shape[1]; ++j) {
                        for (size_m k = 0; k < res.shape[2]; ++k) {
                            T val = in_data[i * res.stridesB[0] + j * res.stridesB[1] + k * res.stridesB[2]];
                            out_data[i * res.stridesA[0] + j * res.stridesA[1] + k * res.stridesA[2]] = std::max(min_t, std::min(val, max_t));
                        }
                    }
                }
            } else {
                size_m* coords = new size_m[cdims];
                std::fill(coords, coords + cdims, 0);

                for (size_m i = 0; i < output.total_size; ++i) {
                    size_m in_idx = 0;
                    size_m out_idx = 0;
                    for (int d = 0; d < cdims; ++d) {
                        in_idx += coords[d] * res.stridesB[d];
                        out_idx += coords[d] * res.stridesA[d];
                    }
                    
                    T val = in_data[in_idx];
                    out_data[out_idx] = std::max(min_t, std::min(val, max_t));

                    for (int d = cdims - 1; d >= 0; --d) {
                        coords[d]++;
                        if (coords[d] < res.shape[d]) {
                            break;
                        }
                        coords[d] = 0;
                    }
                }
                delete[] coords;
            }
        });
    }
}

matrix matrix::abs(const matrix& input) {
    if (input.type == dtype::UInt8 || input.type == dtype::UInt16 || input.type == dtype::UInt32) {
        return input;
    }

    dtype out_type = input.type;
    matrix output(input.dims, out_type);
    memcpy(output.shape(), input.shape(), input.dims * sizeof(size_m));
    output.total_size = input.total_size;
    output.calcStrides();
    auto res = collapse_dims(input.shape(), output.strides(), input.strides(), input.dims, INT32_MAX);
    AbsPrimitive* prim = new AbsPrimitive(input);
    prim->collapsed_dims = res;
    output.tape = prim;
    return output;
}

matrix matrix::log(const matrix& input) {
    dtype out_type = promote_types(input.type, dtype::Float16);
    matrix promoted = (input.type != out_type) ? input.astype(out_type, true) : input;
    matrix output(promoted.dims, promoted.type);
    memcpy(output.shape(), promoted.shape(), promoted.dims * sizeof(size_m));
    output.total_size = promoted.total_size;
    output.calcStrides();
    
    auto res = collapse_dims(promoted.shape(), output.strides(), promoted.strides(), promoted.dims, INT32_MAX);
    LogPrimitive* prim = new LogPrimitive(promoted);
    prim->collapsed_dims = res;
    output.tape = prim;
    return output;
}

void matrix::log(matrix& output, ExecutionDevice exec_device) {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    LogPrimitive* primit = static_cast<LogPrimitive*>(output.tape);
    auto res = primit->collapsed_dims;
    uint32_t cdims = res.out_dims;

    if (exec_device == ExecutionDevice::METAL) {
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        int typeCode = (int)type;
        int kernel_code = cdims > 3 ? 3 : (cdims == 0 ? 0 : cdims - 1);

        if (!GlobalGPUManager.LogInit[typeCode][kernel_code]) {
            GlobalGPUManager.initLog(typeCode, kernel_code);
        }

        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0]; // Output destination
        setBufferOrBytes(commandEncoder, *this, 1); // Input is the normal contiguous/strided source
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);

        if (cdims == 0) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.LogComputeState[typeCode][0]];
            size_m one = 1;
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        } else if (cdims == 1) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.LogComputeState[typeCode][0]];
            [commandEncoder setBytes:res.stridesA length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[0], 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.LogComputeState[typeCode][1]];
            [commandEncoder setBytes:res.stridesA length:2 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:2 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[1], res.shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.LogComputeState[typeCode][2]];
            [commandEncoder setBytes:res.stridesA length:3 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:3 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[2], res.shape[1], res.shape[0]);
        } else {
            [commandEncoder setComputePipelineState:GlobalGPUManager.LogComputeState[typeCode][3]];
            [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:cdims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:res.shape length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:5];
            
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= res.shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(res.shape[cdims-1], res.shape[cdims-2], acc);
        }

        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        // CPU fallback mapping
        dispatch_type(type, output.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
                T* in_data = (T*)buffer;
                if (cdims == 0) {
                    out_data[0] = std::log(in_data[0]);
                } else if (cdims == 1) {
                    size_m out_stride = res.stridesA[0];
                    size_m in_stride = res.stridesB[0];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        out_data[i * out_stride] = std::log(in_data[i * in_stride]);
                    }
                } else if (cdims == 2) {
                    size_m out_stride0 = res.stridesA[0];
                    size_m out_stride1 = res.stridesA[1];
                    size_m in_stride0 = res.stridesB[0];
                    size_m in_stride1 = res.stridesB[1];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        for (size_m j = 0; j < res.shape[1]; j++) {
                            out_data[i * out_stride0 + j * out_stride1] = std::log(in_data[i * in_stride0 + j * in_stride1]);
                        }
                    }
                } else if (cdims == 3) {
                    size_m out_stride0 = res.stridesA[0];
                    size_m out_stride1 = res.stridesA[1];
                    size_m out_stride2 = res.stridesA[2];
                    size_m in_stride0 = res.stridesB[0];
                    size_m in_stride1 = res.stridesB[1];
                    size_m in_stride2 = res.stridesB[2];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        for (size_m j = 0; j < res.shape[1]; j++) {
                            for (size_m k = 0; k < res.shape[2]; k++) {
                                out_data[i * out_stride0 + j * out_stride1 + k * out_stride2] = std::log(in_data[i * in_stride0 + j * in_stride1 + k * in_stride2]);
                            }
                        }
                    }
                } else {
                    for (size_m gid = 0; gid < total_size; gid++) {
                        size_m src_idx = 0;
                        size_m dst_idx = 0;
                        size_m rem = gid;
                        for (int i = cdims - 1; i >= 0; i--) {
                            size_m idx = rem % res.shape[i];
                            src_idx += idx * res.stridesB[i];
                            dst_idx += idx * res.stridesA[i];
                            rem /= res.shape[i];
                        }
                        out_data[dst_idx] = std::log(in_data[src_idx]);
                    }
                }
        });
    }
}
void matrix::abs(matrix& output, ExecutionDevice exec_device) {
    // only for eval_type = execution
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    AbsPrimitive* primit = static_cast<AbsPrimitive*>(output.tape);
    auto res = primit->collapsed_dims;
    uint32_t cdims = res.out_dims;

    if (exec_device == ExecutionDevice::METAL) {
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        int typeCode = (int)type;
        int kernel_code = cdims > 3 ? 3 : (cdims == 0 ? 0 : cdims - 1);

        if (!GlobalGPUManager.AbsInit[typeCode][kernel_code]) {
            GlobalGPUManager.initAbs(typeCode, kernel_code);
        }

        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0]; // Output destination
        setBufferOrBytes(commandEncoder, *this, 1); // Input is the normal contiguous/strided source
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);

        if (cdims == 0) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.AbsComputeState[typeCode][0]];
            size_m one = 1;
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        } else if (cdims == 1) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.AbsComputeState[typeCode][0]];
            [commandEncoder setBytes:res.stridesA length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[0], 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.AbsComputeState[typeCode][1]];
            [commandEncoder setBytes:res.stridesA length:2 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:2 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[1], res.shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.AbsComputeState[typeCode][2]];
            [commandEncoder setBytes:res.stridesA length:3 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:3 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[2], res.shape[1], res.shape[0]);
        } else {
            [commandEncoder setComputePipelineState:GlobalGPUManager.AbsComputeState[typeCode][3]];
            [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:cdims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:res.shape length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:5];
            
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= res.shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(res.shape[cdims-1], res.shape[cdims-2], acc);
        }

        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        // CPU fallback mapping
//        dispatch_type(type, output.buffer, [&](auto *out_data) {
//            using T = std::decay_t<decltype(*out_data)>;
//                T* in_data = (T*)buffer;
//                if (cdims == 1) {
//                    size_m out_stride = res.stridesA[0];
//                    size_m in_stride = res.stridesB[0];
//                    for (size_m i = 0; i < res.shape[0]; i++) {
//                        out_data[i * out_stride] = std::abs(in_data[i * in_stride]);
//                    }
//                } else if (cdims == 2) {
//                    size_m out_stride0 = res.stridesA[0];
//                    size_m out_stride1 = res.stridesA[1];
//                    size_m in_stride0 = res.stridesB[0];
//                    size_m in_stride1 = res.stridesB[1];
//                    for (size_m i = 0; i < res.shape[0]; i++) {
//                        for (size_m j = 0; j < res.shape[1]; j++) {
//                            out_data[i * out_stride0 + j * out_stride1] = std::abs(in_data[i * in_stride0 + j * in_stride1]);
//                        }
//                    }
//                } else if (cdims == 3) {
//                    size_m out_stride0 = res.stridesA[0];
//                    size_m out_stride1 = res.stridesA[1];
//                    size_m out_stride2 = res.stridesA[2];
//                    size_m in_stride0 = res.stridesB[0];
//                    size_m in_stride1 = res.stridesB[1];
//                    size_m in_stride2 = res.stridesB[2];
//                    for (size_m i = 0; i < res.shape[0]; i++) {
//                        for (size_m j = 0; j < res.shape[1]; j++) {
//                            for (size_m k = 0; k < res.shape[2]; k++) {
//                                out_data[i * out_stride0 + j * out_stride1 + k * out_stride2] = std::abs(in_data[i * in_stride0 + j * in_stride1 + k * in_stride2]);
//                            }
//                        }
//                    }
//                } else {
//                    for (size_m gid = 0; gid < total_size; gid++) {
//                        size_m src_idx = 0;
//                        size_m dst_idx = 0;
//                        size_m rem = gid;
//                        for (int i = cdims - 1; i >= 0; i--) {
//                            size_m idx = rem % res.shape[i];
//                            src_idx += idx * res.stridesB[i];
//                            dst_idx += idx * res.stridesA[i];
//                            rem /= res.shape[i];
//                        }
//                        out_data[dst_idx] = std::abs(in_data[src_idx]);
//                    }
//                }
//        });
    }
}


matrix matrix::sin(const matrix& input) {
    dtype out_type = promote_types(input.type, dtype::Float16);
    matrix promoted = (input.type != out_type) ? input.astype(out_type, true) : input;
    matrix output(promoted.dims, promoted.type);
    memcpy(output.shape(), promoted.shape(), promoted.dims * sizeof(size_m));
    output.total_size = promoted.total_size;
    output.calcStrides();
    
    auto res = collapse_dims(promoted.shape(), output.strides(), promoted.strides(), promoted.dims, INT32_MAX);
    SinPrimitive* prim = new SinPrimitive(promoted);
    prim->collapsed_dims = res;
    output.tape = prim;
    return output;
}

void matrix::sin(matrix& output, ExecutionDevice exec_device) {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    SinPrimitive* primit = static_cast<SinPrimitive*>(output.tape);
    auto res = primit->collapsed_dims;
    uint32_t cdims = res.out_dims;

    if (exec_device == ExecutionDevice::METAL) {
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        int typeCode = (int)type;
        int kernel_code = cdims > 3 ? 3 : (cdims == 0 ? 0 : cdims - 1);

        if (!GlobalGPUManager.SinInit_nd[typeCode][kernel_code]) {
            GlobalGPUManager.initSin_nd(typeCode, kernel_code);
        }

        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0]; // Output destination
        setBufferOrBytes(commandEncoder, *this, 1); // Input is the normal contiguous/strided source
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);

        if (cdims == 0) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.SinComputeState_nd[typeCode][0]];
            size_m one = 1;
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        } else if (cdims == 1) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.SinComputeState_nd[typeCode][0]];
            [commandEncoder setBytes:res.stridesA length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[0], 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.SinComputeState_nd[typeCode][1]];
            [commandEncoder setBytes:res.stridesA length:2 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:2 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[1], res.shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.SinComputeState_nd[typeCode][2]];
            [commandEncoder setBytes:res.stridesA length:3 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:3 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[2], res.shape[1], res.shape[0]);
        } else {
            [commandEncoder setComputePipelineState:GlobalGPUManager.SinComputeState_nd[typeCode][3]];
            [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:cdims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:res.shape length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:5];
            
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= res.shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(res.shape[cdims-1], res.shape[cdims-2], acc);
        }

        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        // CPU fallback mapping
        dispatch_type(type, output.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
                T* in_data = (T*)buffer;
                if (cdims == 0) {
                    out_data[0] = std::sin(in_data[0]);
                } else if (cdims == 1) {
                    size_m out_stride = res.stridesA[0];
                    size_m in_stride = res.stridesB[0];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        out_data[i * out_stride] = std::sin(in_data[i * in_stride]);
                    }
                } else if (cdims == 2) {
                    size_m out_stride0 = res.stridesA[0];
                    size_m out_stride1 = res.stridesA[1];
                    size_m in_stride0 = res.stridesB[0];
                    size_m in_stride1 = res.stridesB[1];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        for (size_m j = 0; j < res.shape[1]; j++) {
                            out_data[i * out_stride0 + j * out_stride1] = std::sin(in_data[i * in_stride0 + j * in_stride1]);
                        }
                    }
                } else if (cdims == 3) {
                    size_m out_stride0 = res.stridesA[0];
                    size_m out_stride1 = res.stridesA[1];
                    size_m out_stride2 = res.stridesA[2];
                    size_m in_stride0 = res.stridesB[0];
                    size_m in_stride1 = res.stridesB[1];
                    size_m in_stride2 = res.stridesB[2];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        for (size_m j = 0; j < res.shape[1]; j++) {
                            for (size_m k = 0; k < res.shape[2]; k++) {
                                out_data[i * out_stride0 + j * out_stride1 + k * out_stride2] = std::sin(in_data[i * in_stride0 + j * in_stride1 + k * in_stride2]);
                            }
                        }
                    }
                } else {
                    for (size_m gid = 0; gid < total_size; gid++) {
                        size_m src_idx = 0;
                        size_m dst_idx = 0;
                        size_m rem = gid;
                        for (int i = cdims - 1; i >= 0; i--) {
                            size_m idx = rem % res.shape[i];
                            src_idx += idx * res.stridesB[i];
                            dst_idx += idx * res.stridesA[i];
                            rem /= res.shape[i];
                        }
                        out_data[dst_idx] = std::sin(in_data[src_idx]);
                    }
                }
        });
    }
}

matrix matrix::cos(const matrix& input) {
    dtype out_type = promote_types(input.type, dtype::Float16);
    matrix promoted = (input.type != out_type) ? input.astype(out_type, true) : input;
    matrix output(promoted.dims, promoted.type);
    memcpy(output.shape(), promoted.shape(), promoted.dims * sizeof(size_m));
    output.total_size = promoted.total_size;
    output.calcStrides();
    
    auto res = collapse_dims(promoted.shape(), output.strides(), promoted.strides(), promoted.dims, INT32_MAX);
    CosPrimitive* prim = new CosPrimitive(promoted);
    prim->collapsed_dims = res;
    output.tape = prim;
    return output;
}

void matrix::cos(matrix& output, ExecutionDevice exec_device) {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    CosPrimitive* primit = static_cast<CosPrimitive*>(output.tape);
    auto res = primit->collapsed_dims;
    uint32_t cdims = res.out_dims;

    if (exec_device == ExecutionDevice::METAL) {
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        int typeCode = (int)type;
        int kernel_code = cdims > 3 ? 3 : (cdims == 0 ? 0 : cdims - 1);

        if (!GlobalGPUManager.CosInit_nd[typeCode][kernel_code]) {
            GlobalGPUManager.initCos_nd(typeCode, kernel_code);
        }

        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0]; // Output destination
        setBufferOrBytes(commandEncoder, *this, 1); // Input is the normal contiguous/strided source
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);

        if (cdims == 0) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.CosComputeState_nd[typeCode][0]];
            size_m one = 1;
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        } else if (cdims == 1) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.CosComputeState_nd[typeCode][0]];
            [commandEncoder setBytes:res.stridesA length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[0], 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.CosComputeState_nd[typeCode][1]];
            [commandEncoder setBytes:res.stridesA length:2 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:2 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[1], res.shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.CosComputeState_nd[typeCode][2]];
            [commandEncoder setBytes:res.stridesA length:3 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:3 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[2], res.shape[1], res.shape[0]);
        } else {
            [commandEncoder setComputePipelineState:GlobalGPUManager.CosComputeState_nd[typeCode][3]];
            [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:cdims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:res.shape length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:5];
            
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= res.shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(res.shape[cdims-1], res.shape[cdims-2], acc);
        }

        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        // CPU fallback mapping
        dispatch_type(type, output.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
                T* in_data = (T*)buffer;
                if (cdims == 0) {
                    out_data[0] = std::cos(in_data[0]);
                } else if (cdims == 1) {
                    size_m out_stride = res.stridesA[0];
                    size_m in_stride = res.stridesB[0];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        out_data[i * out_stride] = std::cos(in_data[i * in_stride]);
                    }
                } else if (cdims == 2) {
                    size_m out_stride0 = res.stridesA[0];
                    size_m out_stride1 = res.stridesA[1];
                    size_m in_stride0 = res.stridesB[0];
                    size_m in_stride1 = res.stridesB[1];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        for (size_m j = 0; j < res.shape[1]; j++) {
                            out_data[i * out_stride0 + j * out_stride1] = std::cos(in_data[i * in_stride0 + j * in_stride1]);
                        }
                    }
                } else if (cdims == 3) {
                    size_m out_stride0 = res.stridesA[0];
                    size_m out_stride1 = res.stridesA[1];
                    size_m out_stride2 = res.stridesA[2];
                    size_m in_stride0 = res.stridesB[0];
                    size_m in_stride1 = res.stridesB[1];
                    size_m in_stride2 = res.stridesB[2];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        for (size_m j = 0; j < res.shape[1]; j++) {
                            for (size_m k = 0; k < res.shape[2]; k++) {
                                out_data[i * out_stride0 + j * out_stride1 + k * out_stride2] = std::cos(in_data[i * in_stride0 + j * in_stride1 + k * in_stride2]);
                            }
                        }
                    }
                } else {
                    for (size_m gid = 0; gid < total_size; gid++) {
                        size_m src_idx = 0;
                        size_m dst_idx = 0;
                        size_m rem = gid;
                        for (int i = cdims - 1; i >= 0; i--) {
                            size_m idx = rem % res.shape[i];
                            src_idx += idx * res.stridesB[i];
                            dst_idx += idx * res.stridesA[i];
                            rem /= res.shape[i];
                        }
                        out_data[dst_idx] = std::cos(in_data[src_idx]);
                    }
                }
        });
    }
}

matrix matrix::tan(const matrix& input) {
    dtype out_type = promote_types(input.type, dtype::Float16);
    matrix promoted = (input.type != out_type) ? input.astype(out_type, true) : input;
    matrix output(promoted.dims, promoted.type);
    memcpy(output.shape(), promoted.shape(), promoted.dims * sizeof(size_m));
    output.total_size = promoted.total_size;
    output.calcStrides();
    
    auto res = collapse_dims(promoted.shape(), output.strides(), promoted.strides(), promoted.dims, INT32_MAX);
    TanPrimitive* prim = new TanPrimitive(promoted);
    prim->collapsed_dims = res;
    output.tape = prim;
    return output;
}

void matrix::tan(matrix& output, ExecutionDevice exec_device) {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    TanPrimitive* primit = static_cast<TanPrimitive*>(output.tape);
    auto res = primit->collapsed_dims;
    uint32_t cdims = res.out_dims;

    if (exec_device == ExecutionDevice::METAL) {
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        int typeCode = (int)type;
        int kernel_code = cdims > 3 ? 3 : (cdims == 0 ? 0 : cdims - 1);

        if (!GlobalGPUManager.TanInit_nd[typeCode][kernel_code]) {
            GlobalGPUManager.initTan_nd(typeCode, kernel_code);
        }

        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0]; // Output destination
        setBufferOrBytes(commandEncoder, *this, 1); // Input is the normal contiguous/strided source
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);

        if (cdims == 0) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.TanComputeState_nd[typeCode][0]];
            size_m one = 1;
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        } else if (cdims == 1) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.TanComputeState_nd[typeCode][0]];
            [commandEncoder setBytes:res.stridesA length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[0], 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.TanComputeState_nd[typeCode][1]];
            [commandEncoder setBytes:res.stridesA length:2 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:2 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[1], res.shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.TanComputeState_nd[typeCode][2]];
            [commandEncoder setBytes:res.stridesA length:3 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:3 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[2], res.shape[1], res.shape[0]);
        } else {
            [commandEncoder setComputePipelineState:GlobalGPUManager.TanComputeState_nd[typeCode][3]];
            [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:cdims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:res.shape length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:5];
            
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= res.shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(res.shape[cdims-1], res.shape[cdims-2], acc);
        }

        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        // CPU fallback mapping
        dispatch_type(type, output.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
                T* in_data = (T*)buffer;
                if (cdims == 0) {
                    out_data[0] = std::tan(in_data[0]);
                } else if (cdims == 1) {
                    size_m out_stride = res.stridesA[0];
                    size_m in_stride = res.stridesB[0];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        out_data[i * out_stride] = std::tan(in_data[i * in_stride]);
                    }
                } else if (cdims == 2) {
                    size_m out_stride0 = res.stridesA[0];
                    size_m out_stride1 = res.stridesA[1];
                    size_m in_stride0 = res.stridesB[0];
                    size_m in_stride1 = res.stridesB[1];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        for (size_m j = 0; j < res.shape[1]; j++) {
                            out_data[i * out_stride0 + j * out_stride1] = std::tan(in_data[i * in_stride0 + j * in_stride1]);
                        }
                    }
                } else if (cdims == 3) {
                    size_m out_stride0 = res.stridesA[0];
                    size_m out_stride1 = res.stridesA[1];
                    size_m out_stride2 = res.stridesA[2];
                    size_m in_stride0 = res.stridesB[0];
                    size_m in_stride1 = res.stridesB[1];
                    size_m in_stride2 = res.stridesB[2];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        for (size_m j = 0; j < res.shape[1]; j++) {
                            for (size_m k = 0; k < res.shape[2]; k++) {
                                out_data[i * out_stride0 + j * out_stride1 + k * out_stride2] = std::tan(in_data[i * in_stride0 + j * in_stride1 + k * in_stride2]);
                            }
                        }
                    }
                } else {
                    for (size_m gid = 0; gid < total_size; gid++) {
                        size_m src_idx = 0;
                        size_m dst_idx = 0;
                        size_m rem = gid;
                        for (int i = cdims - 1; i >= 0; i--) {
                            size_m idx = rem % res.shape[i];
                            src_idx += idx * res.stridesB[i];
                            dst_idx += idx * res.stridesA[i];
                            rem /= res.shape[i];
                        }
                        out_data[dst_idx] = std::tan(in_data[src_idx]);
                    }
                }
        });
    }
}

matrix matrix::sqrt(const matrix& input) {
    dtype out_type = promote_types(input.type, dtype::Float16);
    matrix promoted = (input.type != out_type) ? input.astype(out_type, true) : input;
    matrix output(promoted.dims, promoted.type);
    memcpy(output.shape(), promoted.shape(), promoted.dims * sizeof(size_m));
    output.total_size = promoted.total_size;
    output.calcStrides();
    
    auto res = collapse_dims(promoted.shape(), output.strides(), promoted.strides(), promoted.dims, INT32_MAX);
    SqrtPrimitive* prim = new SqrtPrimitive(promoted);
    prim->collapsed_dims = res;
    output.tape = prim;
    return output;
}

void matrix::sqrt(matrix& output, ExecutionDevice exec_device) {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    SqrtPrimitive* primit = static_cast<SqrtPrimitive*>(output.tape);
    auto res = primit->collapsed_dims;
    uint32_t cdims = res.out_dims;

    if (exec_device == ExecutionDevice::METAL) {
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        int typeCode = (int)type;
        int kernel_code = cdims > 3 ? 3 : (cdims == 0 ? 0 : cdims - 1);

        if (!GlobalGPUManager.SqrtInit_nd[typeCode][kernel_code]) {
            GlobalGPUManager.initSqrt_nd(typeCode, kernel_code);
        }

        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0]; // Output destination
        setBufferOrBytes(commandEncoder, *this, 1); // Input is the normal contiguous/strided source
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);

        if (cdims == 0) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.SqrtComputeState_nd[typeCode][0]];
            size_m one = 1;
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        } else if (cdims == 1) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.SqrtComputeState_nd[typeCode][0]];
            [commandEncoder setBytes:res.stridesA length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[0], 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.SqrtComputeState_nd[typeCode][1]];
            [commandEncoder setBytes:res.stridesA length:2 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:2 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[1], res.shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.SqrtComputeState_nd[typeCode][2]];
            [commandEncoder setBytes:res.stridesA length:3 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:3 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[2], res.shape[1], res.shape[0]);
        } else {
            [commandEncoder setComputePipelineState:GlobalGPUManager.SqrtComputeState_nd[typeCode][3]];
            [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:cdims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:res.shape length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:5];
            
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= res.shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(res.shape[cdims-1], res.shape[cdims-2], acc);
        }

        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        // CPU fallback mapping
        dispatch_type(type, output.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
                T* in_data = (T*)buffer;
                if (cdims == 0) {
                    out_data[0] = std::sqrt(in_data[0]);
                } else if (cdims == 1) {
                    size_m out_stride = res.stridesA[0];
                    size_m in_stride = res.stridesB[0];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        out_data[i * out_stride] = std::sqrt(in_data[i * in_stride]);
                    }
                } else if (cdims == 2) {
                    size_m out_stride0 = res.stridesA[0];
                    size_m out_stride1 = res.stridesA[1];
                    size_m in_stride0 = res.stridesB[0];
                    size_m in_stride1 = res.stridesB[1];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        for (size_m j = 0; j < res.shape[1]; j++) {
                            out_data[i * out_stride0 + j * out_stride1] = std::sqrt(in_data[i * in_stride0 + j * in_stride1]);
                        }
                    }
                } else if (cdims == 3) {
                    size_m out_stride0 = res.stridesA[0];
                    size_m out_stride1 = res.stridesA[1];
                    size_m out_stride2 = res.stridesA[2];
                    size_m in_stride0 = res.stridesB[0];
                    size_m in_stride1 = res.stridesB[1];
                    size_m in_stride2 = res.stridesB[2];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        for (size_m j = 0; j < res.shape[1]; j++) {
                            for (size_m k = 0; k < res.shape[2]; k++) {
                                out_data[i * out_stride0 + j * out_stride1 + k * out_stride2] = std::sqrt(in_data[i * in_stride0 + j * in_stride1 + k * in_stride2]);
                            }
                        }
                    }
                } else {
                    for (size_m gid = 0; gid < total_size; gid++) {
                        size_m src_idx = 0;
                        size_m dst_idx = 0;
                        size_m rem = gid;
                        for (int i = cdims - 1; i >= 0; i--) {
                            size_m idx = rem % res.shape[i];
                            src_idx += idx * res.stridesB[i];
                            dst_idx += idx * res.stridesA[i];
                            rem /= res.shape[i];
                        }
                        out_data[dst_idx] = std::sqrt(in_data[src_idx]);
                    }
                }
        });
    }
}

matrix matrix::exp(const matrix& input) {
    dtype out_type = promote_types(input.type, dtype::Float16);
    matrix promoted = (input.type != out_type) ? input.astype(out_type, true) : input;
    matrix output(promoted.dims, promoted.type);
    memcpy(output.shape(), promoted.shape(), promoted.dims * sizeof(size_m));
    output.total_size = promoted.total_size;
    output.calcStrides();
    
    auto res = collapse_dims(promoted.shape(), output.strides(), promoted.strides(), promoted.dims, INT32_MAX);
    ExpPrimitive* prim = new ExpPrimitive(promoted);
    prim->collapsed_dims = res;
    output.tape = prim;
    return output;
}

void matrix::exp(matrix& output, ExecutionDevice exec_device) {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    ExpPrimitive* primit = static_cast<ExpPrimitive*>(output.tape);
    auto res = primit->collapsed_dims;
    uint32_t cdims = res.out_dims;

    if (exec_device == ExecutionDevice::METAL) {
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        int typeCode = (int)type;
        int kernel_code = cdims > 3 ? 3 : (cdims == 0 ? 0 : cdims - 1);

        if (!GlobalGPUManager.ExpInit_nd[typeCode][kernel_code]) {
            GlobalGPUManager.initExp_nd(typeCode, kernel_code);
        }

        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0]; // Output destination
        setBufferOrBytes(commandEncoder, *this, 1); // Input is the normal contiguous/strided source
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);

        if (cdims == 0) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.ExpComputeState_nd[typeCode][0]];
            size_m one = 1;
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        } else if (cdims == 1) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.ExpComputeState_nd[typeCode][0]];
            [commandEncoder setBytes:res.stridesA length:sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[0], 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.ExpComputeState_nd[typeCode][1]];
            [commandEncoder setBytes:res.stridesA length:2 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:2 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[1], res.shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setComputePipelineState:GlobalGPUManager.ExpComputeState_nd[typeCode][2]];
            [commandEncoder setBytes:res.stridesA length:3 * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:3 * sizeof(size_m) atIndex:3];
            _dispatchExecutionSize = MTLSizeMake(res.shape[2], res.shape[1], res.shape[0]);
        } else {
            [commandEncoder setComputePipelineState:GlobalGPUManager.ExpComputeState_nd[typeCode][3]];
            [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:cdims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:res.shape length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:5];
            
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= res.shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(res.shape[cdims-1], res.shape[cdims-2], acc);
        }

        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        // CPU fallback mapping
        dispatch_type(type, output.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
                T* in_data = (T*)buffer;
                if (cdims == 0) {
                    out_data[0] = std::exp(in_data[0]);
                } else if (cdims == 1) {
                    size_m out_stride = res.stridesA[0];
                    size_m in_stride = res.stridesB[0];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        out_data[i * out_stride] = std::exp(in_data[i * in_stride]);
                    }
                } else if (cdims == 2) {
                    size_m out_stride0 = res.stridesA[0];
                    size_m out_stride1 = res.stridesA[1];
                    size_m in_stride0 = res.stridesB[0];
                    size_m in_stride1 = res.stridesB[1];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        for (size_m j = 0; j < res.shape[1]; j++) {
                            out_data[i * out_stride0 + j * out_stride1] = std::exp(in_data[i * in_stride0 + j * in_stride1]);
                        }
                    }
                } else if (cdims == 3) {
                    size_m out_stride0 = res.stridesA[0];
                    size_m out_stride1 = res.stridesA[1];
                    size_m out_stride2 = res.stridesA[2];
                    size_m in_stride0 = res.stridesB[0];
                    size_m in_stride1 = res.stridesB[1];
                    size_m in_stride2 = res.stridesB[2];
                    for (size_m i = 0; i < res.shape[0]; i++) {
                        for (size_m j = 0; j < res.shape[1]; j++) {
                            for (size_m k = 0; k < res.shape[2]; k++) {
                                out_data[i * out_stride0 + j * out_stride1 + k * out_stride2] = std::exp(in_data[i * in_stride0 + j * in_stride1 + k * in_stride2]);
                            }
                        }
                    }
                } else {
                    for (size_m gid = 0; gid < total_size; gid++) {
                        size_m src_idx = 0;
                        size_m dst_idx = 0;
                        size_m rem = gid;
                        for (int i = cdims - 1; i >= 0; i--) {
                            size_m idx = rem % res.shape[i];
                            src_idx += idx * res.stridesB[i];
                            dst_idx += idx * res.stridesA[i];
                            rem /= res.shape[i];
                        }
                        out_data[dst_idx] = std::exp(in_data[src_idx]);
                    }
                }
        });
    }
}


matrix matrix::dot(const matrix& b, bool transposeB) {
    // 1. Graph Level Cache-Locality Optimization
    // The backend wants B transposed so it can read it linearly.
    // If the user didn't transpose B, we add a .transpose() node to the DAG!
    matrix b_optimized(b.dims, b.type);
    if (transposeB) {
        b_optimized = b;
    } else {
        std::vector<size_m> axes(b.dims);
        int stop_idx = b.dims >= 2 ? b.dims - 2 : b.dims;
        for (int i = 0; i < stop_idx; ++i) axes[i] = i;
        if (b.dims >= 2) {
            axes[b.dims - 2] = b.dims - 1;
            axes[b.dims - 1] = b.dims - 2;
        }
        b_optimized = b.transpose(axes);
    }
    b_optimized = (b_optimized.flags & NON_CONTIGUOUS_FLAG) ? b_optimized.astype(b_optimized.type, true) : b_optimized;
    matrix result(std::max(dims, b_optimized.dims), type);
    auto primit = new DotPrimitive(*this, b_optimized);

    primit->desc_a = BroadcastDescriptor::create(result.dims);
    primit->desc_b = BroadcastDescriptor::create(result.dims);
    broadcast_shapes_matmul(array_desc, b_optimized.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, b_optimized.dims);
    primit->collapsed_dims_3 = collapse_dims_matmul(
            result.shape(),
            primit->desc_a->strides(result.dims),
            primit->desc_b->strides(result.dims),
            result.strides(), result.dims, INT32_MAX);
    // We pass b_optimized to the Primitive.
    result.total_size = result.accumul(0, result.dims);
    result.tape = primit;
    return result;
}

void matrix::dot_cpu(matrix& b_transposed, matrix& result) {
    if (this->dims > 2 || b_transposed.dims > 2) {
        this->batched_dot_cpu(b_transposed, result);
        return;
    }
    // Both 'this' (A) and 'b_transposed' (B) have their inner dimension K contiguous in memory!
    
    int M = this->shape()[0];
    int K = this->shape()[1];
    int N = b_transposed.shape()[0];
    
    dispatch_type(type, result.buffer, [&](auto *C_ptr) {
        using T = std::decay_t<decltype(*C_ptr)>;
        
        T* A_ptr = (T*)this->buffer;
        T* B_ptr = (T*)b_transposed.buffer;

        
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                T sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    // High cache hit rate!
                    sum += A_ptr[m * K + k] * B_ptr[n * K + k];
                }
                C_ptr[m * N + n] = sum;
            }
        }
    });
}

void matrix::batched_dot_cpu(matrix& b_transposed, matrix& result) {
    DotPrimitive* primit = (DotPrimitive*)result.tape;
    CollapsedDims_3& collapsed_desc = primit->collapsed_dims_3;
    
    int out_dims = collapsed_desc.out_dims;
    size_m* shapeR = collapsed_desc.shape;
    size_m* strideA = collapsed_desc.stridesA;
    size_m* strideB = collapsed_desc.stridesB;
    size_m* strideR = collapsed_desc.stridesC;
    
    size_m K = this->shape()[this->dims - 1];
    size_m M = shapeR[out_dims - 2];
    size_m N = shapeR[out_dims - 1];
    
    size_m matmul_size = M * N;
    size_m total_batches = result.total_size / matmul_size;
    
    dispatch_type(type, result.buffer, [&](auto *C_ptr) {
        using T = std::decay_t<decltype(*C_ptr)>;
        T* A_ptr = (T*)this->buffer;
        T* B_ptr = (T*)b_transposed.buffer;
        
        for (size_m batch_idx = 0; batch_idx < total_batches; batch_idx++) {
            size_m GindexA = 0;
            size_m GindexB = 0;
            size_m GindexR = 0;
            
            if (out_dims == 3) {
                GindexA = batch_idx * strideA[0];
                GindexB = batch_idx * strideB[0];
                GindexR = batch_idx * strideR[0];
            } else {
                size_m rem = batch_idx;
                for (int i = out_dims - 3; i >= 0; i--) {
                    size_m dim_index = rem % shapeR[i];
                    GindexA += dim_index * strideA[i];
                    GindexB += dim_index * strideB[i];
                    GindexR += dim_index * strideR[i];
                    rem /= shapeR[i];
                }
            }
            
            for (size_m m = 0; m < M; m++) {
                for (size_m n = 0; n < N; n++) {
                    size_m idxA = GindexA + m * strideA[out_dims - 2];
                    size_m idxB = GindexB + n * strideB[out_dims - 2];
                    size_m idxR = GindexR + m * strideR[out_dims - 2] + n * strideR[out_dims - 1];
                    
                    T sum = 0.0f;
                    for (size_m k = 0; k < K; k++) {
                        sum += A_ptr[idxA + k] * B_ptr[idxB + k];
                    }
                    C_ptr[idxR] = sum;
                }
            }
        }
    });
}

void matrix::dot_gpu(matrix& b_transposed, matrix& result) {
    if (this->dims > 2 || b_transposed.dims > 2) {
        this->batched_dot_gpu(b_transposed, result);
        return;
    }
    
    uint8_t typeCode = static_cast<uint8_t>(type);
    
    if (!GlobalGPUManager.GEMMAInit[typeCode]) {
        GlobalGPUManager.initGEMMA_All(typeCode);
    }
    
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
    
    auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1); // Tune this for your tiles later
    auto _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
    
    // Bind Buffers
    [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
    [commandEncoder setBuffer:this->metalBuffer offset:0 atIndex:1];
    [commandEncoder setBuffer:b_transposed.metalBuffer offset:0 atIndex:2];
    
    // Bind Shapes
    [commandEncoder setBytes:this->shape() length:this->dims * sizeof(size_m) atIndex:3];
    
    // CRITICAL: The shader expects the *original* shape of B [K, N] in buffer 4,
    // but b_transposed.shape is physically [N, K]. We construct [K, N] to satisfy the shader.
    size_m original_shapeB[2] = { b_transposed.shape()[1], b_transposed.shape()[0] };
    [commandEncoder setBytes:original_shapeB length:2 * sizeof(size_m) atIndex:4];
    
    // Use your string lookup or switch statement for the pipeline state here
    [commandEncoder setComputePipelineState:GlobalGPUManager.GEMMAComputeState[typeCode]];
    
    [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
}

void matrix::batched_dot_gpu(matrix& b_transposed, matrix& result) {
    uint8_t typeCode = static_cast<uint8_t>(type);
    
    // We expect result.tape to hold the DotPrimitive which already contains collapsed_dims_3
    DotPrimitive* primit = (DotPrimitive*)result.tape;
    CollapsedDims_3& collapsed_desc = primit->collapsed_dims_3;
    
    // Choose Pipeline: 3D if fully collapsed (batch=1 dim, + M, N), otherwise ND
    int spec_idx = (collapsed_desc.out_dims == 3) ? 0 : 1;
    if (!GlobalGPUManager.BatchedMatMulInit[typeCode][spec_idx]) {
        GlobalGPUManager.initBatchedMatMul(typeCode, spec_idx);
    }
    
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
    [commandEncoder setComputePipelineState:GlobalGPUManager.BatchedMatMulComputeState[typeCode][spec_idx]];
    
    [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
    [commandEncoder setBuffer:this->metalBuffer offset:0 atIndex:1];
    [commandEncoder setBuffer:b_transposed.metalBuffer offset:0 atIndex:2];
    
    size_t array_len = sizeof(size_m) * collapsed_desc.out_dims;
    [commandEncoder setBytes:collapsed_desc.stridesA length:array_len atIndex:3];
    [commandEncoder setBytes:collapsed_desc.stridesB length:array_len atIndex:4];
    [commandEncoder setBytes:collapsed_desc.stridesC length:array_len atIndex:5];
    
    // K is the innermost dimension of the dot product itself
    size_m k_val = this->shape()[this->dims - 1];
    
    if (spec_idx == 0) { // 3Dgg
        [commandEncoder setBytes:&k_val length:sizeof(size_m) atIndex:6];
    } else { // NDgg
        [commandEncoder setBytes:collapsed_desc.shape length:array_len atIndex:6];
        [commandEncoder setBytes:&k_val length:sizeof(size_m) atIndex:7];
        
        int ndims_val = collapsed_desc.out_dims;
        [commandEncoder setBytes:&ndims_val length:sizeof(int) atIndex:8];
    }
    
    // Calculate Grid: z -> batch, y -> M (row), x -> N (col)
    size_m N_val = collapsed_desc.shape[collapsed_desc.out_dims - 1];
    size_m M_val = collapsed_desc.shape[collapsed_desc.out_dims - 2];
    
    size_m batch_size = 1;
    for (int i = 0; i < collapsed_desc.out_dims - 2; ++i) {
        batch_size *= collapsed_desc.shape[i];
    }
    
    auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1); 
    auto _dispatchExecutionSize = MTLSizeMake(N_val, M_val, batch_size);
    
    [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
}



// for adding same on cpu
void matrix::add( matrix &other, matrix &result, EvalType evalType) {
    //    result.tape = new AdditionPrimitive(*this, other);
    if (evalType == EvalType::EVAL_AUTO) {
        // lazy evaluation so auto is the default one so
        // when u type a+b it gets called with auto so it
        // just calculates the shape strides and stores
        // them in the primitive
        auto primit = new AdditionPrimitive(*this, other);
        primit->desc_a = BroadcastDescriptor::create(result.dims);
        primit->desc_b = BroadcastDescriptor::create(result.dims);
        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, other.dims);
        result.total_size = result.accumul(0, result.dims);
        result.tape = primit;
        return;
    }

    // FOR PATH BUILD_TRACE AND EVAL_CPU
    // for situations where c = a+b; d= c+c
    // c is an unmaterialised temp so it has no buffer thus both c's where treated differently or for that matter reusing temp nodes recalcuated and allocated everything cause we didnt know they were the same thing but same nodes shared same primitive so we stored some of the properties in the primitive so we can identify if somwhere else memory for c has been allocated then we use the same
//    if (result.tape->out_buffer && !result.buffer) {
//        result.buffer = result.tape->out_buffer;
//        result.metalBuffer = result.tape->out_metal_buffer;
//        result.refCount = result.tape->out_refcount;
//        result.refCount->fetch_add(1);
//        if (evalType == EvalType::COMPILE_TRACE) { return; }
//    } else {
//        result.buffer = new uint8_t[result.effectiveBufferSize() * dtype_size(type)];
//        result.begin_refcount();
//        result.buildMetalBuffer();
//
//        result.tape->out_buffer = (uint8_t*)result.buffer;
//        result.tape->out_metal_buffer = result.metalBuffer;
//        result.tape->out_refcount = result.refCount;
//    }

    // addition logic

    update_from_trace();
    other.update_from_trace();

    // EXECUTION PATH : FOR EVAL_CPU AND EXEC_TRACE_CPU
    if (result.dims != fmax(dims, other.dims)) {
        throw std::invalid_argument("Incompatible dims of the result mat");
    }

    size_m *strideA = ((AdditionPrimitive *)result.tape)->desc_a->strides(result.dims);
    size_m *strideB = ((AdditionPrimitive *)result.tape)->desc_b->strides(result.dims);

    dispatch_type(type, result.buffer, [&](auto *out_data) {
        using T = std::decay_t<decltype(*out_data)>;
        for (int gid = 0; gid < result.total_size; gid++) {
            // globalIndex for A and B
            size_t GindexA = 0;
            size_t GindexB = 0;

            // axis Index for Result like temp storage for i, j, k, l ... along each
            // axis for result
            size_t indexR = 0;

            int rem = gid;
            for (int i = 0; i < result.dims; i++) {
                indexR = rem / result.strides()[i];
                GindexA += indexR * strideA[i];
                GindexB += indexR * strideB[i];
                rem %= result.strides()[i];
            }

            out_data[gid] = static_cast<T *>(buffer)[GindexA] +
            static_cast<T *>(other.buffer)[GindexB];
        }
    });
}

void matrix::multiply(const matrix &other, matrix &result,
                      EvalType evalType) const {
    //    result.tape = new MultiplicationPrimitive(*this, other);
    if (evalType == EvalType::EVAL_AUTO) {
        auto primit = new AdditionPrimitive(*this, other);
        primit->desc_a = BroadcastDescriptor::create(result.dims);
        primit->desc_b = BroadcastDescriptor::create(result.dims);
        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, other.dims);
        result.tape = primit;
        result.total_size = result.accumul(0, result.dims);
    }
}

void matrix::subtract(const matrix &other, matrix &result,
                      EvalType evalType) const {
    //    result.tape = new SubtractionPrimitive(*this, other);
    if (evalType == EvalType::EVAL_AUTO) {
        auto primit = new AdditionPrimitive(*this, other);
        primit->desc_a = BroadcastDescriptor::create(result.dims);
        primit->desc_b = BroadcastDescriptor::create(result.dims);
        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, other.dims);
        result.tape = primit;
        result.total_size = result.accumul(0, result.dims);
    }
}

void matrix::divide(const matrix &other, matrix &result,
                    EvalType evalType) const {
    //    result.tape = new DivisionPrimitive(*this, other);
    if (evalType == EvalType::EVAL_AUTO) {
        auto primit = new AdditionPrimitive(*this, other);
        primit->desc_a = BroadcastDescriptor::create(result.dims);
        primit->desc_b = BroadcastDescriptor::create(result.dims);
        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, other.dims);
        result.tape = primit;
        result.total_size = result.accumul(0, result.dims);
    }
}

void matrix::add_cpu_brodcasted(matrix& other, matrix &result, EvalType evalType)  {
    
    if (evalType == EvalType::EVAL_AUTO) {
        // lazy evaluation so auto is the default one so
        // when u type a+b it gets called with auto so it
        // just calculates the shape strides and stores
        // them in the primitive
        auto primit = new AdditionPrimitive(*this, other);
        primit->desc_a = BroadcastDescriptor::create(result.dims);
        primit->desc_b = BroadcastDescriptor::create(result.dims);
        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, other.dims);
        primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_b->strides(result.dims), result.strides(), result.dims, INT32_MAX);
        primit->dims_collapsed = true;
        result.total_size = result.accumul(0, result.dims);
        result.tape = primit;
        return;
    }
    
    // FOR PATH BUILD_TRACE AND EVAL_CPU
    // for situations where c = a+b; d= c+c
    // c is an unmaterialised temp so it has no buffer thus both c's where treated differently or for that matter reusing temp nodes recalcuated and allocated everything cause we didnt know they were the same thing but same nodes shared same primitive so we stored some of the properties in the primitive so we can identify if somwhere else memory for c has been allocated then we use the same
//    if (result.tape->out_buffer && !result.buffer) {
//        result.buffer = result.tape->out_buffer;
//        result.metalBuffer = result.tape->out_metal_buffer;
//        result.refCount = result.tape->out_refcount;
//        result.refCount->fetch_add(1);
//        if (evalType == EvalType::BUILD_TRACE) { return; }
//    } else {
//        result.buffer = new uint8_t[result.effectiveBufferSize() * dtype_size(type)];
//        result.begin_refcount();
//        result.buildMetalBuffer();
//        
//        result.tape->out_buffer = (uint8_t*)result.buffer;
//        result.tape->out_metal_buffer = result.metalBuffer;
//        result.tape->out_refcount = result.refCount;
//    }
    
    // addition logic
    
    update_from_trace();
    other.update_from_trace();
    
    
    if (result.dims == 0) {
        // scalar - scalar, no broadcasting needed
        dispatch_type(type, result.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
            out_data[0] = static_cast<T*>(buffer)[0] + static_cast<T*>(other.buffer)[0];
        });
        return;
    }
    
    // EXECUTION PATH : FOR EVAL_CPU AND EXEC_TRACE_CPU
    if (result.dims != fmax(dims, other.dims)) {
        throw std::invalid_argument("Incompatible dims of the result mat");
    }
    int cdims = ((AdditionPrimitive *)result.tape)->collapsed_dims_3.out_dims;
    size_m* strideA = ((AdditionPrimitive *)result.tape)->collapsed_dims_3.stridesA;
    size_m* strideB = ((AdditionPrimitive *)result.tape)->collapsed_dims_3.stridesB;
    size_m* strideR = ((AdditionPrimitive *)result.tape)->collapsed_dims_3.stridesC;
    
    dispatch_type(type, result.buffer, [&](auto *out_data) {
        using T = std::decay_t<decltype(*out_data)>;
        for (int gid = 0; gid < result.total_size; gid++) {
            // globalIndex for A and B
            size_t GindexA = 0;
            size_t GindexB = 0;
            
            // axis Index for Result like temp storage for i, j, k, l ... along each
            // axis for result
            size_t indexR = 0;
            
            int rem = gid;
            for (int i = 0; i < cdims; i++) {
                indexR = rem / strideR[i];
                GindexA += indexR * strideA[i];
                GindexB += indexR * strideB[i];
                rem %= strideR[i];
            }
            
            out_data[gid] = static_cast<T *>(buffer)[GindexA] + static_cast<T *>(other.buffer)[GindexB];
        }
    });
}


void matrix::multiply_cpu_brodcasted(matrix &other, matrix &result, EvalType evalType) {
    if (evalType == EvalType::EVAL_AUTO) {
        // lazy evaluation so auto is the default one so
        // when u type a+b it gets called with auto so it
        // just calculates the shape strides and stores
        // them in the primitive
        auto primit = new MultiplicationPrimitive(*this, other);
        primit->desc_a = BroadcastDescriptor::create(result.dims);
        primit->desc_b = BroadcastDescriptor::create(result.dims);
        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, other.dims);
        primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_b->strides(result.dims), result.strides(), result.dims, INT32_MAX);
        primit->dims_collapsed = true;
        result.total_size = result.accumul(0, result.dims);
        result.tape = primit;
        return;
    }
    
    // multiplicatiion logic
    
    update_from_trace();
    other.update_from_trace();
    
    // EXECUTION PATH : FOR EVAL_CPU AND EXEC_TRACE_CPU
    if (result.dims != fmax(dims, other.dims)) {
        throw std::invalid_argument("Incompatible dims of the result mat");
    }
    if (result.dims == 0) {
        // scalar - scalar, no broadcasting needed
        dispatch_type(type, result.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
            out_data[0] = static_cast<T*>(buffer)[0]
                        * static_cast<T*>(other.buffer)[0];
        });
        return;
    }
    
    int cdims = ((MultiplicationPrimitive *)result.tape)->collapsed_dims_3.out_dims;
    size_m* strideA = ((MultiplicationPrimitive *)result.tape)->collapsed_dims_3.stridesA;
    size_m* strideB = ((MultiplicationPrimitive *)result.tape)->collapsed_dims_3.stridesB;
    size_m* strideR = ((MultiplicationPrimitive *)result.tape)->collapsed_dims_3.stridesC;
    
    dispatch_type(type, result.buffer, [&](auto *out_data) {
        using T = std::decay_t<decltype(*out_data)>;
        for (int gid = 0; gid < result.total_size; gid++) {
            // globalIndex for A and B
            size_t GindexA = 0;
            size_t GindexB = 0;
            
            // axis Index for Result like temp storage for i, j, k, l ... along each
            // axis for result
            size_t indexR = 0;
            
            int rem = gid;
            for (int i = 0; i < cdims; i++) {
                indexR = rem / strideR[i];
                GindexA += indexR * strideA[i];
                GindexB += indexR * strideB[i];
                rem %= strideR[i];
            }
            
            out_data[gid] = static_cast<T *>(buffer)[GindexA] * static_cast<T *>(other.buffer)[GindexB];
        }
    });
}
void matrix::subtract_cpu_brodcasted( matrix &other, matrix &result, EvalType evalType)  {
    if (evalType == EvalType::EVAL_AUTO) {
        // lazy evaluation so auto is the default one so
        // when u type a+b it gets called with auto so it
        // just calculates the shape strides and stores
        // them in the primitive
        auto primit = new SubtractionPrimitive(*this, other);
        primit->desc_a = BroadcastDescriptor::create(result.dims);
        primit->desc_b = BroadcastDescriptor::create(result.dims);
        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, other.dims);
        primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_b->strides(result.dims), result.strides(),result.dims, INT32_MAX);
        primit->dims_collapsed = true;
        result.total_size = result.accumul(0, result.dims);
        result.tape = primit;
        return;
    }
    
    update_from_trace();
    other.update_from_trace();
    
    // EXECUTION PATH : FOR EVAL_CPU AND EXEC_TRACE_CPU
    if (result.dims != fmax(dims, other.dims)) {
        throw std::invalid_argument("Incompatible dims of the result mat");
    }
    // Before the stride extraction and loop:
    if (result.dims == 0) {
        // scalar - scalar, no broadcasting needed
        dispatch_type(type, result.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
            out_data[0] = static_cast<T*>(buffer)[0]
                        - static_cast<T*>(other.buffer)[0];
        });
        return;
    }
    
    int cdims = ((SubtractionPrimitive *)result.tape)->collapsed_dims_3.out_dims;
    size_m* strideA = ((SubtractionPrimitive *)result.tape)->collapsed_dims_3.stridesA;
    size_m* strideB = ((SubtractionPrimitive *)result.tape)->collapsed_dims_3.stridesB;
    size_m* strideR = ((SubtractionPrimitive *)result.tape)->collapsed_dims_3.stridesC;
    
    dispatch_type(type, result.buffer, [&](auto *out_data) {
        using T = std::decay_t<decltype(*out_data)>;
        for (int gid = 0; gid < result.total_size; gid++) {
            // globalIndex for A and B
            size_t GindexA = 0;
            size_t GindexB = 0;
            
            // axis Index for Result like temp storage for i, j, k, l ... along each
            // axis for result
            size_t indexR = 0;
            
            int rem = gid;
            for (int i = 0; i < cdims; i++) {
                indexR = rem / strideR[i];
                GindexA += indexR * strideA[i];
                GindexB += indexR * strideB[i];
                rem %= strideR[i];
            }
            
            out_data[gid] = static_cast<T *>(buffer)[GindexA] - static_cast<T *>(other.buffer)[GindexB];
        }
    });
}
void matrix::divide_cpu_brodcasted( matrix &other, matrix &result, EvalType evalType)  {
    if (evalType == EvalType::EVAL_AUTO) {
        // lazy evaluation so auto is the default one so
        // when u type a+b it gets called with auto so it
        // just calculates the shape strides and stores
        // them in the primitive
        auto primit = new DivisionPrimitive(*this, other);
        primit->desc_a = BroadcastDescriptor::create(result.dims);
        primit->desc_b = BroadcastDescriptor::create(result.dims);
        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, other.dims);
        primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_b->strides(result.dims), result.strides(), result.dims, INT32_MAX);
        primit->dims_collapsed = true;
        result.total_size = result.accumul(0, result.dims);
        result.tape = primit;
        return;
    }
    
    // division logic
    
    update_from_trace();
    other.update_from_trace();
    
    if (result.dims == 0) {
        // scalar - scalar, no broadcasting needed
        dispatch_type(type, result.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
            out_data[0] = static_cast<T*>(buffer)[0] / static_cast<T*>(other.buffer)[0];
        });
        return;
    }
    
    // EXECUTION PATH : FOR EVAL_CPU AND EXEC_TRACE_CPU
    if (result.dims != fmax(dims, other.dims)) {
        throw std::invalid_argument("Incompatible dims of the result mat");
    }
    
    int cdims = ((DivisionPrimitive *)result.tape)->collapsed_dims_3.out_dims;
    size_m* strideA = ((DivisionPrimitive *)result.tape)->collapsed_dims_3.stridesA;
    size_m* strideB = ((DivisionPrimitive *)result.tape)->collapsed_dims_3.stridesB;
    size_m* strideR = ((DivisionPrimitive *)result.tape)->collapsed_dims_3.stridesC;
    
    dispatch_type(type, result.buffer, [&](auto *out_data) {
        using T = std::decay_t<decltype(*out_data)>;
        for (int gid = 0; gid < result.total_size; gid++) {
            // globalIndex for A and B
            size_t GindexA = 0;
            size_t GindexB = 0;
            
            // axis Index for Result like temp storage for i, j, k, l ... along each
            // axis for result
            size_t indexR = 0;
            
            int rem = gid;
            for (int i = 0; i < cdims; i++) {
                indexR = rem / strideR[i];
                GindexA += indexR * strideA[i];
                GindexB += indexR * strideB[i];
                rem %= strideR[i];
            }
            
            out_data[gid] = static_cast<T *>(buffer)[GindexA] / static_cast<T *>(other.buffer)[GindexB];
        }
    });
}

void matrix::add_gpu_brodcasted(matrix &other, matrix &result, EvalType evalType)  {
    
//    if (evalType == EvalType::EXEC_TRACE) {
//        return;
//    }
    
    if (evalType == EvalType::EVAL_AUTO) {
        // lazy evaluation so auto is the default one so
        // when u type a+b it gets called with auto so it
        // just calculates the shape strides and stores
        // them in the primitive
        if (result.dims != fmax(dims, other.dims)) {
            throw std::invalid_argument("Incompatible dims of the result mat");
        }
        auto primit = new AdditionPrimitive(*this, other);
        primit->desc_a = BroadcastDescriptor::create(result.dims);
        primit->desc_b = BroadcastDescriptor::create(result.dims);
        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, other.dims);
        primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_a->strides(result.dims), result.strides(), result.dims, INT32_MAX);
        primit->dims_collapsed = true;
        result.total_size = result.accumul(0, result.dims);
        result.tape = primit;
        return;
    }
    

    
//    if (result.total_size != result.accumul(0, result.dims)) {
//        // as during execution of compiled graph
//        // we trust everything that memory has
//        // been allocated shapes are precomputed
//        result.total_size = result.accumul(0, result.dims);
//        result.releaseBuffer();
//        result.buffer = new uint8_t[result.effectiveBufferSize() * dtype_size(type)];
//        result.begin_refcount();
//        if (result.total_size > 10) {
//            result.buildMetalBuffer();
//        }
//    }
    // for situations where c = a+b; d= c+c
    // c is an unmaterialised temp so it has no buffer thus both c's where treated differently or for that matter reusing temp nodes recalcuated and allocated everything cause we didnt know they were the same thing but same nodes shared same primitive so we stored some of the properties in the primitive so we can identify if somwhere else memory for c has been allocated then we use the same
//    if (result.tape->out_buffer && !result.buffer) {
//        result.buffer = result.tape->out_buffer;
//        result.metalBuffer = result.tape->out_metal_buffer;
//        result.refCount = result.tape->out_refcount;
//        result.refCount->fetch_add(1);
//        return;
//    } else {
//        result.buffer = new uint8_t[result.effectiveBufferSize() * dtype_size(type)];
//        result.begin_refcount();
//        result.buildMetalBuffer();
//        
//        result.tape->out_buffer = (uint8_t*)result.buffer;
//        result.tape->out_metal_buffer = result.metalBuffer;
//        result.tape->out_refcount = result.refCount;
//    }
    
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
//    if (evalType == EvalType::BUILD_TRACE && !result.tape->out_buffer) {
//        // as during execution of compiled graph
//        // we trust everything that memory has
//        // been allocated shapes are precomputed
//        result.releaseBuffer();
//        result.buffer = new uint8_t[result.effectiveBufferSize() * dtype_size(type)];
//        result.begin_refcount();
//        result.buildMetalBuffer();
//    }
//    result[0, 0] = {0, 0, 0};
    update_from_trace();
    other.update_from_trace();
    
    
    size_m* strideA = ((AdditionPrimitive *)result.tape)->collapsed_dims_3.stridesA;
    size_m* strideB = ((AdditionPrimitive *)result.tape)->collapsed_dims_3.stridesB;
    size_m* strideR = ((AdditionPrimitive *)result.tape)->collapsed_dims_3.stridesC;
    
    size_m* result_shape = ((AdditionPrimitive *)result.tape)->collapsed_dims_3.shape;
    
    int cdims = ((AdditionPrimitive *)result.tape)->collapsed_dims_3.out_dims;
    
    

    auto _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
    
    [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
    setBufferOrBytes(commandEncoder, *this, 1);
    setBufferOrBytes(commandEncoder, other, 2);
    
    int typeCode = (int)type;
    NSUInteger max_threads;
    if (cdims == 0) {
        if (!GlobalGPUManager.BrodcastedAddInit[typeCode][0]) {
            GlobalGPUManager.initBrodcastedAddInit(typeCode, 0);
        }
        size_m one = 1;
        [commandEncoder setBytes:&one length: sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedAddComputeState[typeCode][0]];
        _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
        max_threads = [GlobalGPUManager.BrodcastedAddComputeState[typeCode][0] maxTotalThreadsPerThreadgroup];
    } else if (cdims == 1) {
        // 1D specialisation
        if (!GlobalGPUManager.BrodcastedAddInit[typeCode][0]) {
            GlobalGPUManager.initBrodcastedAddInit(typeCode, 0);
        }
        [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedAddComputeState[typeCode][0]];
        _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
        max_threads = [GlobalGPUManager.BrodcastedAddComputeState[typeCode][0] maxTotalThreadsPerThreadgroup];
        
    } else if (cdims == 2) {
        // 2D specialisation
        if (!GlobalGPUManager.BrodcastedAddInit[typeCode][1]) {
            GlobalGPUManager.initBrodcastedAddInit(typeCode, 1);
        }
        [commandEncoder setBytes:result.strides() length:result.dims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:result.dims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:result.dims * sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedAddComputeState[typeCode][1]];
        _dispatchExecutionSize = MTLSizeMake(result.shape()[1], result.shape()[0], 1);
        max_threads = [GlobalGPUManager.BrodcastedAddComputeState[typeCode][1] maxTotalThreadsPerThreadgroup];
        
    } else if (cdims == 3) {
        // 3D specialisation
        if (!GlobalGPUManager.BrodcastedAddInit[typeCode][2]) {
            GlobalGPUManager.initBrodcastedAddInit(typeCode, 2);
        }
        [commandEncoder setBytes:result.strides() length:result.dims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:result.dims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:result.dims * sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedAddComputeState[typeCode][2]];
        _dispatchExecutionSize = MTLSizeMake(result.shape()[2], result.shape()[1], result.shape()[0]);
        max_threads = [GlobalGPUManager.BrodcastedAddComputeState[typeCode][2] maxTotalThreadsPerThreadgroup];
        
    } else {
        // ND specialisation
        if (!GlobalGPUManager.BrodcastedAddInit[typeCode][3]) {
            GlobalGPUManager.initBrodcastedAddInit(typeCode, 3);
        }
        [commandEncoder setBytes:result.strides() length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:cdims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:cdims * sizeof(size_m) atIndex:5];
        [commandEncoder setBytes:result.shape() length:result.dims * sizeof(size_m) atIndex:6];
        [commandEncoder setBytes:&cdims length:sizeof(int) atIndex:7];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedAddComputeState[typeCode][3]];
        _dispatchExecutionSize = MTLSizeMake(result.shape()[result.dims - 1],
                                             result.shape()[result.dims - 2],
                                             result.accumul(0, result.dims - 2));
        max_threads = [GlobalGPUManager.BrodcastedAddComputeState[typeCode][3] maxTotalThreadsPerThreadgroup];
    }
    
    auto _threadsPerThreadgroup = MTLSizeMake(max_threads, 1, 1);
    [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    
//    [commandEncoder endEncoding];
//    [commandBuffer commit];
//    [commandBuffer waitUntilCompleted];
//    if (evalType == EvalType::BUILD_TRACE) {
//        // compiles the graph in compile i mean finalises
//        // the shape and size and allocates memory so
//        // great for tasks where called repeatedly
//        return;
//    }

    // addition logic
    //    if (evalType == EvalType::EVAL_GPU) { // non-compiled path: just evaluates the matrix
    //    }
}


void matrix::multiply_gpu_brodcasted( matrix &other, matrix &result, EvalType evalType) {
//    if (evalType == EvalType::EXEC_TRACE) {
//        return;
//    }
    
    if (evalType == EvalType::EVAL_AUTO) {
        // lazy evaluation so auto is the default one so
        // when u type a+b it gets called with auto so it
        // just calculates the shape strides and stores
        // them in the primitive
        if (result.dims != fmax(dims, other.dims)) {
            throw std::invalid_argument("Incompatible dims of the result mat");
        }
        auto primit = new MultiplicationPrimitive(*this, other);
        primit->desc_a = BroadcastDescriptor::create(result.dims);
        primit->desc_b = BroadcastDescriptor::create(result.dims);
        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, other.dims);
        primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_b->strides(result.dims), result.strides(), result.dims, INT32_MAX);
        primit->dims_collapsed = true;
        result.total_size = result.accumul(0, result.dims);
        result.tape = primit;
        return;
    }
    
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
    update_from_trace();
    other.update_from_trace();
    
    int result_dims = result.dims;
    size_m* strideA = ((MultiplicationPrimitive *)result.tape)->collapsed_dims_3.stridesA;
    size_m* strideB = ((MultiplicationPrimitive *)result.tape)->collapsed_dims_3.stridesB;
    size_m* strideR = ((MultiplicationPrimitive *)result.tape)->collapsed_dims_3.stridesC;
    
    size_m* result_shape = ((MultiplicationPrimitive *)result.tape)->collapsed_dims_3.shape;
    int cdims = ((MultiplicationPrimitive *)result.tape)->collapsed_dims_3.out_dims;
    
    
    auto _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
    
    [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
    setBufferOrBytes(commandEncoder, *this, 1);
    setBufferOrBytes(commandEncoder, other, 2);
    
    int typeCode = (int)type;
    NSUInteger max_threads;
    
    if (cdims == 0) {
        if (!GlobalGPUManager.BrodcastedMulInit[typeCode][0]) {
            GlobalGPUManager.initBrodcastedMulInit(typeCode, 0);
        }
        size_m one = 1;
        [commandEncoder setBytes:&one length: sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedMulComputeState[typeCode][0]];
        _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
        max_threads = [GlobalGPUManager.BrodcastedMulComputeState[typeCode][0] maxTotalThreadsPerThreadgroup];
    } else if (cdims == 1) {
        // 1D specialisation
        if (!GlobalGPUManager.BrodcastedMulInit[typeCode][0]) {
            GlobalGPUManager.initBrodcastedMulInit(typeCode, 0);
        }
        [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedMulComputeState[typeCode][0]];
        _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
        max_threads = [GlobalGPUManager.BrodcastedMulComputeState[typeCode][0] maxTotalThreadsPerThreadgroup];

        
    } else if (cdims == 2) {
        // 2D specialisation
        if (!GlobalGPUManager.BrodcastedMulInit[typeCode][1]) {
            GlobalGPUManager.initBrodcastedMulInit(typeCode, 1);
        }
        [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:cdims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:cdims * sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedMulComputeState[typeCode][1]];
        _dispatchExecutionSize = MTLSizeMake(result_shape[1], result_shape[0], 1);
        max_threads = [GlobalGPUManager.BrodcastedMulComputeState[typeCode][1] maxTotalThreadsPerThreadgroup];

        
    } else if (cdims == 3) {
        // 3D specialisation
        if (!GlobalGPUManager.BrodcastedMulInit[typeCode][2]) {
            GlobalGPUManager.initBrodcastedMulInit(typeCode, 2);
        }
        [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:cdims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:cdims * sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedMulComputeState[typeCode][2]];
        _dispatchExecutionSize = MTLSizeMake(result_shape[2], result_shape[1], result_shape[0]);
        max_threads = [GlobalGPUManager.BrodcastedMulComputeState[typeCode][2] maxTotalThreadsPerThreadgroup];

        
    } else {
        // ND specialisation
        if (!GlobalGPUManager.BrodcastedMulInit[typeCode][3]) {
            GlobalGPUManager.initBrodcastedMulInit(typeCode, 3);
        }
        size_t bulk = 1;
        for (int i = 0; i < cdims-2; i++) { bulk *= result_shape[i]; }
        [commandEncoder setBytes:result.strides() length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:cdims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:cdims * sizeof(size_m) atIndex:5];
        [commandEncoder setBytes:result_shape length:cdims * sizeof(size_m) atIndex:6];
        [commandEncoder setBytes:&cdims length:sizeof(int) atIndex:7];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedMulComputeState[typeCode][3]];
        _dispatchExecutionSize = MTLSizeMake(result_shape[cdims - 1],
                                             result_shape[cdims - 2],
                                             bulk);
        max_threads = [GlobalGPUManager.BrodcastedMulComputeState[typeCode][3] maxTotalThreadsPerThreadgroup];

    }
    auto _threadsPerThreadgroup = MTLSizeMake(max_threads, 1, 1);
    
    [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    
}
void matrix::subtract_gpu_brodcasted( matrix &other, matrix &result, EvalType evalType) {
//    if (evalType == EvalType::EXEC_TRACE) {
//        return;
//    }
    
    if (evalType == EvalType::EVAL_AUTO) {
        // lazy evaluation so auto is the default one so
        // when u type a+b it gets called with auto so it
        // just calculates the shape strides and stores
        // them in the primitive
        if (result.dims != fmax(dims, other.dims)) {
            throw std::invalid_argument("Incompatible dims of the result mat");
        }
        auto primit = new SubtractionPrimitive(*this, other);
        primit->desc_a = BroadcastDescriptor::create(result.dims);
        primit->desc_b = BroadcastDescriptor::create(result.dims);
        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, other.dims);
        primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_b->strides(result.dims), result.strides(), result.dims, INT32_MAX);
        primit->dims_collapsed = true;
        result.total_size = result.accumul(0, result.dims);
        result.tape = primit;
        return;
    }
    
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
    update_from_trace();
    other.update_from_trace();
    
    
    size_m* strideA = ((SubtractionPrimitive *)result.tape)->collapsed_dims_3.stridesA;
    size_m* strideB = ((SubtractionPrimitive *)result.tape)->collapsed_dims_3.stridesB;
    size_m* strideR = ((SubtractionPrimitive *)result.tape)->collapsed_dims_3.stridesC;
    
    size_m* result_shape = ((SubtractionPrimitive *)result.tape)->collapsed_dims_3.shape;
    int cdims = ((SubtractionPrimitive *)result.tape)->collapsed_dims_3.out_dims;
    auto _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
    
    
    [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
    setBufferOrBytes(commandEncoder, *this, 1);
    setBufferOrBytes(commandEncoder, other, 2);
    
    int typeCode = (int)type;
    NSUInteger max_threads;
    
    if (cdims == 0) {
        if (!GlobalGPUManager.BrodcastedSubInit[typeCode][0]) {
            GlobalGPUManager.initBrodcastedSubInit(typeCode, 0);
        }
        size_m one = 1;
        [commandEncoder setBytes:&one length: sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedSubComputeState[typeCode][0]];
        _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
        max_threads = [GlobalGPUManager.BrodcastedSubComputeState[typeCode][0] maxTotalThreadsPerThreadgroup];
    } else if (cdims == 1) {
        // 1D specialisation
        if (!GlobalGPUManager.BrodcastedSubInit[typeCode][0]) {
            GlobalGPUManager.initBrodcastedSubInit(typeCode, 0);
        }
        [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedSubComputeState[typeCode][0]];
        _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
        max_threads = [GlobalGPUManager.BrodcastedSubComputeState[typeCode][0] maxTotalThreadsPerThreadgroup];
        
    } else if (cdims == 2) {
        // 2D specialisation
        if (!GlobalGPUManager.BrodcastedSubInit[typeCode][1]) {
            GlobalGPUManager.initBrodcastedSubInit(typeCode, 1);
        }
        [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:cdims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:cdims * sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedSubComputeState[typeCode][1]];
        _dispatchExecutionSize = MTLSizeMake(result_shape[1], result_shape[0], 1);
        max_threads = [GlobalGPUManager.BrodcastedSubComputeState[typeCode][1] maxTotalThreadsPerThreadgroup];
        
    } else if (cdims == 3) {
        // 3D specialisation
        if (!GlobalGPUManager.BrodcastedSubInit[typeCode][2]) {
            GlobalGPUManager.initBrodcastedSubInit(typeCode, 2);
        }
        [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:cdims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:cdims * sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedSubComputeState[typeCode][2]];
        _dispatchExecutionSize = MTLSizeMake(result_shape[2], result_shape[1], result_shape[0]);
        max_threads = [GlobalGPUManager.BrodcastedSubComputeState[typeCode][2] maxTotalThreadsPerThreadgroup];
        
    } else {
        // ND specialisation
        if (!GlobalGPUManager.BrodcastedSubInit[typeCode][3]) {
            GlobalGPUManager.initBrodcastedSubInit(typeCode, 3);
        }
        size_t bulk = 1;
        for (int i = 0; i < cdims-2; i++) { bulk *= result_shape[i]; }
        [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:cdims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:cdims * sizeof(size_m) atIndex:5];
        [commandEncoder setBytes:result_shape length:cdims * sizeof(size_m) atIndex:6];
        [commandEncoder setBytes:&cdims length:sizeof(int) atIndex:7];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedSubComputeState[typeCode][3]];
        _dispatchExecutionSize = MTLSizeMake(result_shape[cdims - 1],
                                             result_shape[cdims - 2],
                                             bulk);
        max_threads = [GlobalGPUManager.BrodcastedSubComputeState[typeCode][3] maxTotalThreadsPerThreadgroup];
    }
    auto _threadsPerThreadgroup = MTLSizeMake(max_threads, 1, 1);
    [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
}

void matrix::divide_gpu_brodcasted( matrix &other, matrix &result, EvalType evalType) {
//    if (evalType == EvalType::EXEC_TRACE) {
//        return;
//    }
    
    if (evalType == EvalType::EVAL_AUTO) {
        // lazy evaluation so auto is the default one so
        // when u type a+b it gets called with auto so it
        // just calculates the shape strides and stores
        // them in the primitive
        if (result.dims != fmax(dims, other.dims)) {
            throw std::invalid_argument("Incompatible dims of the result mat");
        }
        auto primit = new DivisionPrimitive(*this, other);
        primit->desc_a = BroadcastDescriptor::create(result.dims);
        primit->desc_b = BroadcastDescriptor::create(result.dims);
        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
                         primit->desc_a, primit->desc_b, dims, other.dims);
        primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_b->strides(result.dims), result.strides(), result.dims, INT32_MAX);
        primit->dims_collapsed = true;
        result.total_size = result.accumul(0, result.dims);
        result.tape = primit;
        return;
    }
    
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
    update_from_trace();
    other.update_from_trace();
    
    
    size_m* strideA = ((DivisionPrimitive *)result.tape)->collapsed_dims_3.stridesA;
    size_m* strideB = ((DivisionPrimitive *)result.tape)->collapsed_dims_3.stridesB;
    size_m* strideR = ((DivisionPrimitive *)result.tape)->collapsed_dims_3.stridesC;
    
    size_m* result_shape = ((DivisionPrimitive *)result.tape)->collapsed_dims_3.shape;
    int cdims = ((DivisionPrimitive *)result.tape)->collapsed_dims_3.out_dims;
    
    auto _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
    
    [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
    setBufferOrBytes(commandEncoder, *this, 1);
    setBufferOrBytes(commandEncoder, other, 2);
    
    int typeCode = (int)type;
    NSUInteger max_threads;
    
    if (cdims == 0) {
        if (!GlobalGPUManager.BrodcastedDivInit[typeCode][0]) {
            GlobalGPUManager.initBrodcastedDivInit(typeCode, 0);
        }
        size_m one = 1;
        [commandEncoder setBytes:&one length: sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedDivComputeState[typeCode][0]];
        _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
        max_threads = [GlobalGPUManager.BrodcastedDivComputeState[typeCode][0] maxTotalThreadsPerThreadgroup];
    } else if (cdims == 1) {
        // 1D specialisation
        if (!GlobalGPUManager.BrodcastedDivInit[typeCode][0]) {
            GlobalGPUManager.initBrodcastedDivInit(typeCode, 0);
        }
        [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedDivComputeState[typeCode][0]];
        _dispatchExecutionSize = MTLSizeMake(result.total_size, 1, 1);
        max_threads = [GlobalGPUManager.BrodcastedDivComputeState[typeCode][0] maxTotalThreadsPerThreadgroup];
        
    } else if (cdims == 2) {
        // 2D specialisation
        if (!GlobalGPUManager.BrodcastedDivInit[typeCode][1]) {
            GlobalGPUManager.initBrodcastedDivInit(typeCode, 1);
        }
        [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:cdims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:cdims * sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedDivComputeState[typeCode][1]];
        _dispatchExecutionSize = MTLSizeMake(result_shape[1], result_shape[0], 1);
        max_threads = [GlobalGPUManager.BrodcastedDivComputeState[typeCode][1] maxTotalThreadsPerThreadgroup];
        
    } else if (cdims == 3) {
        // 3D specialisation
        if (!GlobalGPUManager.BrodcastedDivInit[typeCode][2]) {
            GlobalGPUManager.initBrodcastedDivInit(typeCode, 2);
        }
        [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:cdims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:cdims * sizeof(size_m) atIndex:5];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedDivComputeState[typeCode][2]];
        _dispatchExecutionSize = MTLSizeMake(result_shape[2], result_shape[1], result_shape[0]);
        max_threads = [GlobalGPUManager.BrodcastedDivComputeState[typeCode][2] maxTotalThreadsPerThreadgroup];
        
    } else {
        // ND specialisation
        if (!GlobalGPUManager.BrodcastedDivInit[typeCode][3]) {
            GlobalGPUManager.initBrodcastedDivInit(typeCode, 3);
        }
        size_t bulk = 1;
        for (int i = 0; i < cdims-2; i++) { bulk *= result_shape[i]; }
        [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
        [commandEncoder setBytes:strideA length:cdims * sizeof(size_m) atIndex:4];
        [commandEncoder setBytes:strideB length:cdims * sizeof(size_m) atIndex:5];
        [commandEncoder setBytes:result_shape length:cdims * sizeof(size_m) atIndex:6];
        [commandEncoder setBytes:&cdims length:sizeof(int) atIndex:7];
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedDivComputeState[typeCode][3]];
        _dispatchExecutionSize = MTLSizeMake(result_shape[cdims - 1],
                                             result_shape[cdims - 2],
                                             bulk);
        max_threads = [GlobalGPUManager.BrodcastedDivComputeState[typeCode][3] maxTotalThreadsPerThreadgroup];
    }
    auto _threadsPerThreadgroup = MTLSizeMake(max_threads, 1, 1);
    [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
}

void matrix::add_cpu( matrix &other, matrix &result, EvalType evalType) {
//    if (evalType == EvalType::EVAL_AUTO) {
//        // lazy evaluation so auto is the default one so
//        // when u type a+b it gets called with auto so it
//        // just calculates the shape strides and stores
//        // them in the primitive
//        auto primit = new AdditionPrimitive(*this, other);
//        primit->desc_a = BroadcastDescriptor::create(result.dims);
//        primit->desc_b = BroadcastDescriptor::create(result.dims);
//        broadcast_shapes(array_desc, other.array_desc, result.array_desc,
//                         primit->desc_a, primit->desc_b, dims, other.dims);
//        result.total_size = result.accumul(0, result.dims);
//        result.tape = primit;
//        return;
//    }
    
    // FOR PATH BUILD_TRACE AND EVAL_CPU
    // for situations where c = a+b; d= c+c
    // c is an unmaterialised temp so it has no buffer thus both c's where treated differently or for that matter reusing temp nodes recalcuated and allocated everything cause we didnt know they were the same thing but same nodes shared same primitive so we stored some of the properties in the primitive so we can identify if somwhere else memory for c has been allocated then we use the same
//    if (result.tape->out_buffer && !result.buffer) {
//        result.buffer = result.tape->out_buffer;
//        result.metalBuffer = result.tape->out_metal_buffer;
//        result.refCount = result.tape->out_refcount;
//        result.refCount->fetch_add(1);
//        if (evalType == EvalType::BUILD_TRACE) { return; }
//    } else {
//        result.buffer = new uint8_t[result.effectiveBufferSize() * dtype_size(type)];
//        result.begin_refcount();
//        result.buildMetalBuffer();
//        
//        result.tape->out_buffer = (uint8_t*)result.buffer;
//        result.tape->out_metal_buffer = result.metalBuffer;
//        result.tape->out_refcount = result.refCount;
//    }
    
    // addition logic
    
    update_from_trace();
    other.update_from_trace();
    
    // EXECUTION PATH : FOR EVAL_CPU AND EXEC_TRACE_CPU
    if (result.dims != fmax(dims, other.dims)) {
        throw std::invalid_argument("Incompatible dims of the result mat");
    }
    
    size_m *strideA = ((AdditionPrimitive *)result.tape)->desc_a->strides(result.dims);
    size_m *strideB = ((AdditionPrimitive *)result.tape)->desc_b->strides(result.dims);
    
    dispatch_type(type, result.buffer, [&](auto *out_data) {
        using T = std::decay_t<decltype(*out_data)>;
        for (int gid = 0; gid < result.total_size; gid++) {
            
            out_data[gid] = static_cast<T *>(buffer)[gid] + static_cast<T *>(other.buffer)[gid];
        }
    });
}
void matrix::multiply_cpu( matrix &other, matrix &result,
                          EvalType evalType)  {}
void matrix::subtract_cpu( matrix &other, matrix &result,
                          EvalType evalType)  {}
void matrix::divide_cpu( matrix &other, matrix &result,
                        EvalType evalType)  {}

void matrix::add_gpu( matrix &other, matrix &result,
                     EvalType evalType)  {}
void matrix::multiply_gpu( matrix &other, matrix &result,
                          EvalType evalType)  {}
void matrix::subtract_gpu( matrix &other, matrix &result,
                          EvalType evalType)  {}
void matrix::divide_gpu( matrix &other, matrix &result,
                        EvalType evalType)  {}

matrix operator+(const matrix& a, const matrix& b) {
    dtype common = promote_types(a.type, b.type);
    const matrix& lhs = (a.type == common) ? a : a.astype(common);
    const matrix& rhs = (b.type == common) ? b : b.astype(common);
    matrix result(std::max(lhs.dims, rhs.dims), common);
    auto primit = new AdditionPrimitive(lhs, rhs);
    primit->desc_a = BroadcastDescriptor::create(result.dims);
    primit->desc_b = BroadcastDescriptor::create(result.dims);
    broadcast_shapes(lhs.array_desc, rhs.array_desc, result.array_desc,
                     primit->desc_a, primit->desc_b, lhs.dims, rhs.dims);
    primit->collapsed_dims_3 = collapse_dims(
        primit->desc_a->shape(),
        primit->desc_a->strides(result.dims),
        primit->desc_b->strides(result.dims),
        result.strides(), result.dims, INT32_MAX);
    primit->dims_collapsed = true;
    result.total_size = result.accumul(0, result.dims);
    result.tape = primit;
    return result;
}

matrix operator-(const matrix& a, const matrix& b) {
    dtype common = promote_types(a.type, b.type);
    const matrix& lhs = (a.type == common) ? a : a.astype(common);
    const matrix& rhs = (b.type == common) ? b : b.astype(common);
    matrix result(std::max(lhs.dims, rhs.dims), common);
    auto primit = new SubtractionPrimitive(lhs, rhs);
    primit->desc_a = BroadcastDescriptor::create(result.dims);
    primit->desc_b = BroadcastDescriptor::create(result.dims);
    broadcast_shapes(lhs.array_desc, rhs.array_desc, result.array_desc,
                     primit->desc_a, primit->desc_b, lhs.dims, rhs.dims);
    primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_b->strides(result.dims), result.strides(), result.dims, INT32_MAX);
    primit->dims_collapsed = true;
    result.total_size = result.accumul(0, result.dims);
    result.tape = primit;
    return result;
}

matrix operator*(const matrix& a, const matrix& b) {
    dtype common = promote_types(a.type, b.type);
    const matrix& lhs = (a.type == common) ? a : a.astype(common);
    const matrix& rhs = (b.type == common) ? b : b.astype(common);
    matrix result(std::max(lhs.dims, rhs.dims), common);
    auto primit = new MultiplicationPrimitive(lhs, rhs);
    primit->desc_a = BroadcastDescriptor::create(result.dims);
    primit->desc_b = BroadcastDescriptor::create(result.dims);
    broadcast_shapes(lhs.array_desc, rhs.array_desc, result.array_desc,
                     primit->desc_a, primit->desc_b, lhs.dims, rhs.dims);
    primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_b->strides(result.dims), result.strides(), result.dims, INT32_MAX);
    primit->dims_collapsed = true;
    result.total_size = result.accumul(0, result.dims);
    result.tape = primit;
    return result;
}

matrix operator/(const matrix& a, const matrix& b) {
    dtype common = promote_types(a.type, b.type);
    const matrix& lhs = (a.type == common) ? a : a.astype(common);
    const matrix& rhs = (b.type == common) ? b : b.astype(common);
    matrix result(std::max(lhs.dims, rhs.dims), common);
    auto primit = new DivisionPrimitive(lhs, rhs);
    primit->desc_a = BroadcastDescriptor::create(result.dims);
    primit->desc_b = BroadcastDescriptor::create(result.dims);
    broadcast_shapes(lhs.array_desc, rhs.array_desc, result.array_desc,
                     primit->desc_a, primit->desc_b, lhs.dims, rhs.dims);
    primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_b->strides(result.dims), result.strides(), result.dims, INT32_MAX);
    primit->dims_collapsed = true;
    result.total_size = result.accumul(0, result.dims);
    result.tape = primit;
    return result;
}

matrix matrix::astype(dtype new_type, bool make_contig) const {
    matrix output(dims, new_type);
    memcpy(output.shape(), shape(), dims * sizeof(size_m));
    if (make_contig) {
        output.calcStrides();
    } else {
        memcpy(output.strides(), strides(), dims * sizeof(size_m));
        output.flags = flags;
        output.flags &= ~NON_OWNERSHIP_FLAG;
    }
    output.total_size = total_size;
    output.tape = new AsTypePrimitive(*this, new_type);
    return output;
}

// intrinsic
void matrix::astype(matrix output, dtype type, EvalType eval_type, ExecutionDevice exec_device) const {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = (total_size > 10) ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    if (exec_device == ExecutionDevice::METAL) {
        if (eval_type == EvalType::EXEC_TRACE || eval_type == EvalType::EVAL_INSTANTLY) {
            copyGPUinplaceTypeCasted(output, *this, 0, Execution::Encode);
        }
    } else {
        if (eval_type == EvalType::EXEC_TRACE || eval_type == EvalType::EVAL_INSTANTLY) {
            copyCPUinplaceTypeCasted(output, *this, 0);
        }
    }
}

void matrix::ensure_evaluated() const {
    if (tape && !tape->evaluated) {
        const_cast<matrix*>(this)->eval();
    }
}

void matrix::eval() {
    if (tape) {
        if (total_size > GPU_EXECUTION_THRESHOLD) {
            eval_metal();
        } else {
            eval_cpu();
        }
    }
}

void matrix::eval_cpu() {
    if (tape) tape->eval_cpu(*this, EvalType::EVAL_INSTANTLY);
}
void matrix::eval_metal() {
    if (!tape) return;
    id<MTLCommandBuffer> oldBuffer = GlobalGPUManager._thread_gCommandBuffer ? GlobalGPUManager.getCommandBuffer() : nullptr;
    tape->eval_metal(*this, EvalType::EVAL_INSTANTLY);
    
    if (GlobalGPUManager._thread_gCommandBuffer && oldBuffer == nullptr) {
        GlobalGPUManager.endCommandEncoding();
        GlobalGPUManager.commitCommandBuffer();
    }
}
void matrix::compile_cpu() {
    if (tape) tape->eval_cpu(*this, EvalType::COMPILE_TRACE);
}
void matrix::compile_metal() {
    if (tape) tape->eval_metal(*this, EvalType::COMPILE_TRACE);
}
void matrix::execute_cpu() {
    if (tape) tape->eval_cpu(*this, EvalType::EXEC_TRACE);
}
void matrix::execute_metal() {
    if (!tape) return;
    id<MTLCommandBuffer> oldBuffer = GlobalGPUManager._thread_gCommandBuffer ? GlobalGPUManager.getCommandBuffer() : nullptr;
    tape->eval_metal(*this, EvalType::EXEC_TRACE);
    
    if (GlobalGPUManager._thread_gCommandBuffer && oldBuffer == nullptr) {
        GlobalGPUManager.endCommandEncoding();
        GlobalGPUManager.commitCommandBuffer();
    }
}
void matrix::clear_trace_checks() {
    if (tape) {
        tape->clear_trace_checks();
    }
}

matrix matrix::linespace(const matrix& a, const matrix& b, size_t no_of_points) {
    matrix t = matrix::linespace(0.0f, 1.0f, no_of_points, a.type);
    if (a.dims > 0) {
        uint32_t axes[a.dims];
        for (uint32_t i = 0; i < a.dims; i++) {
            axes[i] = i + 1;
        }
        t = t.unsqueeze(axes, a.dims);
    }
    return (b - a) * t + a;
}

matrix matrix::insert_break(std::function<void(matrix&)> lambda, ExecutionDevice exec_device) const {
    matrix output(dims, type);
    memcpy(output.shape(), shape(), dims * sizeof(size_m));
    output.calcStrides();
    output.total_size = total_size;
    
    ExecutionBoundaryPrimitive* prim = new ExecutionBoundaryPrimitive(*const_cast<matrix*>(this), lambda, exec_device);
    output.tape = prim;
    
    return output;
}

void matrix::releaseBuffer() {
    if (refCount && buffer) {
        if (refCount->fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // MUST CAST void* to uint8_t* before deleting!
            delete[] static_cast<uint8_t *>(buffer);
            delete refCount;
        }
    } else if (flags & NON_OWNERSHIP_FLAG) {
        return;
    } else if (buffer) {
        // MUST CAST void* to uint8_t* before deleting!
        delete[] static_cast<uint8_t *>(buffer);
    }
    refCount = nullptr;
    buffer = nullptr;
    metalBuffer = nullptr;
}

void matrix::releaseTape() {
    if (tape) {
        if (tape->primitive_refCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete tape;
        }
        tape = nullptr;
    }
}

void matrix::destroyInstance() {
    // 1. Clean up Metadata (Descriptor)
    if (dims > SBO_MAX_DIMS && array_desc.shared_arr_desc) {
        array_desc.shared_arr_desc->release();
    }
    
    if (refCount) {
        if (refCount->fetch_sub(1, std::memory_order_acq_rel) == 1) {
            // MUST CAST void* to uint8_t* before deleting!
            delete[] static_cast<uint8_t *>(buffer);
            delete refCount;
        }
    } else if (flags & NON_OWNERSHIP_FLAG) {
        releaseTape();
        return;
    } else if (buffer) {
        // MUST CAST void* to uint8_t* before deleting!
        delete[] static_cast<uint8_t *>(buffer);
    }
    
    // 3. Reset State
    buffer = nullptr;
    refCount = nullptr;
    dims = 0;
    releaseTape();
}

template <typename Type, typename>
matrix &matrix::operator=(Type value) {
    if (flags & NON_CONTIGUOUS_FLAG) {
        matrix value_mat_view(dims, 1, type);
        memcpy(value_mat_view.shape(), shape(), dims * sizeof(size_m));
        memset(value_mat_view.strides(), 0, (dims) * sizeof(size_m));
        dispatch_type(type, value_mat_view.buffer, [&](auto *typed_buffer) {
            // typed_buffer points to the single element in value_mat_view
            using BufT = std::decay_t<decltype(*typed_buffer)>;
            *typed_buffer = static_cast<BufT>(value);
        });
        value_mat_view.total_size = total_size;
        value_mat_view.flags |= NON_CONTIGUOUS_FLAG;
        if (metalBuffer) {
            copyGPUinplace(*this, value_mat_view, 0);
        } else {
            copyCPUinplace(*this, value_mat_view, 0);
        }
    } else {
        dispatch_type(type, buffer, [&](auto *typed_buffer) {
            // Determine the exact C++ type we are working with
            using BufT = std::decay_t<decltype(*typed_buffer)>;
            
            // Cast the user's value to the buffer's exact type
            BufT casted_val = static_cast<BufT>(value);
            
            TypedPatternFill(typed_buffer, casted_val, (size_m)total_size);
        });
    }
    return *this;
}

matrix::matrix(const matrix &other) {
#ifdef CopyLog
    std::cout << "Copied" << "\n";
#endif
    dims = other.dims;
    type = other.type;
    total_size = other.total_size;
    flags = other.flags;
    tape = other.tape;
    if (tape) tape->primitive_refCount.fetch_add(1, std::memory_order_relaxed);
    
    if (other.refCount || (flags & NON_OWNERSHIP_FLAG) || other.tape) {
        // =======================================================
        // PATH A: SHALLOW COPY (View Sharing)
        // We share the data, so we MUST share the exact metadata!
        // =======================================================
        buffer = other.buffer;
        metalBuffer = other.metalBuffer;
        
        if (other.refCount) {
            refCount = other.refCount;
            refCount->fetch_add(1, std::memory_order_relaxed);
        }
        
        if (dims <= SBO_MAX_DIMS) {
            // Stack copy exact shape & strides
            memcpy(shape(), other.shape(), dims * sizeof(size_m));
            memcpy(strides(), other.strides(), dims * sizeof(size_m));
        } else {
            // Share the heap descriptor exact shape & strides
            array_desc.shared_arr_desc = other.array_desc.shared_arr_desc;
            array_desc.shared_arr_desc->refCount.fetch_add(1,
                                                           std::memory_order_relaxed);
        }
        
    } else {
        // =======================================================
        // PATH B: DEEP COPY (Exclusive Ownership)
        // We allocate fresh data, so we MUST pack it contiguously!
        // =======================================================

        
        flags &= ~NON_OWNERSHIP_FLAG;
        flags &= ~NON_CONTIGUOUS_FLAG; // We are officially packing this tightly
        
        if (dims <= SBO_MAX_DIMS) {
            memcpy(shape(), other.shape(), dims * sizeof(size_m));
        } else {
            // Safely create fresh descriptor (NO fetch_sub needed)
            array_desc.shared_arr_desc = SharedArrayDescriptor::create(dims);
            memcpy(shape(), other.shape(), dims * sizeof(size_m)); // MUST copy shape!
        }
        
        // Calculate fresh, perfectly contiguous strides for our new buffer
        calcStrides();
        tape = other.tape;
        if (tape) tape->primitive_refCount.fetch_add(1, std::memory_order_relaxed);
        
        if (!other.buffer) {return;}
        buffer = new uint8_t[total_size * dtype_size(type)];
        buildMetalBuffer();
        refCount = nullptr;
        
        // Blast the data over
        if (total_size > 10) {
            copyGPUinplace(*this, other, 0);
        } else {
            copyCPUinplace(*this, other, 0);
        }
    }
}

matrix::matrix(matrix &&other) noexcept {
#ifdef MoveLog
    std::cout << "Moved" << "\n";
#endif
    buffer = other.buffer;
    metalBuffer = other.metalBuffer;
    refCount = other.refCount;
    dims = other.dims;
    type = other.type;
    flags = other.flags;
    
    total_size = other.total_size;
    tape = other.tape;
    
    if (dims <= SBO_MAX_DIMS) {
        // Fast path: stack copy (must physically copy the bits for SBO)
        memcpy(shape(), other.shape(), dims * sizeof(size_m));
        memcpy(strides(), other.strides(), dims * sizeof(size_m));
    } else {
        // Fast path: STEAL the descriptor. DO NOT increment the ref count!
        array_desc.shared_arr_desc = other.array_desc.shared_arr_desc;
        // Nullify other's pointer so its destructor doesn't release our memory
        other.array_desc.shared_arr_desc = nullptr;
    }
    
    other.buffer = nullptr;
    other.refCount = nil;
    other.tape = nullptr;
    other.metalBuffer = nil;
}

matrix &matrix::operator=(matrix &&other) {
    if (&other == this) {
        return *this;
    }
    if (flags & NON_OWNERSHIP_FLAG) {
#ifdef CopyLog
        std::cout << "DONT OWN THE DATA COPYInG INSTEAD \n";
#endif
        *this = (const matrix &)other;
        return *this;
    }
#ifdef MoveLog
    std::cout << "Move Assignment" << "\n";
#endif
    
    destroyInstance();
    
    buffer = other.buffer;
    metalBuffer = other.metalBuffer;
    flags = other.flags;
    refCount = other.refCount;
    total_size = other.total_size;
    tape = other.tape;
    dims = other.dims;
    type = other.type;
    
    if (dims <= SBO_MAX_DIMS) {
        // Fast path: stack copy (must physically copy the bits for SBO)
        memcpy(shape(), other.shape(), dims * sizeof(size_m));
        memcpy(strides(), other.strides(), dims * sizeof(size_m));
    } else {
        // Fast path: STEAL the descriptor. DO NOT increment the ref count!
        array_desc.shared_arr_desc = other.array_desc.shared_arr_desc;
        // Nullify other's pointer so its destructor doesn't release our memory
        other.array_desc.shared_arr_desc = nullptr;
    }
    
    other.refCount = nil;
    other.buffer = nullptr;
    other.tape = nil;
    other.metalBuffer = nil;
    return *this;
}

// copy assignment
matrix &matrix::operator=(const matrix &other) {

    if (this == &other) {
        return *this;
    }
    const_cast<matrix&>(other).update_from_trace();
    // for coping ito views
    // Copy Assignment if coping data into a view for eg doing Video[1] = frame;
    // Video[1] is a view in which frame is being copied into
    if (flags & NON_OWNERSHIP_FLAG) {
        if (total_size == other.total_size && dims == other.dims) {
            if (total_size > 10) {
//                eval_metal();
//                const_cast<matrix&>(other).eval_metal();
                copyGPUinplace(*this, other, 0);
            } else {
//                eval_cpu();
//                const_cast<matrix&>(other).eval_cpu();
                copyCPUinplace(*this, other, 0);
            }
            return *this;
        } else {
            matrix Brodcasted_other = other.broadcast_toV2(shape(), dims);
            if (total_size > 10) {
//                eval_metal();
//                Brodcasted_other.eval_metal();
                copyGPUinplace(*this, Brodcasted_other, 0);
            } else {
//                eval_cpu();
//                Brodcasted_other.eval_cpu();
                copyCPUinplace(*this, Brodcasted_other, 0);
            }
            return *this;
        }
    }

    //copy shape and strides
    if (dims > SBO_MAX_DIMS) {
        array_desc.shared_arr_desc->release();
    }

    dims = other.dims;
    if (other.dims <= SBO_MAX_DIMS) {
        memcpy(shape(), other.shape(), dims * sizeof(size_m));
        memcpy(strides(), other.strides(), dims * sizeof(size_m));
    } else {
        array_desc.shared_arr_desc = other.array_desc.shared_arr_desc;
        array_desc.shared_arr_desc->refCount.fetch_add(1, std::memory_order_relaxed);
    }
    type = other.type;
    releaseTape();
    tape = other.tape;
    if (tape) tape->primitive_refCount.fetch_add(1, std::memory_order_relaxed);
    
    // part of graph
    // if a mat has tape then if it is realised (meaning graph has been compiled or exectuted then it must be refcounted)
    if (other.tape) {
        if (other.buffer) {
            this->releaseBuffer();
            refCount = other.refCount;
            refCount->fetch_add(1);
        }
        buffer = other.buffer;
        metalBuffer = other.metalBuffer;
        total_size = other.total_size;
        flags = other.flags;
        return *this;
    }

    // for the times when mat isnt part of a graph and is ref counted
    if (other.refCount) {
        this->releaseBuffer();
        buffer = other.buffer;
        metalBuffer = other.metalBuffer;
        refCount = other.refCount;
        refCount->fetch_add(1);
        return *this;
    }

    // DEEP COPY
    this->releaseBuffer();
    total_size = other.total_size;
    buffer = new uint8_t[total_size * dtype_size(other.type)];
    calcStrides();
    flags = other.flags;
    flags &= ~NON_OWNERSHIP_FLAG;
    flags &= ~NON_CONTIGUOUS_FLAG;
    if (total_size > 10) {
        buildMetalBuffer();
        copyGPUinplace(*this, other, 0);
    }
    copyCPUinplace(*this, other, 0);

    return *this;
}

//matrix &matrix::operator=(const matrix &other) {
//#ifdef CopyLog
//    std::cout << "Copy Assignment\n";
//#endif
//    // 1. SELF-ASSIGNMENT CHECK (Critical!)
//    // If someone does `A = A;`, we must do nothing. If we didn't check this,
//    // we would destroy A's data, and then try to copy from the destroyed A!
//    if (this == &other) {
//        return *this;
//    }
//    
//    if (flags & NON_OWNERSHIP_FLAG) {
//        // Copy Assignment if coping data into a view for eg doing Video[1] = frame;
//        // Video[1] is a view in which frame is being copied into
//        if (buffer == other.buffer) {
//#ifdef CopyLog
//            std::cout << "Ignored redundant self-assignment of views.\n";
//#endif
//            return *this;
//        }
//        
//        // --- THE BROADCAST UPGRADE ---
//        if (total_size == other.total_size || dims == other.dims) {
//            // Exact match fast-path
//            if (metalBuffer) {
//                copyGPUinplace(*this, other, 0);
//            } else {
//                // Inline fast-path: Dodge function jump to `copyCPUinplace` entirely
//                // for contiguous pixel buffers!
//                if (total_size == other.total_size && type == other.type &&
//                    !(flags & NON_CONTIGUOUS_FLAG) &&
//                    !(other.flags & NON_CONTIGUOUS_FLAG)) {
//                    size_t bytes = total_size * dtype_size(type);
//                    if (bytes == 4)
//                        *reinterpret_cast<uint32_t *>(buffer) =
//                        *reinterpret_cast<const uint32_t *>(other.buffer);
//                    else if (bytes == 1)
//                        *reinterpret_cast<uint8_t *>(buffer) =
//                        *reinterpret_cast<const uint8_t *>(other.buffer);
//                    else if (bytes == 8)
//                        *reinterpret_cast<uint64_t *>(buffer) =
//                        *reinterpret_cast<const uint64_t *>(other.buffer);
//                    else
//                        memcpy(buffer, other.buffer, bytes);
//                } else {
//                    copyCPUinplace(*this, other, 0);
//                }
//            }
//            
//        } else {
//            // Create a temporary broadcasted view of 'other' to match 'this'
//            matrix broadcasted_other = other.broadcast_to(shape(), dims);
//            
//            if (metalBuffer)
//                copyGPUinplace(*this, broadcasted_other, 0);
//            else
//                copyCPUinplace(*this, broadcasted_other, 0);
//        }
//        tape = other.tape;
//        return *this;
//    }
//    
//    if (other.refCount) {
//        // =======================================================
//        // PATH A: SHALLOW COPY (View Sharing)
//        // =======================================================
//        destroyInstance();
//        
//        buffer = other.buffer;
//        metalBuffer = other.metalBuffer;
//        
//        // 3. COPY LOGIC (Exact mirror of your perfect Copy Constructor)
//        dims = other.dims;
//        type = other.type;
//        total_size = other.total_size;
//        flags = other.flags;
//        tape = other.tape;
//        
//        if (other.refCount) {
//            refCount = other.refCount;
//            refCount->fetch_add(1, std::memory_order_relaxed);
//        }
//        
//        if (dims <= SBO_MAX_DIMS) {
//            memcpy(shape(), other.shape(), dims * sizeof(size_m));
//            memcpy(strides(), other.strides(), dims * sizeof(size_m));
//        } else {
//            array_desc.shared_arr_desc = other.array_desc.shared_arr_desc;
//            array_desc.shared_arr_desc->refCount.fetch_add(1,
//                                                           std::memory_order_relaxed);
//        }
//        
//    } else {
//        // =======================================================
//        // PATH B: DEEP COPY (Exclusive Ownership)
//        // =======================================================
//        if (buffer == other.buffer &&
//            effectiveBufferSize() == other.effectiveBufferSize()) {
//#ifdef CopyLog
//            std::cout << "Ignored redundant self-assignment of views.\n";
//#endif
//            return *this;
//        }
//        
//        size_t new_byte_size = other.total_size * dtype_size(other.type);
//        size_t old_byte_size = total_size * dtype_size(type);
//        bool can_reuse_memory =
//        (buffer != nullptr && refCount == nullptr &&
//         !(flags & NON_OWNERSHIP_FLAG) && old_byte_size == new_byte_size);
//        // see if i am not sharing this data with anyone else and recieving from
//        // another sorce off same size why allocate again just copy
//        if (can_reuse_memory) {
//            // Rebuild the shape descriptor
//            if (dims <= SBO_MAX_DIMS) {
//                memcpy(shape(), other.shape(), dims * sizeof(size_m));
//            } else {
//                // If we already had a heap descriptor, we can just reuse it too!
//                if (!array_desc.shared_arr_desc) {
//                    array_desc.shared_arr_desc = SharedArrayDescriptor::create(dims);
//                }
//                memcpy(shape(), other.shape(), dims * sizeof(size_m));
//            }
//            
//            // DEEP COPY PATH: Standard fallback
//            if (metalBuffer) {
//                copyGPUinplace(*this, other, 0);
//            } else {
//                copyCPUinplace(*this, other, 0);
//            }
//            tape = other.tape;
//            return *this;
//        }
//        destroyInstance();
//        // 3. COPY LOGIC (Exact mirror of your perfect Copy Constructor)
//        dims = other.dims;
//        type = other.type;
//        total_size = other.total_size;
//        flags = other.flags;
//        tape = other.tape;
//        // 2. CLEANUP EXISTING STATE
//        // We safely release our current buffer and shared descriptors so we don't
//        // leak memory.
//        buffer = new uint8_t[new_byte_size];
//        buildMetalBuffer();
//        refCount = nullptr;
//        
//        flags &= ~NON_OWNERSHIP_FLAG;
//        flags &= ~NON_CONTIGUOUS_FLAG;
//        
//        if (dims <= SBO_MAX_DIMS) {
//            memcpy(shape(), other.shape(), dims * sizeof(size_m));
//        } else {
//            array_desc.shared_arr_desc = SharedArrayDescriptor::create(dims);
//            memcpy(shape(), other.shape(), dims * sizeof(size_m));
//        }
//        
//        calcStrides();
//        copyCPUinplace(*this, other, 0);
//    }
//    return *this;
//}

matrix::~matrix() {
    if (dims > SBO_MAX_DIMS && array_desc.shared_arr_desc) {
        array_desc.shared_arr_desc->release();
    }
    
    if (!refCount && buffer) {
        if (!(flags & NON_OWNERSHIP_FLAG)) {
            delete[] static_cast<uint8_t *>(buffer);
            buffer = nullptr;
#ifdef DestructionLog
            std::cout << "deleted" << "\n";
#endif
        }
    } else {
        if (refCount && refCount->fetch_sub(1, std::memory_order_acq_rel) == 1) {
            delete[] static_cast<uint8_t *>(buffer);
            buffer = nullptr;
            delete refCount;
            refCount = nullptr;
#ifdef DestructionLog
            std::cout << "deleted" << "\n";
#endif
        }
    }
    releaseTape();
}
//};

bool compare_shapes(const matrix &a, const matrix &b) {
    if (a.dims != b.dims)
        return false;
    const size_m *sa = (a.dims > SBO_MAX_DIMS)
    ? a.array_desc.shared_arr_desc->shape()
    : a.array_desc.inline_buffer;
    const size_m *sb = (b.dims > SBO_MAX_DIMS)
    ? b.array_desc.shared_arr_desc->shape()
    : b.array_desc.inline_buffer;
    
    return memcmp(sa, sb, a.dims * sizeof(size_m)) == 0;
}

// template <typename Type>
// class matrix {
// public:
//     Type* buffer;
//     Size_m dims;
//     size_m* shape;
//     size_m* strides;
//     size_t total_size;
//     id<MTLBuffer> metalBuffer = nil;
//     uint8_t flags = 0;
//
//     std::vector<std::shared_ptr<matrix<dims, Type>>> parentNodes;
//     std::function<matrix<dims, Type>(matrix<dims, Type>&)> gradFunc =
//     nullptr; std::conditional_t<
//         !std::is_same<Type, Point3D>::value,
//         Type,
//         std::nullptr_t
//     > grad;
//
//
//     using initializer_type = typename nested_initializer_list<Type,
//     dims>::type; matrix(initializer_type nestedList) {
//         total_size = 1;
//         computeShape(nestedList, 0);
//         buffer = new Type[total_size];
//         int k = 0;
//         writeInBuffer(nestedList, k);
//         metalBuffer = [GlobalGPUManager.metalDevice
//         newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type)
//         options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//         pointer, NSUInteger length) {
//         }];
//         calcStrides();
//     }
//
//     matrix()
//       : buffer(nullptr), total_size(0), metalBuffer(nullptr), dims(0) {
//           std::cout << "Created" << "\n";
//       }
//
//     template <typename T>
//     struct is_initializer_list : std::false_type {};
//
//     template <typename T>
//     struct is_initializer_list<std::initializer_list<T>> : std::true_type {};
//
//     template <typename T>
//     void computeShape(const T& nestedList, int d) {
//         if constexpr (is_initializer_list<T>::value) {
//             if (d == dims - 1) {
//                 shape[d] = nestedList.size();
//                 total_size *= nestedList.size();
//             } else {
//                 shape[d] = nestedList.size();
//                 total_size *= nestedList.size();
//                 computeShape(*nestedList.begin(), d+1);
//             }
//         }
//     }
//
//     template <typename T>
//     void writeInBuffer(const T& nestedList, int& currentIndex) {
//         if constexpr (std::is_same<T, std::initializer_list<Type>>::value) {
//             memcpy(buffer + currentIndex, nestedList.begin(),
//             nestedList.size() * sizeof(Type)); currentIndex +=
//             nestedList.size();
//         } else {
//             for (auto i: nestedList) {
//                 writeInBuffer(i, currentIndex);
//             }
//         }
//
//     };
//
//     matrix(Type value) requires (dims == 0) {
//         buffer = new Type[1];
//         *buffer = value;
//         total_size = 1;
//         metalBuffer = [GlobalGPUManager.metalDevice
//         newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type)
//         options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//         pointer, NSUInteger length) {
//         }];
//     }
//
//     matrix(size_t reserveCapacity) requires (dims != 0) {
//         buffer = new Type[reserveCapacity];
//         shape[0] = reserveCapacity;
//         total_size = reserveCapacity;
//         metalBuffer = [GlobalGPUManager.metalDevice
//         newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type)
//         options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//         pointer, NSUInteger length) {
//         }];
//     }
//
//
//
//     static matrix<dims, Type> solid() {
//         matrix<dims, Type> outputM;
//     }
//
//     matrix<dims, Type> copy() {
//         matrix<dims, Type> result;
//         result.buffer = new Type[total_size];
//         memcpy(buffer, result.buffer, sizeof(Type) * result.total_size);
//         result.total_size = total_size;
//         memcpy(result.shape, shape, sizeof(size_m) * dims);
//         result.metalBuffer = [GlobalGPUManager.metalDevice
//         newBufferWithBytesNoCopy:result.buffer length:result.total_size *
//         sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void
//         * _Nonnull pointer, NSUInteger length) {
//         }];
//         return result;
//     }
//
//     void buildMetalBuffer() {
//         metalBuffer = [GlobalGPUManager.metalDevice
//         newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type)
//         options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//         pointer, NSUInteger length) {
//         }];
//     }
//     void copyFrom(matrix<dims, Type>& input) {
//         if (!buffer) {
//             buffer = new Type[input.total_size];
//             total_size = input.total_size;
//         }
//
//         memcpy(buffer, input.buffer, sizeof(Type) * input.total_size);
//         memcpy(shape,input.shape, sizeof(size_m) * dims);
//         metalBuffer = [GlobalGPUManager.metalDevice
//         newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type)
//         options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//         pointer, NSUInteger length) {
//         }];
//     }
//
//     static size_t accumul(const std::vector<size_t>& shape) {
//         size_t acc = 1;
//         for (int i = 0; i < shape.size(); i ++) {
//             acc *= shape[i];
//         }
//         return acc;
//     }
//
//     size_t accumul(int start, int end) const {
//         size_t acc = 1;
//         for (int i = start; i < end; i ++) {
//             acc *= shape[i];
//         }
//         return acc;
//     }
//
//     void calcStrides() {
//         size_t acc = 1;
//         for (int i = dims-1; i >= 0; i--) {
//             strides[i] = acc;
//             acc *= shape[i];
//         }
//     }
//
//     bool compareShapes(size_m* Othershape) {
//         bool res = true;
//         for (int i = 0; i < dims; i ++) {
//             if (shape[i] != Othershape[i]) {
//                 res = false;
//             }
//         }
//         return res;
//     }
//
//     bool compareShapes(size_m* Othershape, int end) {
//         if (end < 0) {
//             end = dims - end;
//         }
//         bool res = true;
//         for (int i = 0; i < end; i ++) {
//             if (shape[i] != Othershape[i]) {
//                 res = false;
//             }
//         }
//         return res;
//     }
//
//     static matrix<dims, Type> constant(const std::vector<size_t>& shapeI
//     ,Type value) {
//         matrix<dims, Type> result;
//         for (int i = 0; i< dims; i++) {
//             result.shape[i] = shapeI[i];
//         }
//         result.total_size = accumul(shapeI);
//         result.buffer = new Type[result.total_size];
//         std::fill(result.buffer, result.buffer + result.total_size, value);
//         result.metalBuffer = [GlobalGPUManager.metalDevice
//         newBufferWithBytesNoCopy:result.buffer length:result.total_size *
//         sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void
//         * _Nonnull pointer, NSUInteger length) {
//         }];
//         return result;
//     }
//
//     template <int dimsI>
//     static matrix<dims, Type> repeating(const std::vector<size_t>& shapeI,
//     const matrix<dimsI, Type>& pattern) {
//         matrix<dims, Type> result;
//         if (shapeI.size() + dimsI != dims) {
//             std::cerr << "Dimensions Dont Add up, Pattern: " << dimsI << " +
//             Repeat:" << shapeI.size() << " != Total Dim" << dims << "\n";
//             throw ;
//         }
//
//         for (int i = 0; i < shapeI.size(); i++) {
//             result.shape[i] = shapeI[i];
//         }
//         for (int i = 0; i < dimsI; i++) {
//             result.shape[shapeI.size() + i] = pattern.shape[i];
//         }
//
//         result.total_size = result.accumul(0, dims);
//         result.buffer = new Type[result.total_size];
//
////        for (int i = 0; i < result.accumul(0, shapeI.size()); ++i) {
////            memcpy(result.buffer + i * pattern.total_size, pattern.buffer,
/// pattern.total_size * sizeof(Type)); /        }
//        PatternFill(result.buffer, pattern.buffer, pattern.total_size *
//        sizeof(Type), result.accumul(0, shapeI.size())); result.metalBuffer =
//        [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer
//        length:result.total_size * sizeof(Type)
//        options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//        pointer, NSUInteger length) {
//        }];
//        return result;
//    }
//
//    static matrix<dims, Type> Range(Type start, std::initializer_list<size_m>
//    shapeI) {
//        if (shapeI.size() != dims) {std::cerr << "matrix: Shape should not
//        excede dim of matrix"; throw;} matrix<dims, Type> result;
//        std::copy(shapeI.begin(), shapeI.end(), result.shape);
//        result.total_size = result.accumul(0, dims);
//        result.buffer = new Type[result.total_size];
//
//        for (int i = 0; i < result.total_size; i++) {
//            result.buffer[i] = i+start;
//        }
//        result.buildMetalBuffer();
//        result.calcStrides();
//        return result;
//    }
//
//    void printShape() const {
//        std::cout << "Shape is { ";
//        for (int i=0; i<dims; i++) {
//            std::cout << shape[i] << ", ";
//        }
//        std::cout << "}\n";
//    }
//
//    void printShape(bool verbose) const {
//        if (verbose == true) { std::cout << "Shape is { "; }
//
//        for (int i=0; i<dims; i++) {
//            std::cout << shape[i] << ", ";
//        }
//        if (verbose == true) { std::cout << "}\n"; }
//
//    }
//
//    CGColorRef createCGColor(float r, float g, float b, float a) {
//        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
//        CGFloat components[] = {r, g, b, a};
//        CGColorRef color = CGColorCreate(colorSpace, components);
//        CGColorSpaceRelease(colorSpace);
//        return color;  // Remember to CFRelease when done using it
//    }
//
//    CGColorRef createCGColorFrommatrix(const matrix<1, int> &colorMat) {
//        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
//        CGFloat components[] = {colorMat.buffer[0] / 255.0, colorMat.buffer[1]
//        / 255.0, colorMat.buffer[2] / 255.0, colorMat.buffer[3] / 255.0};
//        CGColorRef color = CGColorCreate(colorSpace, components);
//        CGColorSpaceRelease(colorSpace);
//        return color;  // Remember to CFRelease when done using it
//    }
//
//
//    void drawText(char* text, matrix<1, int> point, const matrix<1, int>&
//    colorMat, float fontSize) {
//        if (colorMat.total_size != 4) {
//            std::cerr << "Error: 4 arguments are required for colour" << "\n";
//            return;
//        }
//
//        if (point.total_size != 2) {
//            std::cerr << "Error: 2 arguments are required for position" <<
//            "\n"; return;
//        }
//        CGColorRef color = createCGColorFrommatrix(colorMat);
//
//        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
//        size_t widthsize = total_size / shape[0];
//        CGContextRef context = CGBitmapContextCreate(
//                                                     buffer, shape[1],
//                                                     shape[0], 8 *
//                                                     sizeof(Type),
//                                                     sizeof(Type) * widthsize,
//                                                     colorSpace,
//                                                     kCGImageAlphaPremultipliedLast
//        );
//
//        if (!context) {
//            fprintf(stderr, "Failed to create bitmap context!\n");
//            CGColorSpaceRelease(colorSpace);
//        }
//
//        CFStringRef stringRef = CFStringCreateWithCString(NULL, text,
//        kCFStringEncodingUTF8); CTFontRef font =
//        CTFontCreateWithName(CFSTR("Helvetica"), fontSize, NULL);
//
//        NSDictionary *attributes = @{ (__bridge id)kCTFontAttributeName:
//        (__bridge id)font, (__bridge id)kCTForegroundColorAttributeName:
//        (__bridge id)color }; NSAttributedString *attributedString =
//        [[NSAttributedString alloc] initWithString:(__bridge NSString
//        *)stringRef attributes:attributes];
//
//
//        CTLineRef line = CTLineCreateWithAttributedString((__bridge
//        CFAttributedStringRef)attributedString);
//
//        CGContextSetTextPosition(context, point.buffer[0], point.buffer[1]);
//        CGContextSetTextDrawingMode(context, kCGTextFillClip);
//
//        // Draw text
//        CTLineDraw(line, context);
//    }
//
//
//    static matrix<dims, Type> blend(matrix<dims, Type>& mat1, matrix<dims,
//    Type>& mat2) {
//        matrix<dims, Type> output;
//        for (int i = 0; i < dims; i++) {
//            if (mat1.shape[i] != mat2.shape[i]) {
//                std::cerr << "shape error \n";
//                return output;
//            }
//            output.shape[i] = mat1.shape[i];
//        }
//        output.total_size = mat1.total_size;
//        output.buffer = new Type[output.total_size];
//        output.metalBuffer = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:output.buffer length:output.total_size *
//        sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void *
//        _Nonnull pointer, NSUInteger length) {
//        }];
//        if (!GlobalGPUManager.blendInit) {
//            GlobalGPUManager.initBlend();
//        }
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(mat1.shape[0] *
//        mat1.shape[1],1, 1);
//
//
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:mat1.buffer
//        length:mat1.total_size*sizeof(Type)
//        options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//        pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:mat2.buffer
//        length:mat2.total_size*sizeof(Type)
//        options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//        pointer, NSUInteger length) {
//        }];
//
//        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:output.buffer
//        length:output.total_size*sizeof(Type)
//        options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//        pointer, NSUInteger length) {
//        }];
//
//        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
//        [commandEncoder setBuffer:buffer2 offset:0 atIndex:1];
//        [commandEncoder setBuffer:buffer3 offset:0 atIndex:2];
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.blendCompute];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//        return output;
//    }
//
//    void invertImg(bool evenAlpha) {
//
//
//        if (!GlobalGPUManager.invertInitImg) {
//            GlobalGPUManager.initInvert();
//        }
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(shape[0] * shape[1],1, 1);
//
//
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type)
//        options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//        pointer, NSUInteger length) {
//        }];
//
//
//        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
//        [commandEncoder setBytes:&evenAlpha length:sizeof(bool) atIndex:1];
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.invertImgCompute];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//
//    void chromaKeyImg(simd_packed_char3 key) {
//
//
//        if (!GlobalGPUManager.chromaKeyInit) {
//            GlobalGPUManager.initchromaKey();
//        }
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(shape[0] * shape[1],1, 1);
//
//
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:this->buffer length:total_size*sizeof(Type)
//        options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//        pointer, NSUInteger length) {
//        }];
//
//
//        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
//        [commandEncoder setBytes:&key length:sizeof(simd_packed_char3)
//        atIndex:1]; [commandEncoder
//        setComputePipelineState:GlobalGPUManager.chromaKeyCompute];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//
//    matrix<dims, Type> addImg(matrix<dims, Type> &other, bool evenAlpha) {
//        matrix<dims, Type> result;
//        result.buffer = new Type[total_size];
//        result.total_size = total_size;
//        memcpy(result.shape, shape, sizeof(size_m) * dims);
//
//        if (!GlobalGPUManager.AddImgInit) {
//            GlobalGPUManager.initAddImg();
//        }
//
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type)
//        options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//        pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:other.buffer
//        newBufferWithByteopertsNoCopy:other.buffer
//        length:other.total_size*sizeof(Type)
//        options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//        pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type)
//        options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//        pointer, NSUInteger length) {
//        }];
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(shape[0] * shape[1], 1, 1);
//
//        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
//        [commandEncoder setBuffer:buffer2 offset:0 atIndex:1];
//        [commandEncoder setBuffer:buffer3 offset:0 atIndex:2];
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.AddImgCompute];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//
//        return result;
//    }
//
//    template <int i>
//    matrix<dims+i, Type> unsqueeze() {
//        matrix<dims+i, Type> output(total_size);
//        std::fill(output.shape, output.shape+i, 1);
//        memcpy(output.shape+i, shape, dims * sizeof(size_m));
//        memcpy(output.buffer, buffer, total_size * sizeof(Type));
//        return output;
//    }
//
//    matrix<dims, Type> MulConst(Type constant) {
//        matrix<dims, Type> result;
//        result.buffer = new Type[total_size];
//        result.total_size = total_size;
//        memcpy(result.shape, shape, sizeof(size_m) * dims);
//
//        result.parentNodes.push_back(std::make_shared<matrix<dims,
//        Type>>(*this)); result.gradFunc = [constant](matrix<dims, Type>&
//        selfs) {
//            auto p1 = selfs.parentNodes[0]->gradFunc ?
//            selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) :
//            selfs.parentNodes[0]->ones(); return p1 * constant;
//        };
//
//        if (!GlobalGPUManager.MulAllInit) {
//            GlobalGPUManager.initMulAll();
//        }
//
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type)
//        options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//        pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:&constant length:sizeof(Type)
//        options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//        pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type)
//        options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//        pointer, NSUInteger length) {
//        }];
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
//
//        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
//        [commandEncoder setBuffer:buffer2 offset:0 atIndex:1];
//        [commandEncoder setBuffer:buffer3 offset:0 atIndex:2];
//
//        int type = 0;
//        size_t stride = 1;
//
//        if constexpr (std::is_integral<Type>::value) {
//            type = 0;
//        } else if constexpr (std::is_floating_point<Type>::value) {
//            type = 1;
//        } else {
//            type = 2;
//        }
//
//        [commandEncoder setBytes:&type length:sizeof(int) atIndex:3];
//        [commandEncoder setBytes:&stride length:sizeof(size_m) atIndex:4];
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.MulAllCompute];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//
//        return result;
//    }
//
//    template <int dims2>
//    matrix<dims, Type> MulMat(const matrix<dims2, Type>& other) {
//
//
//
//        size_t stride = 1;
//        size_t strideI = 1;
//        for (int i = 0; i < dims; i++) {
//            for (int j = 0; j < dims2; j++) {
//                if (shape[i] == other.shape[j]) {
//                    stride = shape[i];
//                    strideI = accumul(i+1, dims);
//                }
//            }
//        }
//        matrix<dims, Type> result;
//        result.buffer = new Type[total_size];
//        result.total_size = total_size;
//        memcpy(result.shape, shape, sizeof(size_m) * dims);
//        memcpy(result.strides, strides, sizeof(size_m) * dims);
//        result.buildMetalBuffer();
//
//
//        result.parentNodes.push_back(std::make_shared<matrix<dims,
//        Type>>(*this));
//        result.parentNodes.push_back(std::make_shared<matrix<dims,
//        Type>>(other)); result.gradFunc = [](matrix<dims, Type>& selfs) {
//            auto p1 = selfs.parentNodes[0]->gradFunc ?
//            selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) :
//            selfs.parentNodes[0]->ones(); auto p2 =
//            selfs.parentNodes[1]->gradFunc ?
//            selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :
//            selfs.parentNodes[1]->ones(); return (p1) *
//            *(selfs.parentNodes[1])+
//                       p2 * *(selfs.parentNodes[0]);
//        };
//
//        if (!GlobalGPUManager.MulAllInit) {
//            GlobalGPUManager.initMulAll();
//        }
//
////        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice
/// newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type)
/// options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer,
/// NSUInteger length) { /        }]; /        id<MTLBuffer> buffer2 =
///[GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:other.buffer
/// length:other.total_size*sizeof(Type) options:MTLResourceStorageModeShared
/// deallocator:^(void * _Nonnull pointer, NSUInteger length) { /        }]; /
/// id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice
/// newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type)
/// options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer,
/// NSUInteger length) { /        }];
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
//
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:2];
//
//        int type = 0;
//
//
//        if constexpr (std::is_integral<Type>::value) {
//            if (std::is_unsigned<Type>::value) {
//                type = 2;
//            } else {
//                type = 0;
//            }
//
//        } else if constexpr (std::is_floating_point<Type>::value) {
//            type = 1;
//        } else {
//            type = 3;
//        }
//
//        [commandEncoder setBytes:&type length:sizeof(int) atIndex:3];
//        [commandEncoder setBytes:&stride length:sizeof(size_t) atIndex:4];
//        [commandEncoder setBytes:&strideI length:sizeof(size_t) atIndex:5];
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.MulAllCompute];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//
//        return result;
//    }
//
//
//    static matrix<dims, Type> concat(matrix<dims, Type>& Mat1, matrix<dims,
//    Type>& Mat2, int axis) {
//        matrix<dims, Type> output;
//        for (int i = 0; i < dims; i++) {
//            if (i == axis) {
//                output.shape[i] = Mat1.shape[i] + Mat2.shape[i];
//            } else {
//                output.shape[i] = Mat1.shape[i];
//            }
//        }
//        output.calcStrides();
//        output.total_size = output.accumul(0, dims);
//        output.buffer = new Type[output.total_size];
//        output.buildMetalBuffer();
//        size_t noOfOpp = output.accumul(0, axis);
//        size_t stride = output.accumul(axis, dims);
//
//        size_t strideMat1 = Mat1.accumul(axis, dims);
//
//        size_t strideMat2 = Mat2.accumul(axis, dims);
//
//        for (int i = 0; i < noOfOpp; i++) {
//            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1,
//            strideMat1 * sizeof(Type)); memcpy(output.buffer + strideMat1 + i
//            * stride, Mat2.buffer + i * strideMat2, strideMat2 *
//            sizeof(Type));
//        }
//
//        return output;
//    }
//
//    static void concat(matrix<dims, Type>& Mat1, matrix<dims, Type>& Mat2,
//    matrix<dims, Type>& output, int axis) {
//        for (int i = 0; i < dims; i++) {
//            if (i == axis) {
//                output.shape[i] = Mat1.shape[i] + Mat2.shape[i];
//            } else {
//                output.shape[i] = Mat1.shape[i];
//            }
//        }
//        output.calcStrides();
//        if (output.total_size != output.accumul(0, dims)) {
//            delete [] output.buffer;
//            output.total_size = output.accumul(0, dims);
//            output.buffer = new Type[output.total_size];
//            output.buildMetalBuffer();
//        }
//
//        size_t noOfOpp = output.accumul(0, axis);
//        size_t stride = output.accumul(axis, dims);
//
//        size_t strideMat1 = Mat1.accumul(axis, dims);
//
//        size_t strideMat2 = Mat2.accumul(axis, dims);
//
//        for (int i = 0; i < noOfOpp; i++) {
//            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1,
//            strideMat1 * sizeof(Type)); memcpy(output.buffer + strideMat1 + i
//            * stride, Mat2.buffer + i * strideMat2, strideMat2 *
//            sizeof(Type));
//        }
//    }
//
//    static void concatID(matrix<dims, Type>& Mat1, matrix<dims, Type>& Mat2,
//    matrix<dims+1, Type>& output) {
//        int axis = dims+1;
//        for (int i = 0; i < dims; i++) {
//            if (i == axis) {
//                output.shape[i] = Mat1.shape[i] + Mat2.shape[i];
//            } else {
//                output.shape[i] = Mat1.shape[i];
//            }
//        }
//        output.calcStrides();
//        if (output.total_size != output.accumul(0, dims)) {
//            delete [] output.buffer;
//            output.total_size = output.accumul(0, dims);
//            output.buffer = new Type[output.total_size];
//            output.buildMetalBuffer();
//        }
//
//        size_t noOfOpp = output.accumul(0, axis);
//        size_t stride = output.accumul(axis, dims);
//
//        size_t strideMat1 = Mat1.accumul(axis, dims);
//
//        size_t strideMat2 = Mat2.accumul(axis, dims);
//
//        for (int i = 0; i < noOfOpp; i++) {
//            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1,
//            strideMat1 * sizeof(Type)); memcpy(output.buffer + strideMat1 + i
//            * stride, Mat2.buffer + i * strideMat2, strideMat2 *
//            sizeof(Type));
//        }
//
//        return output;
//    }
//
//    static matrix<dims+1, Type> concatID(matrix<dims, Type>& Mat1,
//    matrix<dims, Type>& Mat2) {
//        matrix<dims+1, Type> output;
//        int axis = dims+1;
//        for (int i = 0; i < dims; i++) {
//            if (i == axis) {
//                output.shape[i] = Mat1.shape[i] + Mat2.shape[i];
//            } else {
//                output.shape[i] = Mat1.shape[i];
//            }
//        }
//        output.calcStrides();
//        output.total_size = output.accumul(0, dims);
//        output.buffer = new Type[output.total_size];
//        output.buildMetalBuffer();
//
//        size_t noOfOpp = output.accumul(0, axis);
//        size_t stride = output.accumul(axis, dims);
//
//        size_t strideMat1 = Mat1.accumul(axis, dims);
//
//        size_t strideMat2 = Mat2.accumul(axis, dims);
//
//        for (int i = 0; i < noOfOpp; i++) {
//            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1,
//            strideMat1 * sizeof(Type)); memcpy(output.buffer + strideMat1 + i
//            * stride, Mat2.buffer + i * strideMat2, strideMat2 *
//            sizeof(Type));
//        }
//
//        return output;
//    }
//
//    static matrix<dims, Type> zeros(std::initializer_list<size_m> shapeI) {
//        if (shapeI.size() != dims) {std::cerr << "matrix: dimensions in
//        provided shape must match the dims of " << dims << "\n"; throw;}
//        matrix<dims, Type> output;
//        memcpy(output.shape, shapeI.begin(), dims * sizeof(size_m));
//        output.calcStrides();
//        output.total_size = output.accumul(0, dims);
//        output.buffer = new Type[output.total_size];
//        output.buildMetalBuffer();
//        memset(output.buffer, 0, output.total_size * sizeof(Type));
//        return output;
//    }
//
//    static matrix<dims, Type> ones(std::initializer_list<size_m> shapeI) {
//        if (shapeI.size() != dims) {std::cerr << "matrix: dimensions in
//        provided shape must match the dims of " << dims << "\n"; throw;}
//        matrix<dims, Type> output;
//        memcpy(output.shape, shapeI.begin(), dims * sizeof(size_m));
//        output.calcStrides();
//        output.total_size = output.accumul(0, dims);
//        output.buffer = new Type[output.total_size];
//        output.buildMetalBuffer();
//        std::fill(output.buffer, output.buffer + output.total_size, 1);
//        return output;
//    }
//
//    matrix<dims, Type> ones() {
//        matrix<dims, Type> output;
//        memcpy(output.shape, shape, dims * sizeof(size_m));
//        output.calcStrides();
//        output.total_size = output.accumul(0, dims);
//        output.buffer = new Type[output.total_size];
//        output.buildMetalBuffer();
//        std::fill(output.buffer, output.buffer + output.total_size, 1);
//        return output;
//    }
//
//    matrix<dims, Type> zeros() {
//        matrix<dims, Type> output;
//        memcpy(output.shape, shape, dims * sizeof(size_m));
//        output.calcStrides();
//        output.total_size = output.accumul(0, dims);
//        output.buffer = new Type[output.total_size];
//        output.buildMetalBuffer();
//        memset(output.buffer, 0, output.total_size * sizeof(Type));
//        return output;
//    }
//
//    static matrix<dims, Type> withShape(std::initializer_list<size_m> shapeI)
//    {
//        if (shapeI.size() != dims) {std::cerr << "matrix: dimensions in
//        provided shape must match the dims of " << dims << "\n"; throw;}
//        matrix<dims, Type> output;
//        memcpy(output.shape, shapeI.begin(), dims * sizeof(size_m));
//        output.calcStrides();
//        output.total_size = output.accumul(0, dims);
//        output.buffer = new Type[output.total_size];
//        output.buildMetalBuffer();
//        return output;
//    }
//
////    template <typename = std::enable_if_t<(dims == 2)>>
//    static matrix<2, Type> eye(uint m, uint n, int k) {
//        matrix<2, Type> output = matrix<2, Type>::zeros({m, n});
//        uint iteration = MIN(m, n-abs(k));
//        if (0 <= k) {
//            for (int i = 0; i < iteration; i++) {
//                // [i, j+k]
//                output.buffer[i * output.shape[1] + i + k] = 1;
//            }
//
//        } else {
//            for (int i = 0; i < iteration; i++) {
//                // [i, i-abs(k)] => same as shifting it down => [i+abs(k), i]
//                // Since k is -ve => [i-k, i]
//                // Moving the Diagnol Left is Same as moving it above as y =
//                (x + k) ==> (y - k) = x output.buffer[(i - k) *
//                output.shape[1] + i] = 1;
//            }
//        }
//        return output;
//    }
//
//    static matrix<2, Type> eye(uint m) {
//        return eye(m, m, 0);
//    }
//
//    void drawRect(simd_int4 rect, matrix<dims-2, Type> element) {
//        int X = rect[0];
//        int Y = rect[1];
//        int width = rect[2];
//        int height = rect[3];
//        for (int i = 0; i < dims - 2; i++) {
//            if (shape[i+2] != element.shape[i]) {
//                std::cerr << "Error Dimensions not equal at index " << i <<
//                "as " << shape[i+2] << " != " << element.shape[i] << "\n";
//                std::cerr << shape << " != " << element.shape << "\n";
//                return;
//            }
//        }
//
//
//        if (X + width > shape[1] || Y + height > shape[0]) {
//            std::cerr << "Error Dimensions excedeError Dimensions excede \n";
//            return;
//        }
//
//
//        size_t widthsize = total_size / shape[0];
//        size_t elementSize = total_size / (shape[0] * shape[1]);
//
//        Type* rowBuffer = new Type[width * element.total_size];
//
//        for (size_t i = 0; i < width; i++) {
//            memcpy(buffer + Y * widthsize + (X + i) * elementSize,
//            element.buffer, element.total_size * sizeof(Type));
//        }
//
//        for (int j = Y+1; j < Y + height; j++) {
//            memcpy(buffer + j * widthsize + X * elementSize , buffer + Y *
//            widthsize + X * elementSize, element.total_size * width *
//            sizeof(Type));
//        }
//    }
//
//    simd_float2 NormaliseShapeToScreen(simd_int2 deviceCoord) {
//        simd_float2 size = simd_make_float2(shape[1], shape[0]);
//
//        return simd_make_float2((float)deviceCoord.x, (float)deviceCoord.y) /
//        size;
//    }
//
//    void drawRect(simd_int2 p1, simd_int2 p2, matrix<dims, Type> element) {
//        for (int i = 0; i < shape.size() - 2; i++) {
//            if (shape[i+2] != element.shape[i]) {
//                std::cerr << "Error Dimensions not equal at index " << i <<
//                "as " << shape[i+2] << " != " << element.shape[i] << "\n";
//                std::cerr << shape << " != " << element.shape << "\n";
//                return;
//            }
//        }
//
//        auto xDiff = p1.x-p2.x;
//        auto yDiff = p1.y-p2.y;
//
//        int width = abs(xDiff);
//        int height = abs(yDiff);
//
//        int X;
//        int Y;
//
//        if (xDiff > 0) {
//            X = p2.x;
//        } else {
//            X = p1.x;
//        }
//
//        if (yDiff > 0) {
//            Y = p2.y;
//        } else {
//            Y = p1.y;
//        }
//
//        if (X + width > shape[1] || Y + height > shape[0]) {
//            std::cerr << "Error Dimensions excede \n";
//            return;
//        }
//
//
//        size_t widthsize = total_size / shape[0];
//        size_t elementSize = total_size / (shape[0] * shape[1]);
//
//        Type* rowBuffer = new Type[width * element.total_size];
//
//        for (size_t i = 0; i < width; i++) {
//            memcpy(buffer + Y * widthsize + (X + i) * elementSize,
//            element.buffer, element.total_size * sizeof(Type));
//        }
//
//        for (int j = Y+1; j < Y + height; j++) {
//            memcpy(buffer + j * widthsize + X * elementSize , buffer + Y *
//            widthsize + X * elementSize, element.total_size * width *
//            sizeof(Type));
//        }
//    }
//    void drawElipse(const simd_int4& rect, const matrix<dims-2, Type>&
//    element) {
//        int X = rect[0];
//        int Y = rect[1];
//        int width = rect[2];
//        int height = rect[3];
//        for (int i = 0; i < dims - 2; i++) {
//            if (shape[i+2] != element.shape[i]) {
//                std::cerr << "Elipse: Error Dimensions not equal at index " <<
//                i << "as " << shape[i+2] << " != " << element.shape[i] <<
//                "\n"; std::cerr << shape << " != " << element.shape << "\n";
//                return;
//            }
//        }
//        if (X + width > shape[1] || Y + height > shape[0]) {
//            std::cerr << "Error Dimensions excede \n";
//            return;
//        }
//
//
//        size_t widthsize = total_size / shape[0];
//        size_t elementSize = total_size / (shape[0] * shape[1]);
//        auto centre = simd_make_float2(X + (width / 2.0), Y + (height / 2.0));
//        auto rad = simd_make_float2(width, height) / 2;
//
//        for (int i = X; i < X + width; i ++) {
//            for (int j = Y; j < Y + height; j ++) {
//                auto coord = simd_make_float2(i, j);
//                float S1 = simd_dot(((coord - centre) / rad), ((coord -
//                centre) / rad)) - 1.0;
//
//                if (S1 < 0.0) {
//                    memcpy(buffer + j * widthsize + i * elementSize ,
//                    element.buffer, element.total_size * sizeof(Type));
//                }
//            }
//        }
//    }
//
//    simd_float4 toSimdFloat4() {
//        return simd_make_float4(buffer[0], buffer[1], buffer[2], buffer[3]);
//    }
//    simd_float3 toSimdFloat3() {
//        return simd_make_float3(buffer[0], buffer[1], buffer[2]);
//    }
//
//    simd_float4 toSimdFloat4(size_t i, size_t j) {
//        size_t offset = i * (shape[1] * 4) + j * (4);
//        return simd_make_float4(buffer[offset + 0], buffer[offset + 1],
//        buffer[offset + 2], buffer[offset + 3]);
//    }
//    CVPixelBufferRef createPixelBufferFromMat() const {
//
//        size_t width = shape[1];
//        size_t height = shape[0];
//        size_t channels = shape[2];
//        NSDictionary *pixelAttributes =
//        @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
//
//        CVPixelBufferRef pixelBuffer;
//        CVPixelBufferCreate(kCFAllocatorDefault, width, height,
//        kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)pixelAttributes,
//        &pixelBuffer);
//
//        CVPixelBufferLockBaseAddress(pixelBuffer, 0);
//        void *bufferAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
//
//        size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
//
//         // Assuming channels is the number of bytes per pixel (should be 4
//         for BGRA) size_t copyBytesPerRow = width * channels;
//
//         // Copy row by row to respect the pixel buffer's stride.
//         for (size_t row = 0; row < height; row++) {
//             memcpy((uint8_t *)bufferAddress + row * bytesPerRow,
//                    buffer + row * copyBytesPerRow,
//                    copyBytesPerRow);
//         }
//        CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
//        return pixelBuffer;
//    }
//
//    static matrix<3, uint8_t> fromImage(bool includeDepth) {
//        #if !TARGET_OS_IPHONE
//        CFStringRef path = CFStringCreateWithCString(NULL,
//        "/Users/adityadude/Documents/IMG_1278.JPG", kCFStringEncodingUTF8);
//        CFURLRef url = CFURLCreateWithFileSystemPath(NULL, path,
//        kCFURLPOSIXPathStyle, false); CGImageSourceRef source =
//        CGImageSourceCreateWithURL(url, NULL); CGImageRef cgImage =
//        CGImageSourceCreateImageAtIndex(source, 0, NULL); CFRelease(url);
//        CFRelease(path);
//        #endif
//
//        #if TARGET_OS_IPHONE
//        UIImage *image = [UIImage imageNamed:@"IMG_1278"];
//        CGImageRef cgImage = image.CGImage;
//        #endif
//
//        if (!cgImage) {
//            std::cerr << "Failed to create CGImage" << std::endl;
////            return;
//        }
//        size_t Imgwidth = CGImageGetWidth(cgImage);
//        size_t Imgheight = CGImageGetHeight(cgImage);
//        std::cout << "Img of Width: " <<Imgwidth<<"and Height: " << Imgheight
//        << "Loaded \n"; size_t bytesPerRow = 4 * Imgwidth; void *data =
//        malloc(bytesPerRow * Imgheight); CGContextRef context =
//        CGBitmapContextCreate(data, Imgwidth, Imgheight, 8, bytesPerRow,
//                                                     CGImageGetColorSpace(cgImage),
//                                                     kCGImageAlphaPremultipliedLast
//                                                     |
//                                                     kCGBitmapByteOrder32Big);
//        CGContextDrawImage(context, CGRectMake(0, 0, Imgwidth, Imgheight),
//        cgImage); CGContextRelease(context); CGImageRelease(cgImage);
//
//        uint8_t* pixelData = static_cast<uint8_t*>(data);
//
//        matrix<3, uint8_t> result;
//        result.buffer = pixelData;
//        result.shape[0] = Imgheight;
//        result.shape[1] = Imgwidth;
//        result.shape[2] = 4;
//        result.calcStrides();
//        result.total_size = Imgwidth * Imgheight * 4;
//        result.buildMetalBuffer();
//        return result;
//    }
//
//    static matrix<4, uint8_t> fromVideo(const char* vidPath) {
//    //    const char* vidPath = "/Users/adityadude/Downloads/WhatsApp Video
//    2025-01-01 at 14.49.11.mp4";
//        NSString *filePath = [NSString stringWithUTF8String:vidPath];
//
//        NSURL* url = [NSURL fileURLWithPath:filePath];
//        AVURLAsset* asset = [[AVURLAsset alloc] initWithURL:url options:nil];
//        NSLog(@"%@", asset);
//        if (!asset) {
//
//            std::cerr << "Asset Invalid \n";
//        }
//        __block AVAssetTrack* videoTrack;
//
//        [asset loadTracksWithMediaType:AVMediaTypeVideo
//        completionHandler:^(NSArray<AVAssetTrack *> * videoArray, NSError *
//        _Nullable) {
//            videoTrack = videoArray.firstObject;
//        }];
//        AVAssetReader* reader = [[AVAssetReader alloc] initWithAsset:asset
//        error:nil]; NSDictionary* outputSettings = @{
//                (NSString*)kCVPixelBufferPixelFormatTypeKey:
//                @(kCVPixelFormatType_32BGRA)  // 4-channel (BGRA)
//        };
//        for (AVAssetTrack *track in asset.tracks) {
//            NSLog(@"Track media type: %@", track.mediaType);
//        }
//        @try {
//            AVAssetReaderTrackOutput* trackOutput = [[AVAssetReaderTrackOutput
//            alloc] initWithTrack:videoTrack outputSettings:outputSettings];
//            // Proceed with using trackOutput
//        }
//        @catch (NSException *exception) {
//            NSLog(@"Exception occurred: %@, %@", exception.name,
//            exception.reason);
//            // Handle the exception appropriately
//        }
//        AVAssetReaderTrackOutput* trackOutput = [[AVAssetReaderTrackOutput
//        alloc] initWithTrack:videoTrack outputSettings:outputSettings];
//        [reader addOutput:trackOutput];
//
//        if ([NSThread isMainThread]) {
//            NSLog(@"Running on the main thread");
//        } else {
//            NSLog(@"Not running on the main thread");
//        }
//
//        // Blocking the Thread till we load the duration async
//        dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);
//
//        __block CMTime duration;
//
//        [asset loadValuesAsynchronouslyForKeys:@[@"duration"]
//        completionHandler:^{
//            NSError *error = nil;
//            AVKeyValueStatus status = [asset statusOfValueForKey:@"duration"
//            error:&error]; if (status == AVKeyValueStatusLoaded) {
//                duration = asset.duration;
//            } else {
//                // Handle error
//            }
//            dispatch_semaphore_signal(semaphore);
//        }];
//
//        dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);
//
//        [reader startReading];
//
//        size_t frameCount = (size_t)(CMTimeGetSeconds(duration) *
//        videoTrack.nominalFrameRate); size_t width =
//        (size_t)videoTrack.naturalSize.width; size_t height =
//        (size_t)videoTrack.naturalSize.height; size_t channels = 4; // BGRA
//        format has 4 channels std::cout << "data " << frameCount <<" "<<
//        height <<" " <<width<< " "<<channels; uint8_t* values = new
//        uint8_t[width*height*frameCount*channels]; NSLog(@"CMTime %f",
//        CMTimeGetSeconds(duration));
//
//        size_t frameIndex = 0;
//        while ([reader status] == AVAssetReaderStatusReading) {
//            CMSampleBufferRef sampleBuffer = [trackOutput
//            copyNextSampleBuffer]; if (!sampleBuffer) {std::cout << "error ";
//            break; } CVImageBufferRef imageBuffer =
//            CMSampleBufferGetImageBuffer(sampleBuffer);
//            CVPixelBufferLockBaseAddress(imageBuffer,
//            kCVPixelBufferLock_ReadOnly); uint8_t* baseAddress =
//            (uint8_t*)CVPixelBufferGetBaseAddress(imageBuffer); size_t
//            bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer); for
//            (size_t y = 0; y < height; y++) {
//                memcpy(values + (frameIndex * height * width * channels) + (y
//                * width * channels),
//                       baseAddress + (y * bytesPerRow),
//                       width * channels * sizeof(Type));
//            }
//            frameIndex++;
//            CVPixelBufferUnlockBaseAddress(imageBuffer,
//            kCVPixelBufferLock_ReadOnly);
//        }
//
//        matrix<4, uint8_t> result = matrix();
//        result.shape[0] = frameCount;
//        result.shape[1] = height;
//        result.shape[2] = width;
//        result.shape[3] = channels;
//        result.total_size = width*height*frameCount*channels;
//        result.buffer = values;
//        result.calcStrides();
//        result.buildMetalBuffer();
//        result.flags |= (1u << 2);
//        return result;
//    }
//
//    void CopyToTexture(id<MTLTexture> texture) {
//        MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)shape[1],
//        (NSUInteger)shape[0]); NSUInteger bytesPerRow = shape[1] * 4;  // 4
//        bytes per pixel for BGRA8
//
////        id<MTLBuffer> Metalbuffer = [GlobalGPUManager.metalDevice
/// newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type)
/// options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer,
/// NSUInteger length) { /        }];
//
//        id<MTLCommandBuffer> commandBuffer = [GlobalGPUManager.gCommandQueue
//        commandBuffer]; id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer
//        blitCommandEncoder];
//
//        [blitEncoder copyFromBuffer:metalBuffer
//                       sourceOffset:0
//                  sourceBytesPerRow:bytesPerRow
//                sourceBytesPerImage:bytesPerRow * shape[0]
//                         sourceSize:region.size
//                          toTexture:texture
//                   destinationSlice:0
//                   destinationLevel:0
//                  destinationOrigin:region.origin];
//
//        [blitEncoder endEncoding];
//        [commandBuffer commit];
//    }
//
//    id<MTLTexture> ToMTLTexture() {
//        id<MTLTexture> resultTexture;
//        MTLTextureDescriptor* drawableDesc = [MTLTextureDescriptor
//        texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB
//                                                                                                width:(NSUInteger)shape[1]
//                                                                                               height:(NSUInteger)shape[0]
//                                                                                            mipmapped:NO];
//        drawableDesc.usage = MTLTextureUsageRenderTarget |
//        MTLTextureUsageShaderRead;
//        // Use shared storage so the CPU can read the texture data.
//        drawableDesc.storageMode = MTLStorageModeShared;
//
//
//        resultTexture = [GlobalGPUManager.metalDevice
//        newTextureWithDescriptor:drawableDesc];
//
//        CopyToTexture(resultTexture);
//
//        return resultTexture;
//    }
//
//    void printNonCont() const {
//        if (!buffer) {return; }
//        else if (dims == 2) {
//            std::cout << "{ \n";
//            for (size_t i = 0; i < shape[0]; i++) {
//                std::cout << "{ ";
//                for (size_t j = 0; j < shape[1]; j++) {
//                    std::cout << buffer[strides[0] * i + j] << " ";
//                }
//                std::cout << "} \n";
//            }
//            std::cout << "} \n";
//        } else if (dims == 3) {
//            std::cout << "{ ";
//            for (size_t i = 0; i < shape[0]; i++) {
//                std::cout << "{ ";
//                for (size_t j = 0; j < shape[1]; j++) {
//                    std::cout << " { ";
//                    for (size_t k = 0; k < shape[2]; k++) {
//                        std::cout << buffer[strides[0] * i + strides[1] * j +
//                        k] << ", ";
//                    }
//                    std::cout << "}, ";
//                }
//                std::cout << "}, \n";
//            }
//            std::cout << "}, \n";
//        } else if (dims == 4) {
//            for (size_t l = 0; l < shape[0]; l++) {
//                for (size_t i = 0; i < shape[1]; i++) {
//                    for (size_t j = 0; j < shape[2]; j++) {
//                        std::cout << "{ ";
//                        for (size_t k = 0; k < shape[3]; k++) {
//                            std::cout << buffer[strides[0] * l + strides[1] *
//                            i + strides[2] * j + k] << " ";
//                        }
//                        std::cout << "} ";
//                    }
//                    std::cout << std::endl;
//                }
//                std::cout <<"\n";
//            }
//        }
//
//        else {
//            std::cerr << "Printing only supported for 2D matrices." <<
//            std::endl; return;
//        }
//    }
//
//    void print() const {
//        if (!buffer) {return; }
//        if (flags & (1u << 1)) { printNonCont(); return;}
//        if (dims == 0) {
//            std::cout << "{ ";
//                std::cout << buffer[0] << " ,";
//            std::cout << "} \n";
//        }
//        else if (dims == 1) {
//            std::cout << "{ ";
//            for (size_t i = 0; i < shape[0]; i++) {
//                std::cout << buffer[i] << " ,";
//            }
//            std::cout << "} \n";
//        }
//        else if (dims == 2) {
//            std::cout << "{ \n";
//            for (size_t i = 0; i < shape[0]; i++) {
//                std::cout << "{ ";
//                for (size_t j = 0; j < shape[1]; j++) {
//                    std::cout << buffer[shape[1] * i + j] << " ";
//                }
//                std::cout << "} \n";
//            }
//            std::cout << "} \n";
//        } else if (dims == 3) {
//            std::cout << "{ ";
//            for (size_t i = 0; i < shape[0]; i++) {
//                std::cout << "{ ";
//                for (size_t j = 0; j < shape[1]; j++) {
//                    std::cout << " { ";
//                    for (size_t k = 0; k < shape[2]; k++) {
//                        std::cout << buffer[shape[2]*(shape[1] * i + j) + k]
//                        << ", ";
//                    }
//                    std::cout << "}, ";
//                }
//                std::cout << "}, \n";
//            }
//            std::cout << "}, \n";
//        } else if (dims == 4) {
//            for (size_t l = 0; l < shape[0]; l++) {
//                for (size_t i = 0; i < shape[1]; i++) {
//                    for (size_t j = 0; j < shape[2]; j++) {
//                        std::cout << "{ ";
//                        for (size_t k = 0; k < shape[3]; k++) {
//                            std::cout << buffer[shape[3]*(shape[2]*(shape[1] *
//                            l + i) + j)  + k] << " ";
//                        }
//                        std::cout << "} ";
//                    }
//                    std::cout << std::endl;
//                }
//                std::cout <<"\n";
//            }
//        }
//
//        else {
//            std::cerr << "Printing only supported for 2D matrices." <<
//            std::endl; return;
//        }
//
//    }
//
//
//    Type* operator()(size_t i) const {
//        Type* value = buffer + i * total_size / shape[0];
//        return value;
//    }
//
//    Type* operator()(size_t i, size_t j) {
//        Type* value = buffer + i * (total_size / shape[0]) + j * (total_size /
//        (shape[0] * shape[1])); return value;
//    }
//
//    Type* operator()(size_t i, size_t j, size_t k) {
//        Type* value = buffer + i * (total_size / shape[0]) + j * (total_size /
//        (shape[0] * shape[1])) + k * (total_size / (shape[0] * shape[1] *
//        shape[2])); return value;
//    }
//
//    template <int Newdims, typename OutType>
//    void To(matrix<Newdims, OutType>& output, int type) const {
//        int valuesIn = 1;
//        int valuesOut = 1;
//        int currentTypeCode = get_dtype_code<Type>();
//        int OutTypeCode = get_dtype_code<OutType>();
//
//        if (currentTypeCode > valueLimit) {
//            auto typeInfo = get_dtype_info<Type>();
//            valuesIn = typeInfo.values;
//            currentTypeCode = typeInfo.baseType;
//        }
//
//        if (OutTypeCode > valueLimit) {
//            auto typeInfo = get_dtype_info<OutType>();
//            valuesOut = typeInfo.values;
//            OutTypeCode = typeInfo.baseType;
//        }
//
//        if (output.total_size * valuesOut != total_size * valuesIn) {
//            std::cerr << "matrix: Invalid Dims, cannot convert from (" <<
//            total_size <<", " << valuesIn << "(imp)) To (" <<
//            output.total_size <<", " << valuesOut << ") \n";
//        }
//
//        if (!GlobalGPUManager.typeCasting[currentTypeCode][OutTypeCode]) {
//            GlobalGPUManager.initTypeCasting(currentTypeCode, OutTypeCode);
//        }
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(total_size * valuesIn, 1,
//        1);
//
//
////        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice
/// newBufferWithBytesNoCopy:output.buffer
/// length:output.total_size*sizeof(OutType)
/// options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer,
/// NSUInteger length) { /        }]; / id<MTLBuffer> buffer2 =
/// [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer
/// length:total_size*sizeof(Type) options:MTLResourceStorageModeShared
/// deallocator:^(void * _Nonnull pointer, NSUInteger length) { /        }];
//
//
//        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
////        [commandEncoder setBytes:&type length:sizeof(int) atIndex:2];
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.typeCasting[currentTypeCode][OutTypeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//
//    template <typename OutType>
//    void To(matrix<dims, OutType>& output, int type) const {
//        int valuesIn = 1;
//        int valuesOut = 1;
//        int currentTypeCode = get_dtype_code<Type>();
//        int OutTypeCode = get_dtype_code<OutType>();
//
//        if (currentTypeCode > valueLimit) {
//            auto typeInfo = get_dtype_info<Type>();
//            valuesIn = typeInfo.values;
//            currentTypeCode = typeInfo.baseType;
//        }
//
//        if (OutTypeCode > valueLimit) {
//            auto typeInfo = get_dtype_info<OutType>();
//            valuesOut = typeInfo.values;
//            OutTypeCode = typeInfo.baseType;
//        }
//
//        if (output.total_size * valuesOut != total_size * valuesIn) {
//            std::cerr << "matrix: Invalid Dims, cannot convert from (" <<
//            total_size <<", " << valuesIn << "(imp)) To (" <<
//            output.total_size <<", " << valuesOut << ") \n";
//        }
//        if (!GlobalGPUManager.typeCasting[currentTypeCode][OutTypeCode]) {
//            GlobalGPUManager.initTypeCasting(currentTypeCode, OutTypeCode);
//        }
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(total_size * valuesIn ,1,
//        1);
//
//
////        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice
/// newBufferWithBytesNoCopy:output.buffer
/// length:output.total_size*sizeof(OutType)
/// options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer,
/// NSUInteger length) { /        }]; / id<MTLBuffer> buffer2 =
/// [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer
/// length:total_size*sizeof(Type) options:MTLResourceStorageModeShared
/// deallocator:^(void * _Nonnull pointer, NSUInteger length) { /        }];
//
//        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.typeCasting[currentTypeCode][OutTypeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//
////    explicit operator matrix<dims, float>() const {
////        matrix<dims, float> result;
////        result.total_size = total_size;
////        result.buffer = new float[total_size];
////        result.metalBuffer = [GlobalGPUManager.metalDevice
/// newBufferWithBytesNoCopy:result.buffer length:result.total_size *
/// sizeof(float) options:MTLResourceStorageModeShared deallocator:^(void *
///_Nonnull pointer, NSUInteger length) { /        }]; /
/// std::memcpy(result.shape, shape, sizeof(size_m) * dims);
////
////        this->To<float>(result, 0);
////
////        return result;
////    }
//
//    template <typename OutType>
//    explicit operator matrix<dims, OutType>() const {
//        int valuesIn = 1;
//        int valuesOut = 1;
//        int currentTypeCode = get_dtype_code<Type>();
//        int OutTypeCode = get_dtype_code<OutType>();
//
//        if (valuesIn != valuesOut) {
//            std::cerr << "matrix: increase dims" << "\n";
//        }
//
//        if (currentTypeCode > valueLimit) {
//            auto typeInfo = get_dtype_info<Type>();
//            valuesIn = typeInfo.values;
//            currentTypeCode = typeInfo.baseType;
//        }
//
//        if (OutTypeCode > valueLimit) {
//            auto typeInfo = get_dtype_info<OutType>();
//            valuesOut = typeInfo.values;
//            OutTypeCode = typeInfo.baseType;
//        }
//
//        matrix<dims, OutType> result;
//        result.total_size = total_size ;
//        result.buffer = new OutType[total_size];
//        result.buildMetalBuffer();
//        std::memcpy(result.shape, shape, sizeof(size_m) * dims);
//        this->To<OutType>(result, 0);
//        return result;
//    }
//
//    template <typename OutType, int DimsNew>
//    explicit operator matrix<DimsNew, OutType>() const {
//        int valuesIn = 1;
//        int valuesOut = 1;
//        int currentTypeCode = get_dtype_code<Type>();
//        int OutTypeCode = get_dtype_code<OutType>();
//        uint8_t dimBias = 0;
//
//
//
//        if (currentTypeCode > valueLimit) {
//            auto typeInfo = get_dtype_info<Type>();
//            valuesIn = typeInfo.values;
//            currentTypeCode = typeInfo.baseType;
//        }
//
//        if (OutTypeCode > valueLimit) {
//            auto typeInfo = get_dtype_info<OutType>();
//            valuesOut = typeInfo.values;
//            OutTypeCode = typeInfo.baseType;
//        }
//
//        matrix<DimsNew, OutType> result;
//        result.total_size = (total_size * valuesIn) / valuesOut;
//        result.buffer = new OutType[result.total_size];
//        result.buildMetalBuffer();
//
//        if (DimsNew - dims == 1){
//            result.shape[DimsNew-1] = valuesIn;
//            std::memcpy(result.shape, shape, sizeof(size_m) * dims);
//        } else if (dims - DimsNew == 1) {
//            if (shape[dims-1] != valuesOut) {
//                std::cerr << "matrix: For conversion last dim should be " <<
//                valuesOut << "\n";
//            }
//            std::memcpy(result.shape, shape, sizeof(size_m) * DimsNew);
//        } else {
//            std::cerr << "matrix: not supported as of yet" << "\n";
//        }
//
//
//        this->To(result, 0);
//
//        return result;
//    }
//
//
//
//
//    template <int dimsNew>
//    explicit operator matrix<dimsNew, Type>() {
//        return this->unsqueeze<dimsNew-dims>();
//    }
//
//
//
////    template<int DimsNew>
////    explicit operator matrix<DimsNew, float>() const {
////        matrix<DimsNew, float> result;
////        uint8_t dimBias = 0;
////        int code = 00;
////        if (std::is_same<Type, simd_float2>::value) {
////            result.total_size = total_size * 2;
////            result.shape[DimsNew-1] = 2;
////            dimBias = 1;
////        } else if (std::is_same<Type, simd_float3>::value) {
////            result.total_size = total_size * 3;
////            result.shape[DimsNew-1] = 3;
////            dimBias = 1;
////        } else if (std::is_same<Type, simd_float4>::value) {
////            result.total_size = total_size * 4;
////            result.shape[DimsNew-1] = 4;
////            dimBias = 1;
////        } else if (std::is_same<Type, simd_float2x2>::value) {
////            result.total_size = total_size * 4;
////            result.shape[DimsNew-1] = 2;
////            result.shape[DimsNew-2] = 2;
////            dimBias = 2;
////        } else if (std::is_same<Type, simd_float3x3>::value) {
////            result.total_size = total_size * 9;
////            result.shape[DimsNew-1] = 3;
////            result.shape[DimsNew-2] = 3;
////            dimBias = 2;
////        } else if (std::is_same<Type, simd_float4x4>::value) {
////            result.total_size = total_size * 16;
////            result.shape[DimsNew-1] = 4;
////            result.shape[DimsNew-2] = 4;
////            dimBias = 2;
////        }
////        else {
////            result.total_size = total_size;
////        }
////
////        result.buffer = new float[total_size];
////        result.metalBuffer = [GlobalGPUManager.metalDevice
/// newBufferWithBytesNoCopy:result.buffer length:result.total_size *
/// sizeof(uint8_t) options:MTLResourceStorageModeShared deallocator:^(void *
///_Nonnull pointer, NSUInteger length) { /        }]; /
/// std::memcpy(result.shape + (DimsNew - dims) - dimBias, shape, sizeof(size_m)
///* dims); /        std::fill(result.shape, result.shape + (DimsNew - dims) -
/// dimBias, 1);
////
////        this->To<DimsNew, float>(result, 1);
////
////
////        return result;
////    }
////
////    template<int DimsNew>
////    explicit operator matrix<DimsNew, uint8_t>() const {
////        matrix<DimsNew, uint8_t> result;
////        result.total_size = total_size;
////        result.buffer = new uint8_t[total_size];
////        result.metalBuffer = [GlobalGPUManager.metalDevice
/// newBufferWithBytesNoCopy:result.buffer length:result.total_size *
/// sizeof(uint8_t) options:MTLResourceStorageModeShared deallocator:^(void *
///_Nonnull pointer, NSUInteger length) { /        }]; /
/// std::memcpy(result.shape, shape, sizeof(size_m) * dims); /
/// std::fill(result.shape + dims, result.shape + DimsNew, 1);
////
////        if (std::is_same<Type, uint8_t>::value) {
////            this->To<DimsNew, uint8_t>(result, 1);
////        }
////        else if (std::is_same<Type, int16_t>::value) {
////            this->To<DimsNew, uint8_t>(result, 3);
////        }
////        else {
////            std::cerr << "matrix: Type Not Suported Yet" << "\n";
////        }
////
////        return result;
////    }
//
//
//
//
//
//
//    matrix<dims, Type> Transpose(const std::initializer_list<size_m>& axis) {
//        if (axis.size() != dims) {
//            std::cerr << "matrix: Axis size should be equal to Dims of " <<
//            dims << "\n";
//        }
//
//        uint8_t typeCode = get_dtype_code<Type>();
//
//        if (!GlobalGPUManager.TransposeInit[typeCode]) {
//            GlobalGPUManager.initTransposeAll(typeCode);
//        }
//
//        matrix<dims, Type> output;
//        output.total_size = total_size;
//        output.buffer = new Type[total_size];
//        output.buildMetalBuffer();
//
//
//
//        size_m inputStrides[dims];
//        size_t acc = 1;
//        for (int i = dims-1; i >= 0; i--) {
//            inputStrides[i] = acc;
//            acc *= shape[i];
//        }
//
//        acc = 1;
//        size_m outputStrides[dims];
//        for (int i = dims-1; i >= 0; i--) {
//            outputStrides[*(axis.begin() + i)] = acc;
//            if (*(axis.begin() + i) >= dims) {std::cerr << "matrix: Rearranged
//            axis should not increase dims" << "\n"; } acc *=
//            shape[*(axis.begin() + i)]; output.shape[i] = shape[*(axis.begin()
//            + i)];
//        }
//
//        int dimensions = dims;
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
//
//        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBytes:inputStrides length:dims * sizeof(size_m)
//        atIndex:2]; [commandEncoder setBytes:outputStrides length:dims *
//        sizeof(size_m) atIndex:3]; [commandEncoder setBytes:&dimensions
//        length: sizeof(dims) atIndex:4]; [commandEncoder
//        setComputePipelineState:GlobalGPUManager.TransposeComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//
//        return output;
//    }
//
//
//
//    static matrix<dims, Type> sin(matrix<dims, Type>& mat) {
//
//
//        uint8_t typeCode = get_dtype_code<Type>();
//
//        if (!GlobalGPUManager.SinInit[typeCode]) {
//            GlobalGPUManager.initSin_All(typeCode);
//        }
//
//        matrix<dims, Type> output;
//        output.total_size = mat.total_size;
//        output.buffer = new Type[mat.total_size];
//        output.buildMetalBuffer();
//        memcpy(output.shape, mat.shape, sizeof(size_m) * dims);
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(mat.total_size, 1, 1);
//
//        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:mat.metalBuffer offset:0 atIndex:1];
//
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.SinComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//
//        return output;
//    }
//
//    matrix<dims-1, Type> Sum(int axis) {
//        if (axis < 0){
//            axis += dims;
//        }
//        if ((dims-1 < axis)) {
//            std::cerr << "matrix: Axis should not excede Dims of " << dims <<
//            "\n"; throw;
//        }
//
//        uint8_t typeCode = get_dtype_code<Type>();
//
//        if (!GlobalGPUManager.SumInit[typeCode]) {
//            GlobalGPUManager.initSum_All(typeCode);
//        }
//
//        matrix<dims-1, Type> output;
//        memcpy(output.shape, shape, axis * sizeof(size_m));
//        memcpy(output.shape + axis, shape + axis + 1, (dims-axis) *
//        sizeof(size_m));
//
//        output.total_size = output.accumul(0, dims-1);
//        output.buffer = new Type[output.total_size];
//        memset(output.buffer, 0, output.total_size * sizeof(Type));
//        output.buildMetalBuffer();
//        std::cout << output.total_size << "\n";
//
//        size_t ElStride = accumul(axis+1, dims);
//
//        size_t noOfOpp = shape[axis];
//        size_t axisStride;
//        if (axis != dims-1) {
//            axisStride = 1;
//        }
//        else {
//            axisStride = shape[axis];
//        }
//
//        size_t inputStrides[dims-1];
//        size_t acc = 1;
//        for (int i = dims-2; i >= 0; i--) {
//            inputStrides[i] = acc;
//            acc *= output.shape[i];
//        }
//
//        size_t maskedStrides[dims-1];
//        memcpy(maskedStrides, inputStrides, sizeof(size_m) * (dims-1));
//
//        acc = 1;
//        for (int i = 0; i < axis; i++) {
//            maskedStrides[i] *= shape[axis];
//        }
////        std::cout << "AxStride: " << axisStride << " ElStride: " << ElStride
///<< " noOfOpp: " << noOfOpp << "\n"; /        printArray(inputStrides,
/// dims-1); /        printArray(maskedStrides, dims-1);
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(output.total_size, 1, 1);
//
//        size_t outputDims = dims -1;
//        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBytes:&axisStride length: sizeof(size_m)
//        atIndex:2]; [commandEncoder setBytes:&ElStride length: sizeof(size_m)
//        atIndex:3]; [commandEncoder setBytes:&noOfOpp length: sizeof(size_m)
//        atIndex:4]; [commandEncoder setBytes:&inputStrides length: (dims-1) *
//        sizeof(size_m) atIndex:5]; [commandEncoder setBytes:&maskedStrides
//        length: (dims-1)* sizeof(size_m) atIndex:6]; [commandEncoder
//        setBytes:&outputDims length:  sizeof(size_m) atIndex:7];
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.SumComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//
//
//
//        return output;
//    }
//
//
//    matrix<dims, Type> T() const {
//        size_t axis[dims];
//
//
//        uint8_t typeCode = get_dtype_code<Type>();
//
//        if (!GlobalGPUManager.TransposeInit[typeCode]) {
//            GlobalGPUManager.initTransposeAll(typeCode);
//        }
//
//        matrix<dims, Type> output;
//        output.total_size = total_size;
//        output.buffer = new Type[total_size];
//        output.buildMetalBuffer();
//
//
//
//        size_t inputStrides[dims];
//        size_t acc = 1;
//        for (int i = dims-1; i >= 0; i--) {
//            inputStrides[i] = acc;
//            acc *= shape[i];
//            axis[dims-1-i]=i;
//        }
//
//
//        acc = 1;
//        size_t outputStrides[dims];
//        for (int i = dims-1; i >= 0; i--) {
//            outputStrides[axis[i]] = acc;
//            if (*(axis + i) >= dims) {std::cerr << "matrix: Rearranged axis
//            should not increase dims" << "\n"; } acc *= shape[axis[i]];
//            output.shape[i] = shape[axis[i]];
//        }
//
//        int dimensions = dims;
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
//
//        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBytes:inputStrides length:dims * sizeof(size_t)
//        atIndex:2]; [commandEncoder setBytes:outputStrides length:dims *
//        sizeof(size_t) atIndex:3]; [commandEncoder setBytes:&dimensions
//        length: sizeof(dims) atIndex:4]; [commandEncoder
//        setComputePipelineState:GlobalGPUManager.TransposeComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//
//
//
//        return output;
//    }
//
//
//    template<int kDims>
//    void conv(matrix<dims, Type>& result, const matrix<kDims, Type> &kernel,
//    int sepAxis = kDims, ConvMode mode = ConvMode::Same) {
//
//        uint8_t typeCode = get_dtype_code<Type>();
//
//        if (!GlobalGPUManager.ConvolveInit[typeCode]) {
//            GlobalGPUManager.initConvolve_All(typeCode);
//        }
//
//        size_m elementStride = accumul(sepAxis, dims);
//        size_m noOfOpp;
//        size_m newShape[dims];
//        switch (mode) {
//            case ConvMode::Same:
//                noOfOpp = accumul(0, sepAxis);
//                memcpy(newShape, shape, dims * sizeof(size_m));
//                break;
//            case ConvMode::Valid:
//                noOfOpp = 1;
//                for (size_m i = 0; i < sepAxis; ++i) {
//                    newShape[i] = shape[i] - (kernel.shape[i] - 1);
//                    noOfOpp *= newShape[i];
//                }
//                memcpy(newShape + sepAxis, shape + sepAxis, (dims - sepAxis) *
//                sizeof(size_m)); break;
//            case ConvMode::Full:
//                noOfOpp = 1;
//                for (size_m i = 0; i < sepAxis; ++i) {
//                    newShape[i] = shape[i] + (kernel.shape[i] - 1);
//                    noOfOpp *= newShape[i];
//                }
//                memcpy(newShape + sepAxis, shape + sepAxis, (dims - sepAxis) *
//                sizeof(size_m)); break;
//        }
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        MTLLogStateDescriptor *logStateDesc = [MTLLogStateDescriptor new];
//        logStateDesc.bufferSize = 1024*1024;
//        logStateDesc.level = MTLLogLevelInfo;
//        NSError* error = nil;
//        id<MTLLogState> logState = [GlobalGPUManager.metalDevice
//        newLogStateWithDescriptor:logStateDesc error:&error]; [logState
//        addLogHandler:^(NSString *substring, NSString *category,
//                                  MTLLogLevel level, NSString *message)
//        {
//        }];
//
//
//        MTLCommandBufferDescriptor *cbufDesc = [MTLCommandBufferDescriptor
//        new]; cbufDesc.logState = logState;
//
//        id<MTLCommandBuffer> commandBuffer = [commandQueue
//        commandBufferWithDescriptor:cbufDesc]; id<MTLComputeCommandEncoder>
//        commandEncoder = [commandBuffer computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(noOfOpp, 1, 1);
//
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBuffer:kernel.metalBuffer offset:0 atIndex:2];
//
//        [commandEncoder setBytes:&kernel.total_size length: sizeof(size_t)
//        atIndex:3]; [commandEncoder setBytes:shape length: kDims *
//        sizeof(size_m) atIndex:4]; [commandEncoder setBytes:kernel.shape
//        length:kDims * sizeof(size_m) atIndex:5]; if (sepAxis == dims) {
//            [commandEncoder setBytes:strides length:kDims * sizeof(size_m)
//            atIndex:6];
//        } else {
//            size_m stridesNew[kDims];
//            size_m acc = 1;
//            for (int i = kDims-1; i >= 0; i--) {
//                stridesNew[i] = acc;
//                acc *= shape[i];
//            }
//            [commandEncoder setBytes:stridesNew length:kDims * sizeof(size_m)
//            atIndex:6];
//        }
//        [commandEncoder setBytes:&elementStride length: sizeof(size_m)
//        atIndex:7]; [commandEncoder setBytes:&sepAxis length: sizeof(int)
//        atIndex:8];
//
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.ConvolveComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//
//    void Dot(matrix<dims, Type>& result, const matrix<dims, Type> &other, bool
//    TransposeB) {
//
//        uint8_t typeCode = get_dtype_code<Type>();
//
//        if (!GlobalGPUManager.GEMMAInit[typeCode]) {
//            GlobalGPUManager.initGEMMA_All(typeCode);
//        }
//
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
//
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//
//        if (!TransposeB) {
//            matrix<dims, Type> B_transposed = other.T();
//            [commandEncoder setBuffer:B_transposed.metalBuffer offset:0
//            atIndex:2]; [commandEncoder setBytes:other.shape length:dims *
//            sizeof(size_m) atIndex:4];
//
//        } else {
//            size_m* reverseShapeBuffer = new size_m[dims];
//            reverseBuffer(other.shape, reverseShapeBuffer, dims);
//            [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:2];
//            [commandEncoder setBytes:reverseShapeBuffer length:dims *
//            sizeof(size_m) atIndex:4]; delete [] reverseShapeBuffer;
//        }
//
//        [commandEncoder setBytes:shape length:dims * sizeof(size_m)
//        atIndex:3];
//
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.GEMMAComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//
//    matrix<dims, Type> Dot(const matrix<dims, Type> &other) {
//        if (shape[dims - 1] != other.shape[0]) {
//            std::cerr << "ValueError: shapes (" ;
//            printShape(false);
//            std::cerr << ") and (";
//            other.printShape(false);
//            std::cerr << ") not aligned: "<< shape[dims-1]<<" (dim "<< dims-1
//            <<") != "<< other.shape[0] <<" (dim 0) \n"; throw;
//        }
//
//        matrix<dims, Type> result = matrix<dims, Type>::withShape({shape[0],
//        other.shape[1]}); Dot(result, other, false); return result;
//    }
//
//    matrix<dims, Type> Dot(const matrix<dims, Type> &other, bool TransposeB) {
//        if (shape[dims - 1] != other.shape[0]) {
//            std::cerr << "ValueError: shapes (" ;
//            printShape(false);
//            std::cerr << ") and (";
//            other.printShape(false);
//            std::cerr << ") not aligned: "<< shape[dims-1]<<" (dim "<< dims-1
//            <<") != "<< other.shape[0] <<" (dim 0) \n"; throw;
//        }
//
//        matrix<dims, Type> result = matrix<dims, Type>::withShape({shape[0],
//        other.shape[1]}); Dot(result, other, TransposeB); return result;
//    }
//
//    template <int dimsB, int resultDims>
//    void BrodcastedAdd(matrix<resultDims, Type>& result, const matrix<dimsB,
//    Type> &other) const {
////        result.parentNodes.push_back(std::make_shared<matrix<dims,
/// Type>>(*this)); /
/// result.parentNodes.push_back(std::make_shared<matrix<dimsB, Type>>(other));
/// /        result.gradFunc = [](matrix<dims, Type>& selfs) { / auto p1 =
/// selfs.parentNodes[0]->gradFunc ?
/// selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) :
/// selfs.parentNodes[0]->ones(); /            auto p2 =
/// selfs.parentNodes[1]->gradFunc ?
/// selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :
/// selfs.parentNodes[1]->ones(); /            return p1 + p2; /        };
////
//        uint8_t typeCode = get_dtype_code<Type>();
//
//        if (!GlobalGPUManager.BrodcastedAddInit[typeCode]) {
//            GlobalGPUManager.initBrodcastedAddInit(typeCode);
//        }
//
//        if (resultDims != fmax(dims, dimsB)) {
//            std::invalid_argument("Incompatible dims of the result mat");
//            throw;
//        }
//
//        size_m* strideA = new size_m[resultDims];
//        size_m* strideB = new size_m[resultDims];
//
//        memcpy(strideA + (resultDims -  dims), strides, dims *
//        sizeof(size_m)); memcpy(strideB + (resultDims - dimsB), other.strides,
//        dimsB * sizeof(size_m));
//
//        for (int i = 0; i < resultDims; i++) {
//            // dims - i-1 < 0
//            if (dims < i+1) {
//                result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
//                strideA[resultDims-i-1] =0;
//            } else if (dimsB < i+1) {
//                result.shape[resultDims-i-1] = shape[dims-i-1];
//                strideB[resultDims-i-1] =0;
//            } else if (shape[dims-i-1] != other.shape[dimsB-i-1]) {
//                if (shape[dims-i-1] == 1) {
//                    result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
//                    strideA[resultDims-i-1] =0;
//                } else if (other.shape[dimsB-i-1] == 1) {
//                    result.shape[resultDims-i-1] = shape[dims-i-1];
//                    strideB[resultDims-i-1] =0;
//                } else {
//                    std::invalid_argument("Incompatible shapes for
//                    broadcasting");
//                }
//            } else {
//                result.shape[resultDims-i-1] = shape[dims-i-1];
//            }
//        }
//
//        result.calcStrides();
//        if (result.total_size != result.accumul(0, resultDims)) {
//            result.total_size = result.accumul(0, resultDims);
//            if (result.buffer) {delete [] result.buffer; }
//            result.buffer = new Type[result.total_size];
//            result.buildMetalBuffer();
//        }
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
//        int rDims = resultDims;
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:2];
//        [commandEncoder setBytes:result.strides length:resultDims *
//        sizeof(size_m) atIndex:3]; [commandEncoder setBytes:strideA
//        length:resultDims * sizeof(size_m) atIndex:4]; [commandEncoder
//        setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
//        [commandEncoder setBytes:&rDims length:sizeof(int) atIndex:6];
//
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.BrodcastedAddComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//
//    template <int dimsB, int resultDims>
//    void BrodcastedSub(matrix<resultDims, Type>& result, const matrix<dimsB,
//    Type> &other) const {
////        result.parentNodes.push_back(std::make_shared<matrix<dims,
/// Type>>(*this)); /
/// result.parentNodes.push_back(std::make_shared<matrix<dimsB, Type>>(other));
/// /        result.gradFunc = [](matrix<dims, Type>& selfs) { / auto p1 =
/// selfs.parentNodes[0]->gradFunc ?
/// selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) :
/// selfs.parentNodes[0]->ones(); /            auto p2 =
/// selfs.parentNodes[1]->gradFunc ?
/// selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :
/// selfs.parentNodes[1]->ones(); /            return p1 + p2; /        };
////
//        uint8_t typeCode = get_dtype_code<Type>();
//
//        if (!GlobalGPUManager.BrodcastedSubInit[typeCode]) {
//            GlobalGPUManager.initBrodcastedSubInit(typeCode);
//        }
//
//        if (resultDims != fmax(dims, dimsB)) {
//            std::invalid_argument("Incompatible dims of the result mat");
//            throw;
//        }
//
//        size_m* strideA = new size_m[resultDims];
//        size_m* strideB = new size_m[resultDims];
//
//        memcpy(strideA + (resultDims -  dims), strides, dims *
//        sizeof(size_m)); memcpy(strideB + (resultDims - dimsB), other.strides,
//        dimsB * sizeof(size_m));
//
//        for (int i = 0; i < resultDims; i++) {
//            // dims - i-1 < 0
//            if (dims < i+1) {
//                result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
//                strideA[resultDims-i-1] =0;
//            } else if (dimsB < i+1) {
//                result.shape[resultDims-i-1] = shape[dims-i-1];
//                strideB[resultDims-i-1] =0;
//            } else if (shape[dims-i-1] != other.shape[dimsB-i-1]) {
//                if (shape[dims-i-1] == 1) {
//                    result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
//                    strideA[resultDims-i-1] =0;
//                } else if (other.shape[dimsB-i-1] == 1) {
//                    result.shape[resultDims-i-1] = shape[dims-i-1];
//                    strideB[resultDims-i-1] =0;
//                } else {
//                    std::invalid_argument("Incompatible shapes for
//                    broadcasting");
//                }
//            } else {
//                result.shape[resultDims-i-1] = shape[dims-i-1];
//            }
//        }
//
//        result.calcStrides();
//        if (result.total_size != result.accumul(0, resultDims)) {
//            result.total_size = result.accumul(0, resultDims);
//            if (result.buffer) {delete [] result.buffer; }
//            result.buffer = new Type[result.total_size];
//            result.buildMetalBuffer();
//        }
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
//        int rDims = resultDims;
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:2];
//        [commandEncoder setBytes:result.strides length:resultDims *
//        sizeof(size_m) atIndex:3]; [commandEncoder setBytes:strideA
//        length:resultDims * sizeof(size_m) atIndex:4]; [commandEncoder
//        setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
//        [commandEncoder setBytes:&rDims length:sizeof(int) atIndex:6];
//
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.BrodcastedSubComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//
//    template <int dimsB, int resultDims>
//    void BrodcastedMul(matrix<resultDims, Type>& result, const matrix<dimsB,
//    Type> &other) const {
////        result.parentNodes.push_back(std::make_shared<matrix<dims,
/// Type>>(*this)); /
/// result.parentNodes.push_back(std::make_shared<matrix<dimsB, Type>>(other));
/// /        result.gradFunc = [](matrix<dims, Type>& selfs) { / auto p1 =
/// selfs.parentNodes[0]->gradFunc ?
/// selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) :
/// selfs.parentNodes[0]->ones(); /            auto p2 =
/// selfs.parentNodes[1]->gradFunc ?
/// selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :
/// selfs.parentNodes[1]->ones(); /            return p1 + p2; /        };
////
//        uint8_t typeCode = get_dtype_code<Type>();
//
//        if (!GlobalGPUManager.BrodcastedMulInit[typeCode]) {
//            GlobalGPUManager.initBrodcastedMulInit(typeCode);
//        }
//
//        if (resultDims != fmax(dims, dimsB)) {
//            std::invalid_argument("Incompatible dims of the result mat");
//            throw;
//        }
//
//        size_m* strideA = new size_m[resultDims];
//        size_m* strideB = new size_m[resultDims];
//
//        memcpy(strideA + (resultDims -  dims), strides, dims *
//        sizeof(size_m)); memcpy(strideB + (resultDims - dimsB), other.strides,
//        dimsB * sizeof(size_m));
//
//        for (int i = 0; i < resultDims; i++) {
//            // dims - i-1 < 0
//            if (dims < i+1) {
//                result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
//                strideA[resultDims-i-1] =0;
//            } else if (dimsB < i+1) {
//                result.shape[resultDims-i-1] = shape[dims-i-1];
//                strideB[resultDims-i-1] =0;
//            } else if (shape[dims-i-1] != other.shape[dimsB-i-1]) {
//                if (shape[dims-i-1] == 1) {
//                    result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
//                    strideA[resultDims-i-1] =0;
//                } else if (other.shape[dimsB-i-1] == 1) {
//                    result.shape[resultDims-i-1] = shape[dims-i-1];
//                    strideB[resultDims-i-1] =0;
//                } else {
//                    std::invalid_argument("Incompatible shapes for
//                    broadcasting");
//                }
//            } else {
//                result.shape[resultDims-i-1] = shape[dims-i-1];
//            }
//        }
//
//        result.calcStrides();
//        if (result.total_size != result.accumul(0, resultDims)) {
//            result.total_size = result.accumul(0, resultDims);
//            if (result.buffer) {delete [] result.buffer; }
//            result.buffer = new Type[result.total_size];
//            result.buildMetalBuffer();
//        }
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
//        int rDims = resultDims;
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:2];
//        [commandEncoder setBytes:result.strides length:resultDims *
//        sizeof(size_m) atIndex:3]; [commandEncoder setBytes:strideA
//        length:resultDims * sizeof(size_m) atIndex:4]; [commandEncoder
//        setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
//        [commandEncoder setBytes:&rDims length:sizeof(int) atIndex:6];
//
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.BrodcastedMulComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//
//    template <int dimsB, int resultDims>
//    void BrodcastedDiv(matrix<resultDims, Type>& result, const matrix<dimsB,
//    Type> &other) const {
////        result.parentNodes.push_back(std::make_shared<matrix<dims,
/// Type>>(*this)); /
/// result.parentNodes.push_back(std::make_shared<matrix<dimsB, Type>>(other));
/// /        result.gradFunc = [](matrix<dims, Type>& selfs) { / auto p1 =
/// selfs.parentNodes[0]->gradFunc ?
/// selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) :
/// selfs.parentNodes[0]->ones(); /            auto p2 =
/// selfs.parentNodes[1]->gradFunc ?
/// selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :
/// selfs.parentNodes[1]->ones(); /            return p1 + p2; /        };
////
//        uint8_t typeCode = get_dtype_code<Type>();
//
//        if (!GlobalGPUManager.BrodcastedDivInit[typeCode]) {
//            GlobalGPUManager.initBrodcastedDivInit(typeCode);
//        }
//
//        if (resultDims != fmax(dims, dimsB)) {
//            std::invalid_argument("Incompatible dims of the result mat");
//            throw;
//        }
//
//        size_m* strideA = new size_m[resultDims];
//        size_m* strideB = new size_m[resultDims];
//
//        memcpy(strideA + (resultDims -  dims), strides, dims *
//        sizeof(size_m)); memcpy(strideB + (resultDims - dimsB), other.strides,
//        dimsB * sizeof(size_m));
//
//        for (int i = 0; i < resultDims; i++) {
//            // dims - i-1 < 0
//            if (dims < i+1) {
//                result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
//                strideA[resultDims-i-1] =0;
//            } else if (dimsB < i+1) {
//                result.shape[resultDims-i-1] = shape[dims-i-1];
//                strideB[resultDims-i-1] =0;
//            } else if (shape[dims-i-1] != other.shape[dimsB-i-1]) {
//                if (shape[dims-i-1] == 1) {
//                    result.shape[resultDims-i-1] = other.shape[dimsB-i-1];
//                    strideA[resultDims-i-1] =0;
//                } else if (other.shape[dimsB-i-1] == 1) {
//                    result.shape[resultDims-i-1] = shape[dims-i-1];
//                    strideB[resultDims-i-1] =0;
//                } else {
//                    std::invalid_argument("Incompatible shapes for
//                    broadcasting");
//                }
//            } else {
//                result.shape[resultDims-i-1] = shape[dims-i-1];
//            }
//        }
//
//        result.calcStrides();
//        if (result.total_size != result.accumul(0, resultDims)) {
//            result.total_size = result.accumul(0, resultDims);
//            if (result.buffer) {delete [] result.buffer; }
//            result.buffer = new Type[result.total_size];
//            result.buildMetalBuffer();
//        }
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
//        int rDims = resultDims;
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:2];
//        [commandEncoder setBytes:result.strides length:resultDims *
//        sizeof(size_m) atIndex:3]; [commandEncoder setBytes:strideA
//        length:resultDims * sizeof(size_m) atIndex:4]; [commandEncoder
//        setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
//        [commandEncoder setBytes:&rDims length:sizeof(int) atIndex:6];
//
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.BrodcastedDivComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//
//    void Add(matrix<dims, Type>& result, const matrix<dims, Type> &other) {
//
//        result.parentNodes.push_back(std::make_shared<matrix<dims,
//        Type>>(*this));
//        result.parentNodes.push_back(std::make_shared<matrix<dims,
//        Type>>(other)); result.gradFunc = [](matrix<dims, Type>& selfs) {
//            auto p1 = selfs.parentNodes[0]->gradFunc ?
//            selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) :
//            selfs.parentNodes[0]->ones(); auto p2 =
//            selfs.parentNodes[1]->gradFunc ?
//            selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :
//            selfs.parentNodes[1]->ones(); return p1 + p2;
//        };
//
//        id<MTLComputePipelineState> computeState;
//        if constexpr (std::is_integral<Type>::value) {
//            if (!GlobalGPUManager.AddIntInit) {
//                GlobalGPUManager.initAddInt();
//            }
//            computeState = GlobalGPUManager.AddIntCompute;
//        } else if constexpr (std::is_floating_point<Type>::value) {
//            if (!GlobalGPUManager.AddFloatInit) {
//                GlobalGPUManager.initAddFloat();
//            }
//            computeState = GlobalGPUManager.AddFloatCompute;
//        } else {
//            std::cerr << "matrix: Type not supported" << "\n";
//        }
//
//
//
////        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice
/// newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type)
/// options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer,
/// NSUInteger length) { /        }]; /        id<MTLBuffer> buffer2 =
///[GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:other.buffer
/// length:other.total_size*sizeof(Type) options:MTLResourceStorageModeShared
/// deallocator:^(void * _Nonnull pointer, NSUInteger length) { /        }]; /
/// id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice
/// newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type)
/// options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer,
/// NSUInteger length) { /        }];
//
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
//
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:2];
//        [commandEncoder setComputePipelineState:computeState];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//
//    template<int dimsB>
//    matrix<dims, Type> operator+(const matrix<dimsB, Type> &other) requires
//    (dims > dimsB) {
//        matrix<dims, Type> result;
//        BrodcastedAdd(result, other);
//        return result;
//    }
//
//    template<int dimsB>
//    matrix<dimsB, Type> operator+(const matrix<dimsB, Type> &other) requires
//    (dims < dimsB) {
//        matrix<dimsB, Type> result;
//        BrodcastedAdd(result, other);
//        return result;
//    }
//
//    matrix<dims, Type> operator+(const Type value) {
//        matrix<dims, Type> result;
//        auto other = matrix<0, Type>(value);
//        BrodcastedAdd(result, other);
//        return result;
//    }
//    friend matrix<dims, Type> operator+(Type value, const matrix<dims, Type>&
//    mat) {
//        matrix<dims, Type> result;
//        auto other = matrix<0, Type>(value);
//        mat.BrodcastedAdd(result, other);
//        return result;
//    }
//
//    matrix<dims, Type> operator+(const matrix<dims, Type> &other) {
//        matrix<dims, Type> result;
//        if (total_size == other.total_size) {
//            result.buffer = new Type[total_size];
//            result.total_size = total_size;
//            result.metalBuffer = [GlobalGPUManager.metalDevice
//            newBufferWithBytesNoCopy:result.buffer length:total_size *
//            sizeof(Type) options:MTLResourceStorageModeShared
//            deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//            }];
//            memcpy(result.shape, shape, sizeof(size_m) * dims);
//            Add(result, other);
//        } else {
//            BrodcastedAdd(result, other);
//        }
//
////        id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
////
////        id<MTLBuffer> buffer1 = [metalDevice newBufferWithBytesNoCopy:buffer
/// length:total_size*sizeof(Type) options:MTLResourceStorageModeShared
/// deallocator:^(void * _Nonnull pointer, NSUInteger length) { /        }]; /
/// id<MTLBuffer> buffer2 = [metalDevice newBufferWithBytesNoCopy:other.buffer
/// length:other.total_size*sizeof(Type) options:MTLResourceStorageModeShared
/// deallocator:^(void * _Nonnull pointer, NSUInteger length) { /        }]; /
/// id<MTLBuffer> buffer3 = [metalDevice newBufferWithBytesNoCopy:result.buffer
/// length:total_size*sizeof(Type) options:MTLResourceStorageModeShared
/// deallocator:^(void * _Nonnull pointer, NSUInteger length) { /        }];
////
////        id<MTLLibrary> lib = [metalDevice newDefaultLibrary];
////        id<MTLFunction> func1;
////
////        if constexpr (std::is_integral<Type>::value) {
////            func1 = [lib newFunctionWithName:@"AddGPU_I"];
////        } else if constexpr (std::is_floating_point<Type>::value) {
////            func1 = [lib newFunctionWithName:@"AddGPU_F"];
////        } else {
////            func1 = [lib newFunctionWithName:@"AddGPU_C"];
////        }
////
////
////        NSError *error = nil;
////        id<MTLComputePipelineState> computeState = [metalDevice
/// newComputePipelineStateWithFunction:func1 error:&error];
////
////        if (error) {
////            NSLog(@"Adder: %@", error.localizedDescription);
////        }
////
////        id<MTLCommandQueue> commandQueue = [metalDevice newCommandQueue];
////        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
////        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
/// computeCommandEncoder];
////
////        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
////        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
////
////        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
////        [commandEncoder setBuffer:buffer2 offset:0 atIndex:1];
////        [commandEncoder setBuffer:buffer3 offset:0 atIndex:2];
////        [commandEncoder setComputePipelineState:computeState];
////        [commandEncoder dispatchThreads:_dispatchExecutionSize
////                  threadsPerThreadgroup:_threadsPerThreadgroup];
////
////        [commandEncoder endEncoding];
////        [commandBuffer commit];
////        [commandBuffer waitUntilCompleted];
//
//        return result;
//    }
//
//    template<int dimsB>
//    matrix<dims, Type> operator*(const matrix<dimsB, Type> &other) requires
//    (dims > dimsB) {
//        matrix<dims, Type> result;
//        BrodcastedMul(result, other);
//        return result;
//    }
//
//    template<int dimsB>
//    matrix<dimsB, Type> operator*(const matrix<dimsB, Type> &other) requires
//    (dims < dimsB) {
//        matrix<dimsB, Type> result;
//        BrodcastedMul(result, other);
//        return result;
//    }
//
//    matrix<dims, Type> operator*(const Type value) {
//        matrix<dims, Type> result;
//        auto other = matrix<0, Type>(value);
//        BrodcastedMul(result, other);
//        return result;
//    }
//    friend matrix<dims, Type> operator*(Type value, const matrix<dims, Type>&
//    mat) {
//        matrix<dims, Type> result;
//        auto other = matrix<0, Type>(value);
//        mat.BrodcastedMul(result, other);
//        return result;
//    }
//
//
//    matrix<dims, Type> operator*(const matrix<dims, Type>& other) {
//        if (total_size == other.total_size) {
//            return MulMat(other);
//        } else {
//            matrix<dims, Type> result;
//            BrodcastedMul(result, other);
//            return result;
//        }
//    }
//
//
//    template <size_t D = dims, typename = std::enable_if_t<(D > 1)>>
//    matrix<dims-1, Type> operator[] (int i) {
//        if (i < 0) {
//            i = shape[0] + i;
//        }
//        if (i >= shape[0]) {
//            throw std::invalid_argument( "Index Out Of range" );
//        }
//        matrix<dims-1, Type> result;
//        result.total_size = accumul(1, dims);
//        std::memcpy(result.shape, shape + 1, sizeof(size_m) * (dims-1));
//        result.buffer = buffer + result.total_size * i;
//        flags |= (1u << 0);      // sets bit 0 to
//        result.metalBuffer = [GlobalGPUManager.metalDevice
//        newBufferWithBytesNoCopy:result.buffer length:result.total_size *
//        sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void *
//        _Nonnull pointer, NSUInteger length) {
//        }];
//        return result;
//    }
//
//    template <size_t D = dims, typename = std::enable_if_t<(D == 1)>>
//    Type& operator[] (int i) const {
//    #ifdef SAFE_MODE
//        if (i < 0) {
//            i = shape[0] + i;
//        }
//        if (i >= shape[0]) {
//            throw std::invalid_argument( "Index Out Of range" );
//        }
//    #endif
//        return buffer[i];
//    }
//
//    template <size_t D = dims, typename = std::enable_if_t<(D == 2)>>
//    Type& operator[] (int i, int j) const {
//    #ifdef SAFE_MODE
//        if (i < 0) {
//            i = shape[0] + i;
//        }
//        if (i >= shape[0]) {
//            throw std::invalid_argument( "Index Out Of range" );
//        }
//    #endif
//        return buffer[shape[1] * i + j];
//    }
//
//    template <size_t D = dims, typename = std::enable_if_t<(D == 4)>>
//    Type& operator[] (int i, int j, int k, int l) const {
//    #ifdef SAFE_MODE
//        if (i < 0) {
//            i = shape[0] + i;
//        }
//        if (i >= shape[0]) {
//            throw std::invalid_argument( "Index Out Of range" );
//        }
//
//        if (j < 0) {
//            j = shape[1] + j;
//        }
//        if (j >= shape[1]) {
//            throw std::invalid_argument( "Index Out Of range" );
//        }
//
//        if (k < 0) {
//            k = shape[2] + k;
//        }
//        if (k >= shape[2]) {
//            throw std::invalid_argument( "Index Out Of range" );
//        }
//    #endif
//        return buffer[strides[0] * i + strides[1] * j + strides[2] * k +
//        strides[3] * l];
//    }
//
//    template <size_t D = dims, typename = std::enable_if_t<(D > 2)>>
//    matrix<dims-2, Type> operator[] (int i, int j) const {
//    #ifdef SAFE_MODE
//        if (i < 0) {
//            i = shape[0] + i;
//        }
//        if (i >= shape[0]) {
//            throw std::invalid_argument( "Index Out Of range" );
//        }
//
//        if (j < 0) {
//            j = shape[1] + j;
//        }
//        if (j >= shape[1]) {
//            throw std::invalid_argument( "Index Out Of range" );
//        }
//    #endif
//        matrix<dims-2, Type> slicedMat;
//        slicedMat.buffer = buffer + strides[0] * i + strides[1] * j;
//        memcpy(slicedMat.strides, strides + 2, (dims-2) *sizeof(size_m));
//        memcpy(slicedMat.shape, shape + 2, (dims-2) *sizeof(size_m));
//        slicedMat.total_size = strides[1];
//        slicedMat.buildMetalBuffer();
//        return slicedMat;
//    }
//
//    template <size_t D = dims, typename = std::enable_if_t<(D > 3)>>
//    matrix<dims-3, Type> operator[] (int i, int j, int k) const {
//    #ifdef SAFE_MODE
//        if (i < 0) {
//            i = shape[0] + i;
//        }
//        if (i >= shape[0]) {
//            throw std::invalid_argument( "Index Out Of range" );
//        }
//
//        if (j < 0) {
//            j = shape[1] + j;
//        }
//        if (j >= shape[1]) {
//            throw std::invalid_argument( "Index Out Of range" );
//        }
//
//        if (k < 0) {
//            k = shape[2] + k;
//        }
//        if (k >= shape[2]) {
//            throw std::invalid_argument( "Index Out Of range" );
//        }
//    #endif
//        matrix<dims-3, Type> slicedMat;
//
//        slicedMat.buffer = buffer + strides[0] * i + strides[1] * j +
//        strides[2] * k; memcpy(slicedMat.strides, strides + 3, (dims-3)
//        *sizeof(size_m)); memcpy(slicedMat.shape, shape + 3, (dims-3)
//        *sizeof(size_m)); slicedMat.total_size = strides[2];
//        slicedMat.buildMetalBuffer();
//        return slicedMat;
//    }
//
//
//
//    ~matrix() {
////        std::cout << "Matrix Destroyed" << "\n";
//        if (flags & 0) {
//            delete [] buffer;
//        }
//
//    }
//
//    matrix(const matrix<dims, Type>& other) {
//        std::cout << "Copied" << "\n";
//        // copy constructor doesnt need to delete its buffer as  its called
//        only on uninitlised matricies
////        if () {
//            buffer = new Type[other.total_size];
//            total_size = other.total_size;
//            metalBuffer = [GlobalGPUManager.metalDevice
//            newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type)
//            options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull
//            pointer, NSUInteger length) {
//            }];
//        gradFunc = other.gradFunc;
//        parentNodes = other.parentNodes;
////        } else if (total_size != other.total_size) {
////            // copy constructor doesnt need to delete it
//////            if (buffer) {
//////                delete [] buffer;
//////            }
////            buffer = new Type[other.total_size];
////            total_size = other.total_size;
////
////        }
//
//        memcpy(buffer, other.buffer, sizeof(Type) * total_size);
//        memcpy(shape, other.shape, sizeof(size_m) * dims);
//        memcpy(strides, other.strides, dims * sizeof(size_m));
//    }
//
//    matrix( matrix<dims, Type>&& other) {
//        std::cout << "Moved" << "\n";
//        if (buffer) {
//            delete [] buffer;
//        }
//        buffer = other.buffer;
//        other.buffer = nullptr;
//        memcpy(shape, other.shape, dims * sizeof(size_m));
//        memcpy(strides, other.strides, dims * sizeof(size_m));
//        metalBuffer = other.metalBuffer;
//        total_size = other.total_size;
//        gradFunc = std::move(other.gradFunc);
//        parentNodes = std::move(other.parentNodes);
//        other.~matrix();
//    }
//
//    // const fill
//    matrix<dims, Type>& operator=(Type value) {
//        if (flags & (1u << 1)) {
//            fill_nd_iterative(buffer, shape, strides, dims, value);
//        } else {
//            std::fill(buffer, buffer+total_size, value);
////            memset(buffer, 0, total_size * sizeof(Type));
//        }
//
//        return *this;
//    }
//
//    // copy assignment
//    matrix<dims, Type>& operator=(const matrix<dims, Type>& other) {
//
//        if (&other == this) { }
//        else if (total_size == other.total_size) {
//            std::cout << "Copy Assignment" << "\n";
//            memcpy(buffer, other.buffer, total_size * sizeof(Type));
//            memcpy(shape, other.shape, dims * sizeof(size_m));
//            memcpy(strides, other.strides, dims * sizeof(size_m));
//            gradFunc = other.gradFunc;
//            parentNodes = other.parentNodes;
//            other.print();
//
//            print();
//
//        } else {
//            std::cout << "Copy Create Assignment" << "\n";
//            if (buffer) {
//                delete [] buffer;
//            }
//            total_size = other.total_size;
//            buffer = new Type[total_size];
//            buildMetalBuffer();
//            memcpy(buffer, other.buffer, total_size * sizeof(Type));
//            memcpy(shape, other.shape, dims * sizeof(size_m));
//            memcpy(strides, other.strides, dims * sizeof(size_m));
//            gradFunc = other.gradFunc;
//            parentNodes = other.parentNodes;
//        }
//
//        return *this;
//    }
//
//    matrix<dims, Type>& operator=(matrix<dims, Type>&& other) {
//        if (&other == this) { }
////        else if (total_size == other.total_size) {
////            std::cout << "Copy Assignment" << "\n";
////            memcpy(buffer, other.buffer, total_size * sizeof(Type));
////            memcpy(shape, other.shape, dims * sizeof(size_m));
////        } else {
//        std::cout << "Move Assignment" << "\n";
//        if (buffer && (flags & 0)) {
//            delete [] buffer;
//        }
//        buffer = other.buffer;
//        metalBuffer = other.metalBuffer;
//        flags = other.flags;
//        other.buffer = nullptr;
//        memcpy(shape, other.shape, dims * sizeof(size_m));
//        memcpy(strides, other.strides, dims * sizeof(size_m));
//        total_size = other.total_size;
//        gradFunc = std::move(other.gradFunc);
//        parentNodes = std::move(other.parentNodes);
//        other.~matrix();
//        return *this;
//    }
//
////    template <int d>
////    matrix<d, Type>& operator=(matrix<d, Type>&& other) {
////
////        this->~matrix();
////        return *other;
////    }
//
//    matrix<dims, Type> Derivative(matrix<dims, Type>& result, int axis, int
//    loopBack) {
//        size_t stride = accumul(axis+1, dims);
//        size_t max = shape[axis] - 1;
//        int lastResolve = loopBack;
//        if (!result.buffer) {
//            result.buffer = new Type[total_size];
//            result.total_size = total_size;
//            result.buildMetalBuffer();
//        }
//        else if (result.total_size != total_size) {
//            delete [] result.buffer;
//            result.buffer = new Type[total_size];
//            result.total_size = total_size;
//            result.buildMetalBuffer();
//        }
//
//        if (!compareShapes(result.shape)) {
//            memcpy(result.shape, shape, sizeof(size_m) * dims);
//        }
//
//
//        if (!GlobalGPUManager.DerivativeAllInit) {
//            GlobalGPUManager.initDerivativeAll();
//        }
//
//        int type = 0;
//
//        // for treating simd_float2 as 2 floats
//        int typeBias = 1;
//
//        if constexpr (std::is_integral<Type>::value) {
//            if (std::is_unsigned<Type>::value) {
//                type = 2;
//            } else {
//                type = 1;
//            }
//
//        } else if constexpr (std::is_floating_point<Type>::value) {
//            type = 0;
//        } else if constexpr (std::is_same<Type, simd_float2>::value) {
//            type = 0;
//            stride *= 2;
//            typeBias *= 2;
//        }
//        else {
//            std::cerr << "matrix: Type Not supported" << "\n";
//        }
//
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer
//        computeCommandEncoder];
//
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(typeBias*total_size, 1, 1);
//
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBytes:&stride length:sizeof(size_t) atIndex:2];
//        [commandEncoder setBytes:&max length:sizeof(size_t) atIndex:3];
//        [commandEncoder setBytes:&lastResolve length:sizeof(int) atIndex:4];
//
//
//
//        [commandEncoder setBytes:&type length:sizeof(int) atIndex:5];
//        [commandEncoder
//        setComputePipelineState:GlobalGPUManager.DerivativeAll];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//
//        return result;
//    }
//
//    void SliceBuffer(Type* outBuff, Type* inBuff, size_t stride, size_t size,
//    uint rows) {
//        for (int i = 0; i < rows; i++) {
//            memcpy(outBuff + i * size, inBuff + i * stride, size *
//            sizeof(Type));
//        }
//    }
//
//
//
//    void SliceCopy(matrix<dims, Type>& slicedMat,
//    std::initializer_list<std::optional<std::pair<size_t, size_t>>> slice) {
//        uint index = 0;
//        size_t offsets[dims];
//        bool firstNonNullStrideFound = false;
//        memset(offsets, 0, dims * sizeof(size_m));
//        memcpy(slicedMat.shape, shape, dims * sizeof(size_m));
//        Type* MemArena;
//
//        for (auto i : slice) {
//            if (i.has_value()) {
//                offsets[index] = i->first;
//                size_t stride = slicedMat.accumul(index, dims);
//                slicedMat.shape[index] = i->second - i->first;
//                size_t size = slicedMat.accumul(index, dims);
//                uint noOfOp = slicedMat.accumul(0, index);
//                if (!firstNonNullStrideFound) {
//                    MemArena = new Type[slicedMat.accumul(index, dims)];
//                    SliceBuffer(MemArena, buffer + offsets[index] * stride,
//                    stride, size, noOfOp); firstNonNullStrideFound = true;
//                    index++;
//                    continue;
//
//                }
//                SliceBuffer(MemArena, MemArena + offsets[index] * stride,
//                stride, size, noOfOp);
//            }
//            index++;
//        }
//        slicedMat.calcStrides();
////        slicedMat.total_size = slicedMat.accumul(0, dims);
//        slicedMat.total_size = slicedMat.strides[0] * slicedMat.shape[0];
//        memcpy(slicedMat.buffer, MemArena, sizeof(Type) *
//        slicedMat.total_size);
//    }
//
//    matrix<dims, Type>
//    SliceCopy(std::initializer_list<std::optional<std::pair<size_t, size_t>>>
//    slice) {
//        uint index = 0;
//        size_t offsets[dims];
//        bool firstNonNullStrideFound = false;
//
//        size_t acc = 1;
//        for (auto i : slice) {
//            if (i.has_value()) {
//                acc *= i->second - i->first;
//            }
//            else {
//                acc *= shape[index];
//            }
//            index ++;
//        }
//
//        acc *= accumul(index, dims);
//        matrix<dims, Type> slicedMat(acc);
//
//        index = 0;
//        memset(offsets, 0, dims * sizeof(size_m));
//        memcpy(slicedMat.shape, shape, dims * sizeof(size_m));
//        Type* MemArena;
//
//        for (auto i : slice) {
//            if (i.has_value()) {
//                offsets[index] = i->first;
//                size_t stride = slicedMat.accumul(index, dims);
//                slicedMat.shape[index] = i->second - i->first;
//                size_t size = slicedMat.accumul(index, dims);
//                uint noOfOp = slicedMat.accumul(0, index);
//                std::cout << "Stride: " << stride << "size: " << size << "no
//                of Opp: " << noOfOp << "\n"; if (!firstNonNullStrideFound) {
//                    MemArena = new Type[slicedMat.accumul(index, dims)];
//                    SliceBuffer(MemArena, buffer + offsets[index] * stride,
//                    stride, size, noOfOp); firstNonNullStrideFound = true;
//                    index++;
//                    continue;
//
//                }
//                SliceBuffer(MemArena, MemArena + offsets[index] * stride,
//                stride, size, noOfOp);
//            }
//            index++;
//        }
//        slicedMat.calcStrides();
//        slicedMat.total_size = acc;
//        memcpy(slicedMat.buffer, MemArena, sizeof(Type) *
//        slicedMat.total_size); return slicedMat;
//    }
//
//    matrix<dims, Type>
//    Slice(std::initializer_list<std::optional<std::pair<size_t, size_t>>>
//    slice) {
//        uint index = 0;
//        size_m offsets[dims];
//        bool firstNonNullStrideFound = false;
//        memset(offsets, 0, dims * sizeof(size_m));
//
//        matrix<dims, Type> slicedMat;
//        for (auto i : slice) {
//            if (i.has_value()) {
//                slicedMat.shape[index] = i->second - i->first;
//                offsets[index] = i->first;
//            }
//            else {
//                slicedMat.shape[index] *= shape[index];
//            }
//            index ++;
//        }
//
//        memcpy(slicedMat.strides, strides, dims * sizeof(size_m));
//        slicedMat.total_size = slicedMat.accumul(0, dims);
//        slicedMat.buffer = buffer + dotArray(offsets, strides, dims);
//        return slicedMat;
//    }
//
////    void operator=(const matrix<dims, Type> &other) {
////        if (buffer && total_size == other.total_size) {
////            memcpy(buffer, other.buffer, other.total_size * sizeof(Type));
////            memcpy(shape, other.shape, dims * sizeof(size_m));
//////            delete [] other.buffer;
////
////        } else {
////            if (buffer) {
////                delete [] buffer;
////            }
////            buffer = new Type[other.total_size];
////            total_size = other.total_size;
////            memcpy(buffer, other.buffer, other.total_size * sizeof(Type));
////            memcpy(shape, other.shape, dims * sizeof(size_m));
////        }
////    }
//
//};

matrix matrix::conv(const matrix& input, const matrix& kernel, const std::vector<int>& padding, const std::vector<int>& stride, const std::vector<int>& dilation, int groups) {
    // generic ND convolution
    // input shape [batch, S1, S2, ..., SN, channels]
    // kernel shape [out_channel, K1, K2, ..., KN, in_channel]
    // padding, stride, dilation lengths are N
    // N = input.dims - 2
    
    matrix output(input.dims, input.type);
    output.shape()[0] = input.shape()[0];
    
    int N = input.dims - 2;
    for (int i = 0; i < N; i++) {
        output.shape()[i + 1] = (input.shape()[i + 1] + 2 * padding[i] - dilation[i] * (kernel.shape()[i + 1] - 1) - 1) / stride[i] + 1;
    }
    
    output.shape()[input.dims - 1] = kernel.shape()[0];
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);
    output.tape = new ConvolvePrimitive(input, kernel, padding, stride, dilation, groups);
    return output;
}

void matrix::conv_gpu(const matrix& kernel, matrix& output) {
    id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
    id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
    
    int typeCode = (int)type;
    if (!GlobalGPUManager.ConvInit[typeCode]) {
        GlobalGPUManager.initConv(typeCode);
    }
    
    [commandEncoder setComputePipelineState:GlobalGPUManager.ConvComputeState[typeCode]];
    
    ConvolvePrimitive* prim = (ConvolvePrimitive*)output.tape;
    int groups = prim->groups;
    
    // N = output.dims - 2
    int N = output.dims - 2;
    int num_spatial_dims = N;
    
    // Calculate total spatial dimensions for thread grid
    size_t batch_and_remaining_spatial = output.shape()[0];
    for (int i = 1; i < N; i++) {
        batch_and_remaining_spatial *= output.shape()[i];
    }
    
    // Grid: [Channels, Last Spatial Dim, Batch * Remaining Spatial Dims]
    MTLSize gridSize = MTLSizeMake(output.shape()[output.dims - 1], output.shape()[N], batch_and_remaining_spatial);
    
    NSUInteger threadGroupSize = GlobalGPUManager.ConvComputeState[typeCode].maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > gridSize.width) {
        threadGroupSize = gridSize.width;
    }
    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    
    [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
    [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
    [commandEncoder setBuffer:kernel.metalBuffer offset:0 atIndex:2];
    
    [commandEncoder setBytes:shape() length:dims * sizeof(size_m) atIndex:3];
    [commandEncoder setBytes:kernel.shape() length:kernel.dims * sizeof(size_m) atIndex:4];
    [commandEncoder setBytes:output.shape() length:output.dims * sizeof(size_m) atIndex:5];
    
    [commandEncoder setBytes:prim->padding.data() length:prim->padding.size() * sizeof(int) atIndex:6];
    [commandEncoder setBytes:prim->stride.data() length:prim->stride.size() * sizeof(int) atIndex:7];
    [commandEncoder setBytes:prim->dilation.data() length:prim->dilation.size() * sizeof(int) atIndex:8];
    [commandEncoder setBytes:&groups length:sizeof(int) atIndex:9];
    
    [commandEncoder setBytes:strides() length:dims * sizeof(size_m) atIndex:10];
    [commandEncoder setBytes:kernel.strides() length:kernel.dims * sizeof(size_m) atIndex:11];
    [commandEncoder setBytes:output.strides() length:output.dims * sizeof(size_m) atIndex:12];
    [commandEncoder setBytes:&num_spatial_dims length:sizeof(int) atIndex:13];
    
    int kernel_dot_totalsize = 1;
    for (int i = 1; i < kernel.dims; i++) {
        kernel_dot_totalsize *= kernel.shape()[i];
    }
    [commandEncoder setBytes:&kernel_dot_totalsize length:sizeof(int) atIndex:14];
    
    [commandEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

template matrix &matrix::operator=<float>(float);
//template matrix &matrix::operator=<float16_t>(float16_t);
template matrix &matrix::operator=<uint8_t>(uint8_t);
template matrix &matrix::operator=<int>(int);
template matrix &matrix::operator=<int16_t>(int16_t);
template matrix &matrix::operator=<uint32_t>(uint32_t);
template matrix &matrix::operator=<uint16_t>(uint16_t);

// MARK: Max / Min Boilerplate

matrix matrix::max(const matrix& a, const matrix& b) {
    dtype res_type = promote_types(a.type, b.type);
    matrix result(std::max(a.dims, b.dims), res_type);
    auto primit = new MaxPrimitive(a, b);
    primit->desc_a = BroadcastDescriptor::create(result.dims);
    primit->desc_b = BroadcastDescriptor::create(result.dims);
    broadcast_shapes(a.array_desc, b.array_desc, result.array_desc,
                     primit->desc_a, primit->desc_b, a.dims, b.dims);
    primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_b->strides(result.dims), result.strides(), result.dims, INT32_MAX);
    primit->dims_collapsed = true;
    result.total_size = result.accumul(0, result.dims);
    result.tape = primit;
    return result;
}

void matrix::max(const matrix& other, matrix& result, ExecutionDevice exec_device) const {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    const_cast<matrix*>(this)->update_from_trace();
    const_cast<matrix*>(&other)->update_from_trace();
    
    if (result.dims != std::max(dims, other.dims)) {
        throw std::invalid_argument("Incompatible dims of the result mat");
    }
    
    int cdims = ((MaxPrimitive *)result.tape)->collapsed_dims_3.out_dims;
    size_m* strideA = ((MaxPrimitive *)result.tape)->collapsed_dims_3.stridesA;
    size_m* strideB = ((MaxPrimitive *)result.tape)->collapsed_dims_3.stridesB;
    size_m* strideR = ((MaxPrimitive *)result.tape)->collapsed_dims_3.stridesC;
    size_m* result_shape = ((MaxPrimitive *)result.tape)->collapsed_dims_3.shape;
    
    if (exec_device == ExecutionDevice::METAL) {
        int typeCode = (int)type;
        int kernelCode = -1;
        if (cdims == 0) { kernelCode = 0; }
        else if (cdims == 1) { kernelCode = 0; }
        else if (cdims == 2) { kernelCode = 1; }
        else if (cdims == 3) { kernelCode = 2; }
        else { kernelCode = 3; }
        
        if (!GlobalGPUManager.BrodcastedMaxInit[typeCode][kernelCode]) {
            GlobalGPUManager.initBrodcastedMaxInit(typeCode, kernelCode);
        }
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedMaxComputeState[typeCode][kernelCode]];
        
        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
        setBufferOrBytes(commandEncoder, *this, 1);
        setBufferOrBytes(commandEncoder, other, 2);
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        
        if (cdims == 0) {
            size_m one = 1;
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        } else if (cdims == 1) {
            [commandEncoder setBytes:strideR length:sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(result_shape[0], 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setBytes:strideR length:2 * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:2 * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:2 * sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(result_shape[1], result_shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setBytes:strideR length:3 * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:3 * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:3 * sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(result_shape[2], result_shape[1], result_shape[0]);
        } else {
            [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:cdims * sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:result_shape length:cdims * sizeof(size_m) atIndex:6];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:7];
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= result_shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(result_shape[cdims-1], result_shape[cdims-2], acc);
        }
        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        if (result.dims == 0) {
            dispatch_type(type, result.buffer, [&](auto *out_data) {
                using T = std::decay_t<decltype(*out_data)>;
                out_data[0] = std::max(static_cast<T*>(buffer)[0], static_cast<T*>(other.buffer)[0]);
            });
            return;
        }
        dispatch_type(type, result.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
            for (int gid = 0; gid < result.total_size; gid++) {
                size_t GindexA = 0;
                size_t GindexB = 0;
                size_t indexR = 0;
                int rem = gid;
                for (int i = 0; i < cdims; i++) {
                    indexR = rem / strideR[i];
                    GindexA += indexR * strideA[i];
                    GindexB += indexR * strideB[i];
                    rem %= strideR[i];
                }
                out_data[gid] = std::max(static_cast<T *>(buffer)[GindexA], static_cast<T *>(other.buffer)[GindexB]);
            }
        });
    }
}

matrix matrix::min(const matrix& a, const matrix& b) {
    dtype res_type = promote_types(a.type, b.type);
    matrix result(std::max(a.dims, b.dims), res_type);
    auto primit = new MinPrimitive(a, b);
    primit->desc_a = BroadcastDescriptor::create(result.dims);
    primit->desc_b = BroadcastDescriptor::create(result.dims);
    broadcast_shapes(a.array_desc, b.array_desc, result.array_desc,
                     primit->desc_a, primit->desc_b, a.dims, b.dims);
    primit->collapsed_dims_3 = collapse_dims(primit->desc_a->shape(), primit->desc_a->strides(result.dims), primit->desc_b->strides(result.dims), result.strides(), result.dims, INT32_MAX);
    primit->dims_collapsed = true;
    result.total_size = result.accumul(0, result.dims);
    result.tape = primit;
    return result;
}

void matrix::min(const matrix& other, matrix& result, ExecutionDevice exec_device) const {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    const_cast<matrix*>(this)->update_from_trace();
    const_cast<matrix*>(&other)->update_from_trace();
    
    if (result.dims != std::max(dims, other.dims)) {
        throw std::invalid_argument("Incompatible dims of the result mat");
    }
    
    int cdims = ((MinPrimitive *)result.tape)->collapsed_dims_3.out_dims;
    size_m* strideA = ((MinPrimitive *)result.tape)->collapsed_dims_3.stridesA;
    size_m* strideB = ((MinPrimitive *)result.tape)->collapsed_dims_3.stridesB;
    size_m* strideR = ((MinPrimitive *)result.tape)->collapsed_dims_3.stridesC;
    size_m* result_shape = ((MinPrimitive *)result.tape)->collapsed_dims_3.shape;
    
    if (exec_device == ExecutionDevice::METAL) {
        int typeCode = (int)type;
        int kernelCode = -1;
        if (cdims == 0) { kernelCode = 0; }
        else if (cdims == 1) { kernelCode = 0; }
        else if (cdims == 2) { kernelCode = 1; }
        else if (cdims == 3) { kernelCode = 2; }
        else { kernelCode = 3; }
        
        if (!GlobalGPUManager.BrodcastedMinInit[typeCode][kernelCode]) {
            GlobalGPUManager.initBrodcastedMinInit(typeCode, kernelCode);
        }
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedMinComputeState[typeCode][kernelCode]];
        
        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
        setBufferOrBytes(commandEncoder, *this, 1);
        setBufferOrBytes(commandEncoder, other, 2);
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        
        if (cdims == 0) {
            size_m one = 1;
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&one length:sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        } else if (cdims == 1) {
            [commandEncoder setBytes:strideR length:sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(result_shape[0], 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setBytes:strideR length:2 * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:2 * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:2 * sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(result_shape[1], result_shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setBytes:strideR length:3 * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:3 * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:3 * sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(result_shape[2], result_shape[1], result_shape[0]);
        } else {
            [commandEncoder setBytes:strideR length:cdims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:strideA length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:strideB length:cdims * sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:result_shape length:cdims * sizeof(size_m) atIndex:6];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:7];
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= result_shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(result_shape[cdims-1], result_shape[cdims-2], acc);
        }
        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        if (result.dims == 0) {
            dispatch_type(type, result.buffer, [&](auto *out_data) {
                using T = std::decay_t<decltype(*out_data)>;
                out_data[0] = std::min(static_cast<T*>(buffer)[0], static_cast<T*>(other.buffer)[0]);
            });
            return;
        }
        dispatch_type(type, result.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
            for (int gid = 0; gid < result.total_size; gid++) {
                size_t GindexA = 0;
                size_t GindexB = 0;
                size_t indexR = 0;
                int rem = gid;
                for (int i = 0; i < cdims; i++) {
                    indexR = rem / strideR[i];
                    GindexA += indexR * strideA[i];
                    GindexB += indexR * strideB[i];
                    rem %= strideR[i];
                }
                out_data[gid] = std::min(static_cast<T *>(buffer)[GindexA], static_cast<T *>(other.buffer)[GindexB]);
            }
        });
    }
}


matrix matrix::max(int axis, bool keepdims) const {
    if (dims == 0) {
        return *this;
    }
    int outputDims = keepdims ? dims : dims - 1;
    matrix output(outputDims, type);
    
    if (axis < 0) axis += dims;
    
    if (keepdims) {
        memcpy(output.shape(), shape(), output.dims * sizeof(size_m));
        output.shape()[axis] = 1;
    } else {
        memcpy(output.shape(), shape(), axis * sizeof(size_m));
        memcpy(output.shape() + axis, shape() + axis + 1, (output.dims - axis) * sizeof(size_m));
    }
    
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);
    
    MaxReductionPrimitive* prim = new MaxReductionPrimitive(const_cast<matrix&>(*this), axis, keepdims);
    prim->collapsed_dims = collapse_dims_reduce(shape(), output.strides(), strides(), dims, axis, INT32_MAX, keepdims);
    prim->has_collapsed_dims = true;
    output.tape = prim;
    
    return output;
}

void matrix::max(matrix& output, int axis, bool keepdims, ExecutionDevice exec_device) {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    size_m reduce_axis_stride = (size_m)accumul(axis+1, dims);
    size_m noOfOpp = shape()[axis];
    
    MaxReductionPrimitive* primit = static_cast<MaxReductionPrimitive*>(output.tape);
    
    CollapsedDims_2 collapsed;
    uint32_t cdims;
    if (primit->has_collapsed_dims) {
        collapsed = primit->collapsed_dims;
        cdims = collapsed.out_dims;
    } else {
        cdims = 1;
        collapsed.stridesA[0] = 1;
        collapsed.stridesB[0] = 1;
        collapsed.shape[0] = 1;
    }
    
    if (cdims == 0) {
        cdims = 1;
        collapsed.stridesA[0] = 1;
        collapsed.stridesB[0] = 1;
        collapsed.shape[0] = 1;
    }

    if (exec_device == ExecutionDevice::METAL) {
        bool use_tgr = (noOfOpp > 256);
        int kernel_idx = cdims >= 4 ? 3 : cdims - 1;
        if (use_tgr) {
            kernel_idx += 4; // TGR kernels are at offset +4
        }
        
        if (!GlobalGPUManager.MaxInit[(int)type][kernel_idx]) {
            GlobalGPUManager.initMax_nd((int)type, kernel_idx);
        }
        
        id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();

        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        setBufferOrBytes(commandEncoder, *this, 1);
        
        [commandEncoder setBytes:&reduce_axis_stride length: sizeof(size_m) atIndex:2];
        [commandEncoder setBytes:&noOfOpp length: sizeof(size_m) atIndex:3];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        
        int x_multiplier = use_tgr ? 256 : 1;
        if (use_tgr) {
            _threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
        }

        if (cdims == 1) {
            [commandEncoder setBytes:&collapsed.stridesA[0] length:sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&collapsed.stridesB[0] length:sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(collapsed.shape[0] * x_multiplier, 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setBytes:collapsed.stridesA length:2 * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:collapsed.stridesB length:2 * sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(collapsed.shape[1] * x_multiplier, collapsed.shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setBytes:collapsed.stridesA length:3 * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:collapsed.stridesB length:3 * sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(collapsed.shape[2] * x_multiplier, collapsed.shape[1], collapsed.shape[0]);
        } else {
            [commandEncoder setBytes:collapsed.stridesA length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:collapsed.stridesB length:cdims * sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:collapsed.shape length:cdims * sizeof(size_m) atIndex:6];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:7];
            
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= collapsed.shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(collapsed.shape[cdims-1] * x_multiplier, collapsed.shape[cdims-2], acc);
        }

        [commandEncoder setComputePipelineState:GlobalGPUManager.MaxComputeState_nd[(int)type][kernel_idx]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        dispatch_type(type, output.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
            T* in_data = (T*)buffer;
            size_t total_out = output.total_size;
            
            for (size_t gid = 0; gid < total_out; gid++) {
                size_t AxisOffset = 0;
                size_t remaining = gid;
                for (size_t i = 0; i < cdims; i++) {
                    AxisOffset += (remaining / collapsed.stridesA[i]) * collapsed.stridesB[i];
                    remaining %= collapsed.stridesA[i];
                }
                T current_max = std::numeric_limits<T>::lowest();
                for (size_t i = 0; i < noOfOpp; i++) {
                    current_max = std::max(current_max, in_data[AxisOffset + i * reduce_axis_stride]);
                }
                out_data[gid] = current_max;
            }
        });
    }
}

matrix matrix::max(int start, int end, bool keepdims) const {
    matrix flat = this->flatten(start, end);
    return flat.max(start, keepdims);
}

matrix matrix::max() const{
    matrix flat = this->flatten(0, -1);
    return flat.max(0);
}

matrix matrix::min(int axis, bool keepdims) const{
    if (dims == 0) {
        return *this;
    }
    
    int outputDims = keepdims ? dims : dims - 1;
    matrix output(outputDims, type);
    
    if (axis < 0) axis += dims;
    
    if (keepdims) {
        memcpy(output.shape(), shape(), output.dims * sizeof(size_m));
        output.shape()[axis] = 1;
    } else {
        memcpy(output.shape(), shape(), axis * sizeof(size_m));
        memcpy(output.shape() + axis, shape() + axis + 1, (output.dims - axis) * sizeof(size_m));
    }
    
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);

    MinReductionPrimitive* prim = new MinReductionPrimitive(const_cast<matrix&>(*this), axis, keepdims);
    prim->collapsed_dims = collapse_dims_reduce(shape(), output.strides(), strides(), dims, axis, INT32_MAX, keepdims);
    prim->has_collapsed_dims = true;
    output.tape = prim;
    
    return output;
}

void matrix::min(matrix& output, int axis, bool keepdims, ExecutionDevice exec_device) {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    size_m reduce_axis_stride = (size_m)accumul(axis+1, dims);
    size_m noOfOpp = shape()[axis];
    
    MinReductionPrimitive* primit = static_cast<MinReductionPrimitive*>(output.tape);
    
    CollapsedDims_2 collapsed;
    uint32_t cdims;
    if (primit->has_collapsed_dims) {
        collapsed = primit->collapsed_dims;
        cdims = collapsed.out_dims;
    } else {
        cdims = 1;
        collapsed.stridesA[0] = 1;
        collapsed.stridesB[0] = 1;
        collapsed.shape[0] = 1;
    }
    
    if (cdims == 0) {
        cdims = 1;
        collapsed.stridesA[0] = 1;
        collapsed.stridesB[0] = 1;
        collapsed.shape[0] = 1;
    }

    if (exec_device == ExecutionDevice::METAL) {
        bool use_tgr = (noOfOpp > 256);
        int kernel_idx = cdims >= 4 ? 3 : cdims - 1;
        if (use_tgr) {
            kernel_idx += 4; // TGR kernels are at offset +4
        }
        
        if (!GlobalGPUManager.MinInit[(int)type][kernel_idx]) {
            GlobalGPUManager.initMin_nd((int)type, kernel_idx);
        }

        id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();

        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        setBufferOrBytes(commandEncoder, *this, 1);
        
        [commandEncoder setBytes:&reduce_axis_stride length: sizeof(size_m) atIndex:2];
        [commandEncoder setBytes:&noOfOpp length: sizeof(size_m) atIndex:3];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        
        int x_multiplier = use_tgr ? 256 : 1;
        if (use_tgr) {
            _threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
        }

        if (cdims == 1) {
            [commandEncoder setBytes:&collapsed.stridesA[0] length:sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&collapsed.stridesB[0] length:sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(collapsed.shape[0] * x_multiplier, 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setBytes:collapsed.stridesA length:2 * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:collapsed.stridesB length:2 * sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(collapsed.shape[1] * x_multiplier, collapsed.shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setBytes:collapsed.stridesA length:3 * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:collapsed.stridesB length:3 * sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(collapsed.shape[2] * x_multiplier, collapsed.shape[1], collapsed.shape[0]);
        } else {
            [commandEncoder setBytes:collapsed.stridesA length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:collapsed.stridesB length:cdims * sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:collapsed.shape length:cdims * sizeof(size_m) atIndex:6];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:7];
            
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= collapsed.shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(collapsed.shape[cdims-1] * x_multiplier, collapsed.shape[cdims-2], acc);
        }

        [commandEncoder setComputePipelineState:GlobalGPUManager.MinComputeState_nd[(int)type][kernel_idx]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        dispatch_type(type, output.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
            T* in_data = (T*)buffer;
            size_t total_out = output.total_size;
            
            for (size_t gid = 0; gid < total_out; gid++) {
                size_t AxisOffset = 0;
                size_t remaining = gid;
                for (size_t i = 0; i < cdims; i++) {
                    AxisOffset += (remaining / collapsed.stridesA[i]) * collapsed.stridesB[i];
                    remaining %= collapsed.stridesA[i];
                }
                T current_min = std::numeric_limits<T>::max();
                for (size_t i = 0; i < noOfOpp; i++) {
                    current_min = std::min(current_min, in_data[AxisOffset + i * reduce_axis_stride]);
                }
                out_data[gid] = current_min;
            }
        });
    }
}

matrix matrix::min(int start, int end, bool keepdims) const {
    matrix flat = this->flatten(start, end);
    return flat.min(start, keepdims);
}

matrix matrix::min() const {
    matrix flat = this->flatten(0, -1);
    return flat.min(0);
}

matrix matrix::sum(int axis, bool keepdims) const {
    if (dims == 0) {
        return *this;
    }
    int outputDims = keepdims ? dims : dims - 1;
    matrix output(outputDims, type);
    
    if (axis < 0) axis += dims;
    
    if (keepdims) {
        memcpy(output.shape(), shape(), output.dims * sizeof(size_m));
        output.shape()[axis] = 1;
    } else {
        memcpy(output.shape(), shape(), axis * sizeof(size_m));
        memcpy(output.shape() + axis, shape() + axis + 1, (output.dims - axis) * sizeof(size_m));
    }
    
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);
    
    SumPrimitive* prim = new SumPrimitive(const_cast<matrix&>(*this), axis, keepdims);
    prim->collapsed_dims = collapse_dims_reduce(shape(), output.strides(), strides(), dims, axis, INT32_MAX, keepdims);
    prim->has_collapsed_dims = true;
    output.tape = prim;
    
    return output;
}

void matrix::sum(matrix& output, int axis, bool keepdims, ExecutionDevice exec_device) {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    size_m reduce_axis_stride = (size_m)accumul(axis+1, dims);
    size_m noOfOpp = shape()[axis];
    
    SumPrimitive* primit = static_cast<SumPrimitive*>(output.tape);
    
    CollapsedDims_2 collapsed;
    uint32_t cdims;
    if (primit->has_collapsed_dims) {
        collapsed = primit->collapsed_dims;
        cdims = collapsed.out_dims;
    } else {
        cdims = 1;
        collapsed.stridesA[0] = 1;
        collapsed.stridesB[0] = 1;
        collapsed.shape[0] = 1;
    }
    
    if (cdims == 0) {
        cdims = 1;
        collapsed.stridesA[0] = 1;
        collapsed.stridesB[0] = 1;
        collapsed.shape[0] = 1;
    }

    if (exec_device == ExecutionDevice::METAL) {
        bool use_tgr = (noOfOpp > 256);
        int kernel_idx = cdims >= 4 ? 3 : cdims - 1;
        if (use_tgr) {
            kernel_idx += 4; // TGR kernels are at offset +4
        }
        
        if (!GlobalGPUManager.SumInit_nd[(int)type][kernel_idx]) {
            GlobalGPUManager.initSum_nd((int)type, kernel_idx);
        }

        id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();

        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        setBufferOrBytes(commandEncoder, *this, 1);
        
        [commandEncoder setBytes:&reduce_axis_stride length: sizeof(size_m) atIndex:2];
        [commandEncoder setBytes:&noOfOpp length: sizeof(size_m) atIndex:3];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);
        
        int x_multiplier = use_tgr ? 256 : 1;
        if (use_tgr) {
            _threadsPerThreadgroup = MTLSizeMake(256, 1, 1);
        }

        if (cdims == 1) {
            [commandEncoder setBytes:&collapsed.stridesA[0] length:sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&collapsed.stridesB[0] length:sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(collapsed.shape[0] * x_multiplier, 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setBytes:collapsed.stridesA length:2 * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:collapsed.stridesB length:2 * sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(collapsed.shape[1] * x_multiplier, collapsed.shape[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setBytes:collapsed.stridesA length:3 * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:collapsed.stridesB length:3 * sizeof(size_m) atIndex:5];
            _dispatchExecutionSize = MTLSizeMake(collapsed.shape[2] * x_multiplier, collapsed.shape[1], collapsed.shape[0]);
        } else {
            [commandEncoder setBytes:collapsed.stridesA length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:collapsed.stridesB length:cdims * sizeof(size_m) atIndex:5];
            [commandEncoder setBytes:collapsed.shape length:cdims * sizeof(size_m) atIndex:6];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:7];
            
            size_m acc = 1;
            for (int i = 0; i < cdims-2; i++) { acc *= collapsed.shape[i]; }
            _dispatchExecutionSize = MTLSizeMake(collapsed.shape[cdims-1] * x_multiplier, collapsed.shape[cdims-2], acc);
        }

        [commandEncoder setComputePipelineState:GlobalGPUManager.SumComputeState_nd[(int)type][kernel_idx]];
        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        dispatch_type(type, output.buffer, [&](auto *out_data) {
            using T = std::decay_t<decltype(*out_data)>;
            T* in_data = (T*)buffer;
            size_t total_out = output.total_size;
            
            for (size_t gid = 0; gid < total_out; gid++) {
                size_t AxisOffset = 0;
                size_t remaining = gid;
                for (size_t i = 0; i < cdims; i++) {
                    AxisOffset += (remaining / collapsed.stridesA[i]) * collapsed.stridesB[i];
                    remaining %= collapsed.stridesA[i];
                }
                T current_sum = 0;
                for (size_t i = 0; i < noOfOpp; i++) {
                    current_sum += in_data[AxisOffset + i * reduce_axis_stride];
                }
                out_data[gid] = current_sum;
            }
        });
    }
}

matrix matrix::sum(int start, int end, bool keepdims) const {
    matrix flat = this->flatten(start, end);
    return flat.sum(start, keepdims);
}

matrix matrix::sum() const {
    matrix flat = this->flatten(0, -1);
    return flat.sum(0);
}

matrix matrix::mean(int axis, bool keepdims) const {
    matrix s = this->sum(axis, keepdims);
    return s / (float)shape()[axis];
}

matrix matrix::mean(int start, int end, bool keepdims) const {
    matrix s = this->sum(start, end, keepdims);
    int real_end = (end < 0) ? end + dims : end;
    size_m collapse_size = 1;
    for (int i = start; i <= real_end; i++) {
        collapse_size *= shape()[i];
    }
    return s / (float)collapse_size;
}

matrix matrix::mean() const {
    matrix s = this->sum();
    return s / (float)total_size;
}

matrix matrix::rms(int axis, bool keepdims) const {
    matrix sq = (*this) * (*this);
    matrix m = sq.mean(axis, keepdims);
    return matrix::sqrt(m);
}

matrix matrix::rms(int start, int end, bool keepdims) const {
    matrix sq = (*this) * (*this);
    matrix m = sq.mean(start, end, keepdims);
    return matrix::sqrt(m);
}

matrix matrix::rms() const {
    matrix sq = (*this) * (*this);
    matrix m = sq.mean();
    return matrix::sqrt(m);
}

//matrix matrix::operator[](int index) {
//    if (dims == 0) throw std::runtime_error("Cannot slice a 0-dimensional matrix");
//    if (index >= shape()[0]) throw std::runtime_error("Index out of bounds");
//    if (index < 0) index + dims;
//    
//    matrix output(dims-1, type);
//    
//    output.flags = flags;
//    output.flags |= NON_OWNERSHIP_FLAG;
//    
//    
//    if (dims > 1) {
//        memcpy(output.shape(),   shape() + 1, (dims - 1) * sizeof(size_m));
//        memcpy(output.strides(), strides() + 1, (dims - 1) * sizeof(size_m));
//    }
//    output.total_size = output.accumul(0, output.dims);
//    output.tape = new SlicePrimitive()
//    return output;
//}

matrix matrix::take(const matrix& index, int axis) const {
    matrix output(dims - 1 + index.dims, type);
    // replaces the input axis shape with index shape =>
    // inp shape [s0, s1, ... axis ..., s_n]
    // index shape [i0, i1 .. i_m]
    // output shape [[s0, s1, ... (i0, i1 .. i_m) ..., s_n]]
    memcpy(output.shape(), shape(), axis * sizeof(size_m));
    memcpy(output.shape() + axis, index.shape(), index.dims * sizeof(size_m));
    memcpy(output.shape() + axis + index.dims, shape() + axis + 1, (dims - axis - 1) * sizeof(size_m));
    
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);
    output.tape = new TakePrimitive(const_cast<matrix&>(*this), const_cast<matrix&>(index), axis);
    return output;
}

void matrix::take_backend(const matrix& index, matrix& output, int axis, ExecutionDevice exec_device) const {
    if (exec_device == ExecutionDevice::AUTO) {
        exec_device = output.total_size > 10 ? ExecutionDevice::METAL : ExecutionDevice::CPU;
    }
    
    uint32_t cdims = output.dims; // Dimension should be based on output
    
    size_m eff_src_strides[cdims];
    size_m eff_idx_strides[cdims];
    if (cdims != 0) {
        memset(eff_src_strides, 0, cdims * sizeof(size_m));
        memset(eff_idx_strides, 0, cdims * sizeof(size_m));
        
        memcpy(eff_src_strides, strides(), axis * sizeof(size_m));
        memcpy(eff_idx_strides + axis, index.strides(), index.dims * sizeof(size_m));
        memcpy(eff_src_strides + index.dims + axis, strides() + axis + 1, (dims-axis-1) * sizeof(size_m));
    }
    if (exec_device == ExecutionDevice::METAL) {
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        int typeCode = (int)type;
        int kernel_code = cdims > 3 ? 3 : (cdims == 0 ? 0 : cdims - 1);

        if (!GlobalGPUManager.TakeInit_nd[typeCode][kernel_code]) {
            GlobalGPUManager.initTake_nd(typeCode, kernel_code);
        }
        [commandEncoder setComputePipelineState:GlobalGPUManager.TakeComputeState_nd[typeCode][kernel_code]];

        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0]; // dst
        setBufferOrBytes(commandEncoder, *this, 1); // src
        setBufferOrBytes(commandEncoder, index, 2); // indices
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(1, 1, 1);

        if (cdims == 0 || cdims == 1) {
            size_m out_stride = cdims == 0 ? 1 : output.strides()[0];
            size_m src_stride = cdims == 0 ? 0 : eff_src_strides[0];
            size_m idx_stride = cdims == 0 ? 0 : eff_idx_strides[0];
            [commandEncoder setBytes:&out_stride length:sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:&src_stride length:sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:&idx_stride length:sizeof(size_m) atIndex:5];
            
            size_m src_axis_stride = strides()[axis];
            [commandEncoder setBytes:&src_axis_stride length:sizeof(size_m) atIndex:6];
            
            _dispatchExecutionSize = MTLSizeMake(cdims == 0 ? 1 : output.shape()[0], 1, 1);
        } else if (cdims == 2) {
            [commandEncoder setBytes:output.strides() length:2 * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:eff_src_strides length:2 * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:eff_idx_strides length:2 * sizeof(size_m) atIndex:5];
            
            size_m src_axis_stride = strides()[axis];
            [commandEncoder setBytes:&src_axis_stride length:sizeof(size_m) atIndex:6];
            
            _dispatchExecutionSize = MTLSizeMake(output.shape()[1], output.shape()[0], 1);
        } else if (cdims == 3) {
            [commandEncoder setBytes:output.strides() length:3 * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:eff_src_strides length:3 * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:eff_idx_strides length:3 * sizeof(size_m) atIndex:5];
            
            size_m src_axis_stride = strides()[axis];
            [commandEncoder setBytes:&src_axis_stride length:sizeof(size_m) atIndex:6];
            
            _dispatchExecutionSize = MTLSizeMake(output.shape()[2], output.shape()[1], output.shape()[0]);
        } else {
            [commandEncoder setBytes:output.strides() length:cdims * sizeof(size_m) atIndex:3];
            [commandEncoder setBytes:eff_src_strides length:cdims * sizeof(size_m) atIndex:4];
            [commandEncoder setBytes:eff_idx_strides length:cdims * sizeof(size_m) atIndex:5];
            
            size_m src_axis_stride = strides()[axis];
            [commandEncoder setBytes:&src_axis_stride length:sizeof(size_m) atIndex:6];
            
            [commandEncoder setBytes:output.shape() length:cdims * sizeof(size_m) atIndex:7];
            int ndim = cdims;
            [commandEncoder setBytes:&ndim length:sizeof(int) atIndex:8];
            
            size_m total_threads = output.total_size;
            size_m threads_y = output.shape()[cdims-2];
            size_m threads_x = output.shape()[cdims-1];
            size_m threads_z = total_threads / (threads_x * threads_y);
            _dispatchExecutionSize = MTLSizeMake(threads_x, threads_y, threads_z);
        }
        
        // We don't need to pass axis to the GPU anymore since we resolved it on CPU!
        
        [commandEncoder dispatchThreads:_dispatchExecutionSize threadsPerThreadgroup:_threadsPerThreadgroup];
    } else {
        // CPU Execution Fallback
        int typeCode = (int)type;
        if (typeCode == (int)dtype::Float) {
            float* dst_ptr = (float*)output.buffer;
            float* src_ptr = (float*)buffer;
            int* idx_ptr = (int*)index.buffer;
            
            size_m src_axis_stride = strides()[axis];
            
            for (size_t i = 0; i < output.total_size; ++i) {
                size_m remA = i;
                size_m dst_idx = 0;
                size_m idx_idx = 0;
                size_m src_idx = 0;
                for (int d = cdims - 1; d >= 0; --d) {
                    size_m mod = remA % output.shape()[d];
                    dst_idx += mod * output.strides()[d];
                    idx_idx += mod * eff_idx_strides[d];
                    src_idx += mod * eff_src_strides[d];
                    remA /= output.shape()[d];
                }
                int idx_val = idx_ptr[idx_idx];
                src_idx += idx_val * src_axis_stride;
                dst_ptr[dst_idx] = src_ptr[src_idx];
            }
        }
    }
}
