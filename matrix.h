//
// Created by Aditya Dudeja on 06/06/26.
//

#ifndef WORLDOF3D_MATRIX_H
#define WORLDOF3D_MATRIX_H

#import <iostream>
@import Utils;
@import GPUManager;
#include <CoreGraphics/CoreGraphics.h>
#include <ImageIO/ImageIO.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <arm_fp16.h>
#include <arm_neon.h>
#include <atomic>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <vector>
#include <concepts>
#include <functional>
#if TARGET_OS_IPHONE
#import <UIKit/UIKit.h>
#endif

using R = AxisRange;
using r = Range;
enum class dtype: uint8_t {
    Float = 0,
    Float16 = 1,
    UInt8 = 2,
    Int32 = 3,
    Int16 = 4,
    UInt32 = 5,
    UInt16 = 6
    // Add more as needed
};

template <dtype code> struct type_from_dtype;

template <> struct type_from_dtype<dtype::Float> {
    using type = float;
};
template <> struct type_from_dtype<dtype::Float16> {
    using type = float16_t;
};
template <> struct type_from_dtype<dtype::UInt8> {
    using type = uint8_t;
};
template <> struct type_from_dtype<dtype::Int32> {
    using type = int;
};
template <> struct type_from_dtype<dtype::Int16> {
    using type = int16_t;
};
template <> struct type_from_dtype<dtype::UInt32> {
    using type = uint32_t;
};
template <> struct type_from_dtype<dtype::UInt16> {
    using type = uint16_t;
};

template <typename T> dtype dtype_from_type();

template <> constexpr inline dtype dtype_from_type<float>() {
    return dtype::Float;
}
template <> constexpr inline dtype dtype_from_type<float16_t>() {
    return dtype::Float16;
}
template <> constexpr inline dtype dtype_from_type<uint8_t>() {
    return dtype::UInt8;
}
template <> constexpr inline dtype dtype_from_type<int>() {
    return dtype::Int32;
}
template <> constexpr inline dtype dtype_from_type<int16_t>() {
    return dtype::Int16;
}
template <> constexpr inline dtype dtype_from_type<uint32_t>() {
    return dtype::UInt32;
}
template <> constexpr inline dtype dtype_from_type<uint16_t>() {
    return dtype::UInt16;
}

constexpr size_t dtype_size(dtype d) {
    static constexpr size_t sizes[] = {
        sizeof(float),     // 0: Float
        sizeof(float16_t), // 1: Float16
        sizeof(uint8_t),   // 2: UInt8
        sizeof(int32_t),   // 3: Int32
        sizeof(int16_t),   // 4: Int16
        sizeof(uint32_t),  // 5: UInt32
        sizeof(uint16_t)   // 6: UInt16
    };
    return sizes[static_cast<size_t>(d)];
}
//                    Float  Float16  UInt8  Int32  Int16  UInt32  UInt16
constexpr dtype type_rules[7][7] = {
/*Float  */ { dtype::Float,  dtype::Float,   dtype::Float,  dtype::Float,   dtype::Float,  dtype::Float,   dtype::Float  },
/*Float16*/ { dtype::Float,  dtype::Float16, dtype::Float16,dtype::Float16, dtype::Float16,dtype::Float16, dtype::Float16},
/*UInt8  */ { dtype::Float,  dtype::Float16, dtype::UInt8,  dtype::Int32,   dtype::Int16,  dtype::UInt32,  dtype::UInt16 },
/*Int32  */ { dtype::Float,  dtype::Float16, dtype::Int32,  dtype::Int32,   dtype::Int32,  dtype::Int32,   dtype::Int32  },
/*Int16  */ { dtype::Float,  dtype::Float16, dtype::Int16,  dtype::Int32,   dtype::Int16,  dtype::Int32,   dtype::Int32  },
/*UInt32 */ { dtype::Float,  dtype::Float16, dtype::UInt32, dtype::Int32,   dtype::Int32,  dtype::UInt32,  dtype::UInt32 },
/*UInt16 */ { dtype::Float,  dtype::Float16, dtype::UInt16, dtype::Int32,   dtype::Int32,  dtype::UInt32,  dtype::UInt16 },
};

dtype promote_types(dtype a, dtype b);

struct SharedArrayDescriptor {
    std::atomic<uint32_t> refCount;
    inline size_m* shape() {
        return reinterpret_cast<size_m *>(this + 1);
    }
    inline size_m* strides(int dims) {
        return shape() + dims;
    }
    inline SharedArrayDescriptor* retain() {
        refCount.fetch_add(1, std::memory_order_relaxed);
        return this;
    }
    static SharedArrayDescriptor *create(uint32_t dims) {
        size_t total_bytes =
        sizeof(SharedArrayDescriptor) + (dims * 2 * sizeof(size_m));
        void *mem = ::operator new(total_bytes);
        SharedArrayDescriptor *shared = new (mem) SharedArrayDescriptor();
        shared->refCount.store(1, std::memory_order_relaxed);
        return shared;
    }
    void release();
};

//struct BroadcastDescriptor {
//  inline size_m *shape();
//  inline size_m *strides(int dims);
//
//  static BroadcastDescriptor *create(int dims) {
//    size_t bytes = sizeof(BroadcastDescriptor) + 2 * dims * sizeof(size_m);
//    void *mem = ::operator new(bytes);
//    return new (mem) BroadcastDescriptor();
//  }
//
//  static void destroy(BroadcastDescriptor *p) {
//    p->~BroadcastDescriptor();
//    ::operator delete(p);
//  }
//};

struct alignas(size_m) BroadcastDescriptor {
    inline size_m *shape();
    inline size_m *strides(int dims);
    static BroadcastDescriptor *create(int dims) {
        size_t bytes = sizeof(BroadcastDescriptor) + 2 * dims * sizeof(size_m);
        void *mem = ::operator new(bytes, std::align_val_t{alignof(BroadcastDescriptor)});
        return new (mem) BroadcastDescriptor();
    }
    
    static void destroy(BroadcastDescriptor *p) {
        p->~BroadcastDescriptor();
        ::operator delete(p, std::align_val_t{alignof(BroadcastDescriptor)});
    }
};

static constexpr int SBO_MAX_DIMS = 3;
union array_descriptor {
    size_m inline_buffer[SBO_MAX_DIMS * 2]; // For dims <= 3
    SharedArrayDescriptor *shared_arr_desc; // For dims > 3
    void retain(int dims) {
        if (dims > SBO_MAX_DIMS)
            shared_arr_desc->retain();
    }

    void release(int dims) {
        if (dims > SBO_MAX_DIMS)
            shared_arr_desc->release();
    }
};

void broadcast_shapes(const array_descriptor &arr_desc1, const array_descriptor &arr_desc2, array_descriptor &out_shape, BroadcastDescriptor *new_desc1, BroadcastDescriptor *new_desc2, int dim1, int dim2);
void broadcast_shapes_matmul(const array_descriptor &arr_desc1,
                             const array_descriptor &arr_desc2,
                             array_descriptor &out_shape,
                             BroadcastDescriptor *new_desc1,
                             BroadcastDescriptor *new_desc2, int dim1, int dim2);
template <typename Func>
inline void dispatch_type(dtype type, void *buffer, Func &&function_to_run) {
    if (!buffer)
        return;
    
    switch (type) {
        case dtype::Float:
            function_to_run(static_cast<float *>(buffer));
            break;
        case dtype::Float16:
            function_to_run(static_cast<uint16_t *>(buffer));
            break;
        case dtype::Int32:
            function_to_run(static_cast<int32_t *>(buffer));
            break;
        case dtype::UInt32:
            function_to_run(static_cast<uint32_t *>(buffer));
            break;
        case dtype::UInt8:
            function_to_run(static_cast<uint8_t *>(buffer));
            break;
        case dtype::Int16:
            function_to_run(static_cast<int16_t *>(buffer));
            break;
        case dtype::UInt16:
            function_to_run(static_cast<uint16_t *>(buffer));
            break;
        default:
            throw std::runtime_error("Unsupported dtype during dispatch");
    }
}

// Detect if a type is an initializer_list
template <typename T> struct is_init_list : std::false_type {};
template <typename T>
struct is_init_list<std::initializer_list<T>> : std::true_type {};

// Count the depth of the nested initializer lists (Compile-time)
template <typename T> struct init_list_depth {
    static constexpr int value = 0;
};
template <typename T> struct init_list_depth<std::initializer_list<T>> {
    static constexpr int value = 1 + init_list_depth<T>::value;
};

// Find the lowest-level base type (e.g., float, int) (Compile-time)
template <typename T> struct init_list_base {
    using type = T;
};
template <typename T> struct init_list_base<std::initializer_list<T>> {
    using type = typename init_list_base<T>::type;
};

#include <atomic>

inline std::atomic<uint64_t> global_epoch{1};

class Primitive;
class StackPrimitive;
struct data {
    void *buffer = nullptr;
    id<MTLBuffer> metalBuffer = nil;
    std::atomic<uint32_t> *refCount = nil;
    data(){}
    static data allocate(size_t num_elements, dtype type) {
        data d;
        d.buffer = new uint8_t[num_elements * dtype_size(type)];
        d.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:d.buffer length:num_elements * dtype_size(type) options:MTLResourceStorageModeShared deallocator:^(void *_Nonnull pointer, NSUInteger length){
        }];
        d.refCount = new std::atomic<uint32_t>(1);
        return d;
    }
    static data just_allocate(size_t num_elements, dtype type) {
        data d;
        d.buffer = new uint8_t[num_elements * dtype_size(type)];
        return d;
    }
    data(const data& other) {
        buffer = other.buffer;
        
    }
    ~data() {
        if (refCount) {
            // if has refcount meaning it is a realised node and realised nodes must have a buffer
            if (refCount->fetch_sub(1) == 1) {
                delete [] (uint8_t*)buffer;
            }
        } else {
            // if it doesnt have a refcount meaning its either a unrealised node or a leaf node without refcount
            // NON_OWNERSHIP Flag support is done by matrix
            if (buffer) {
                delete [] (uint8_t*)buffer;
            }
        }
    }
};
class matrix {
public:
    array_descriptor array_desc;
    
    void *buffer = nullptr;
    size_t total_size;
    id<MTLBuffer> metalBuffer = nil;
    std::atomic<uint32_t> *refCount = nil;
    Primitive *tape = nullptr;
    
    uint32_t dims = 0;
    uint8_t flags = 0;
    dtype type;
    
    template <typename Type>
    matrix(std::initializer_list<Type> list) {
        total_size = list.size();
        shape()[0] = (size_m)list.size();
        buffer = new Type[total_size];
        memcpy(buffer, list.begin(), sizeof(Type) * list.size());
        strides()[0] = 1;
        type = dtype_from_type<Type>();
        dims = 1;
        if (total_size > 10) {
            buildMetalBuffer();
        }
    }
    
    template <typename Type>
    static matrix of(std::initializer_list<Type> list) {
        matrix output(1, dtype_from_type<Type>());
        output.total_size = list.size();
        output.shape()[0] = (size_m)list.size();
        output.buffer = new Type[output.total_size];
        memcpy(output.buffer, list.begin(), sizeof(Type) * list.size());
        output.strides()[0] = 1;
        if (output.total_size > 10) {
            output.buildMetalBuffer();
        }
        return output;
    }
    
    template <typename Type>
    static matrix of(std::initializer_list<std::initializer_list<Type>> list) {
        matrix output(2, dtype_from_type<Type>());
        output.setup_from_list<2, Type>(list);
        if (output.total_size > 10) {
            output.buildMetalBuffer();
        }
        return output;
    }
    
    inline simd_float4x4& SIMD_MAT(int i) {
        eval();
        if (i >= shape()[0]) {
            throw std::invalid_argument(
                std::format("matrix: Index {} out of bounds for the axis shape {}", i, shape()[0])
            );
        }
        return *(simd_float4x4*)((uint8_t*)buffer + i * sizeof(simd_float4x4));
    }
    
    matrix(uint32_t rank, size_t total_size_inp, dtype type_inp);
    
    // Default constructor for empty declarations
    matrix() : matrix((uint32_t)1, (size_t)0, dtype::Float) {}

    matrix(uint32_t rank, dtype type_inp);
    
    // MARK: // --- Unified Initialization Logic ---
    template <int Dims, typename BaseType, typename NestedList>
    void setup_from_list(const NestedList &list) {
        this->dims = Dims;
        
        type = dtype_from_type<BaseType>();
        
        // 2. Trigger SBO logic if dimensions exceed threshold
        if (this->dims > SBO_MAX_DIMS) {
            array_desc.shared_arr_desc = SharedArrayDescriptor::create(this->dims);
        }
        
        // 3. Extract the shape dynamically
        extract_shape(this->shape(), list, 0);
        
        // 4. Calculate strides and total size
        this->total_size = 1;
        for (int i = 0; i < this->dims; ++i) {
            this->total_size *= this->shape()[i];
        }
        
        size_m current_stride = 1;
        for (int i = this->dims - 1; i >= 0; --i) {
            this->strides()[i] = current_stride;
            current_stride *= this->shape()[i];
        }
        
        // 5. Allocate raw memory
        this->buffer = new uint8_t[this->total_size * sizeof(BaseType)];
        
        // 6. Flatten and copy data
        BaseType *raw_ptr = static_cast<BaseType *>(this->buffer);
        size_m offset = 0;
        copy_data(raw_ptr, list, offset);
    }

    // --- Recursive Helpers (from previous step) ---
    template <typename ListType>
    void extract_shape(size_m *shape_arr,
                               const std::initializer_list<ListType> &list,
                               int current_dim) {
        shape_arr[current_dim] = static_cast<size_m>(list.size());
        if constexpr (is_init_list<ListType>::value) {
            if (list.size() > 0) {
                extract_shape(shape_arr, *list.begin(), current_dim + 1);
            }
        }
    }

    template <typename ListType, typename BaseType>
    void copy_data(BaseType *dest,
                           const std::initializer_list<ListType> &list,
                           size_m &offset) {
        if constexpr (is_init_list<ListType>::value) {
            for (const auto &sub_list : list) {
                copy_data(dest, sub_list, offset);
            }
        } else {
            for (const auto &item : list) {
                dest[offset++] = item;
            }
        }
    }
    
    template <typename Type>
    matrix(std::initializer_list<std::initializer_list<Type>> list) {
        setup_from_list<2, Type>(list);
    }
    template <typename Type>
    matrix(
           std::initializer_list<std::initializer_list<std::initializer_list<Type>>>
           list) {
        setup_from_list<3, Type>(list);
    }
    template <typename Type>
    matrix(std::initializer_list<std::initializer_list<
           std::initializer_list<std::initializer_list<Type>>>>
           list) {
        setup_from_list<4, Type>(list);
    }
    
    
    template <typename T>
    static matrix scalar(T val, dtype type = dtype_from_type<T>()) {
        matrix s(0, type);
        s.total_size = 1;
        s.flags = 0;
        s.buffer = new uint8_t[dtype_size(type)];
        
        dispatch_type(type, s.buffer, [&](auto* typed_buffer) {
            using BufT = std::decay_t<decltype(*typed_buffer)>;
            *typed_buffer = static_cast<BufT>(val);
        });
        s.begin_refcount();
        return s;
    }
    
    inline size_m* shape() {
        return dims <= SBO_MAX_DIMS ? array_desc.inline_buffer : array_desc.shared_arr_desc->shape();
    }
    inline size_m* strides() {
        return dims <= SBO_MAX_DIMS ? array_desc.inline_buffer + SBO_MAX_DIMS : array_desc.shared_arr_desc->strides(dims);
    }

    inline const size_m* shape() const {
        return dims <= SBO_MAX_DIMS ? array_desc.inline_buffer : array_desc.shared_arr_desc->shape();
    }

    inline const size_m* strides() const {
        return dims <= SBO_MAX_DIMS ? array_desc.inline_buffer + SBO_MAX_DIMS : array_desc.shared_arr_desc->strides(dims);
    }

    inline void buildMetalBuffer() {
        metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:effectiveBufferSize() * dtype_size(type) options:MTLResourceStorageModeShared deallocator:^(void *_Nonnull pointer, NSUInteger length){
        }];
    }
    inline void detach_shape() {
        if (dims > SBO_MAX_DIMS) {
            auto old_desc = array_desc.shared_arr_desc;
            if (old_desc->refCount.load(std::memory_order_acquire) == 1)
                return; // already exclusive
            array_desc.shared_arr_desc = SharedArrayDescriptor::create(dims);
            memcpy(shape(), old_desc->shape(), dims * sizeof(size_m));
            memcpy(strides(), old_desc->strides(dims), dims * sizeof(size_m));
            
            old_desc->release();
        }
    }
    void shareBuffer(matrix &mat) const;
    
    void beginReferenceCounting();
    void calcStrides();
    
    size_t accumul(uint32_t start, uint32_t end) const;
    size_t effectiveBufferSize() const;
    
    static matrix withShape(std::initializer_list<size_m> shape, dtype type) {
        matrix output(shape.size(), type);
        memcpy(output.shape(), shape.begin(), output.dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = output.accumul(0, output.dims);
        output.buffer = new uint8_t[output.total_size * dtype_size(type)];
        if (output.total_size > 10) {
            output.buildMetalBuffer();
        }
        return output;
    }
    
    void print() const;
    __attribute__((used)) void printShape(bool verbose = true) const {
        if (verbose == true) { std::cout << "Shape is [ "; }
        const size_m* shape_ptr = shape();
        for (int i=0; i<dims; i++) {
            std::cout << shape_ptr[i] << ", ";
        }
        if (verbose == true) { std::cout << "]\n"; }
        
    }
    
    __attribute__((used)) void printStrides(bool verbose = true) const {
        if (verbose == true) { std::cout << "Strides are [ "; }
        const size_m* stride_ptr = strides();
        for (int i=0; i<dims; i++) {
            std::cout << stride_ptr[i] << ", ";
        }
        if (verbose == true) { std::cout << "]\n"; }
        
    }
    
    friend void setBufferOrBytes(id<MTLComputeCommandEncoder> commandEncoder, const matrix &tensor, NSUInteger index);
    
    static matrix zeros(std::initializer_list<size_m> shapeI, dtype type = dtype::Float) {
        matrix output((uint32_t)shapeI.size(), type);
        memcpy(output.shape(), shapeI.begin(), output.dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = output.accumul(0, output.dims);
        output.buffer = new uint8_t[output.total_size * dtype_size(type)];
        if (output.total_size > 10) {
            output.buildMetalBuffer();
        }
        memset(output.buffer, 0, output.total_size * dtype_size(type));
        return output;
    }
    static matrix zeros(const std::vector<size_m>& shapeI, dtype type = dtype::Float) {
        matrix output((uint32_t)shapeI.size(), type);
        memcpy(output.shape(), shapeI.data(), output.dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = output.accumul(0, output.dims);
        output.buffer = new uint8_t[output.total_size * dtype_size(type)];
        if (output.total_size > 10) {
            output.buildMetalBuffer();
        }
        memset(output.buffer, 0, output.total_size * dtype_size(type));
        return output;
    }
    
    static matrix ones(std::initializer_list<size_m> shapeI, dtype type = dtype::Float) {
        matrix output((uint32_t)shapeI.size(), type);
        memcpy(output.shape(), shapeI.begin(), output.dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = output.accumul(0, output.dims);
        output.buffer = new uint8_t[output.total_size * dtype_size(type)];
        if (output.total_size > 10) {
            output.buildMetalBuffer();
        }
        dispatch_type(type, output.buffer, [&](auto* data) {
            std::fill(data, data + output.total_size, static_cast<std::decay_t<decltype(*data)>>(1));
        });
        return output;
    }
    
    static matrix gaussian(std::initializer_list<size_m> shapeI, float std_dev = 1.0f, bool normalize = true) {
        matrix output((uint32_t)shapeI.size(), dtype::Float);
        memcpy(output.shape(), shapeI.begin(), output.dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = output.accumul(0, output.dims);
        output.buffer = new uint8_t[output.total_size * dtype_size(dtype::Float)];
        float* buf = (float*)output.buffer;
        float sum = 0.0f;
        
        if (output.dims == 1) {
            float c0 = (output.shape()[0] - 1) / 2.0f;
            for (size_m i = 0; i < output.shape()[0]; ++i) {
                float dx = i - c0;
                float val = std::exp(-(dx*dx) / (2 * std_dev * std_dev));
                buf[i] = val;
                sum += val;
            }
        } else if (output.dims == 2) {
            float c0 = (output.shape()[0] - 1) / 2.0f;
            float c1 = (output.shape()[1] - 1) / 2.0f;
            for (size_m i = 0; i < output.shape()[0]; ++i) {
                for (size_m j = 0; j < output.shape()[1]; ++j) {
                    float dx = i - c0;
                    float dy = j - c1;
                    float val = std::exp(-(dx*dx + dy*dy) / (2 * std_dev * std_dev));
                    buf[i * output.strides()[0] + j] = val;
                    sum += val;
                }
            }
        } else if (output.dims == 3) {
            float c0 = (output.shape()[0] - 1) / 2.0f;
            float c1 = (output.shape()[1] - 1) / 2.0f;
            float c2 = (output.shape()[2] - 1) / 2.0f;
            for (size_m i = 0; i < output.shape()[0]; ++i) {
                for (size_m j = 0; j < output.shape()[1]; ++j) {
                    for (size_m k = 0; k < output.shape()[2]; ++k) {
                        float dx = i - c0;
                        float dy = j - c1;
                        float dz = k - c2;
                        float val = std::exp(-(dx*dx + dy*dy + dz*dz) / (2 * std_dev * std_dev));
                        buf[i * output.strides()[0] + j * output.strides()[1] + k] = val;
                        sum += val;
                    }
                }
            }
        }
        
        if (normalize && sum > 0.0f) {
            for (size_t i = 0; i < output.total_size; ++i) {
                buf[i] /= sum;
            }
        }
        
        if (output.total_size > 10) {
            output.buildMetalBuffer();
        }
        
        return output;
    }

    
    static matrix eye(uint m, uint n, int k, dtype type = dtype::Float);
    static matrix eye(uint m, dtype type = dtype::Float);
    
    static matrix leaf(std::initializer_list<size_m> shape, dtype type = dtype::Float);
    static matrix leaf(const std::vector<size_m>& shape, dtype type = dtype::Float);
    void make_leaf();
    
    matrix ones() const;
    matrix zeros() const;

    static matrix stack(const std::vector<matrix>& mats, int axis);

    static void stack(const std::vector<matrix>& mats, matrix& output, int axis = 0, ExecutionDevice exec_device = ExecutionDevice::AUTO);
    matrix pad(std::initializer_list<std::pair<size_m, size_m>> padding_range, const matrix& value) const;
    
    void pad(std::vector<size_m>& padding_range, matrix& padded_mat, const matrix& value, ExecutionDevice exec_device = ExecutionDevice::AUTO);
    matrix pad(int left, int right, int axis, const matrix& val) const;
    matrix pad(const std::vector<size_m>& padding_range, const matrix& value) const;
    matrix pad_range_normalised(const std::vector<size_m>& padding_range, const size_m* target_shape, const matrix& value) const;
    
    matrix broadcast_toV2(size_m* broadcasted_shape, int broadcasted_dim) const;
    matrix broadcast_toV2(std::initializer_list<size_m> inp_broadcasted_shape) const;
    matrix broadcast_to_impl(const size_m* broadcasted_shape, int broadcasted_dim) const;
    
    // matrix.h
    template <typename... Args> requires (std::convertible_to<Args, size_m> && ...)
    matrix reshape(Args... newShape);
    
    void reshape_eval(matrix& output, array_descriptor reshape_desc, size_t reshape_dims, ExecutionDevice execDev = ExecutionDevice::AUTO);
    matrix reshape(array_descriptor reshape_desc, size_t reshape_dims);
    matrix unsqueeze(int axis) const;
    matrix unsqueeze(uint32_t* axes, uint32_t num_axes) const;
    matrix unsqueeze(int insertion_axis, int num) const;
    matrix squeeze(int axis) const;
    matrix squeeze() const;
    
    matrix astype(dtype type, bool make_contig = false) const;
    void astype(matrix output, dtype type, EvalType eval_type = EvalType::EVAL_AUTO, ExecutionDevice execDev = ExecutionDevice::AUTO) const;
    
    static std::pair<matrix, matrix> meshgrid(const matrix& x, const matrix& y, bool sparse = false);
    static std::tuple<matrix, matrix, matrix> meshgrid(const matrix& x, const matrix& y, const matrix& z, bool sparse = false);
    
    static matrix concat(const std::vector<matrix>& mats, int axis=0);
    static void concat(const std::vector<matrix>& mats, matrix& output, int axis, ExecutionDevice exec_device);

    static matrix repeating(std::initializer_list<size_m> shapeI, const matrix &pattern) {

        matrix output(pattern.dims + shapeI.size(), pattern.type);
        
        memcpy(output.shape(), shapeI.begin(), shapeI.size() * sizeof(size_m));
        memcpy(output.shape() + shapeI.size(), pattern.shape(),
               pattern.dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = output.accumul(0, output.dims);
        output.buffer = new uint8_t[output.total_size * dtype_size(pattern.type)];
        PatternFill(output.buffer, pattern.buffer,
                    pattern.total_size * dtype_size(pattern.type),
                    output.accumul(0, shapeI.size()));
        if (output.total_size > 10 || pattern.metalBuffer)
            output.buildMetalBuffer();
        return output;
    }
    
    static matrix repeatingGPU(std::initializer_list<size_m> shapeI,
                               const matrix &pattern) {

        matrix output(pattern.dims + shapeI.size(), pattern.type);
        matrix patternView(pattern.dims + shapeI.size(), pattern.type);
        
        pattern.shareBuffer(patternView);
        memcpy(patternView.shape(), shapeI.begin(), shapeI.size() * sizeof(size_m));
        memcpy(patternView.shape() + shapeI.size(), pattern.shape(),
               pattern.dims * sizeof(size_m));
        memset(patternView.strides(), 0, shapeI.size() * sizeof(size_m));
        memcpy(patternView.strides() + shapeI.size(), pattern.strides(),
               pattern.dims * sizeof(size_m));
        patternView.total_size = patternView.accumul(0, patternView.dims);
        patternView.metalBuffer = pattern.metalBuffer;
        patternView.flags |= NON_CONTIGUOUS_FLAG;
        
        memcpy(output.shape(), patternView.shape(),
               patternView.dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = patternView.total_size;
        output.buffer = new uint8_t[output.total_size * dtype_size(pattern.type)];
        output.buildMetalBuffer();
        
        copyGPUinplace(output, patternView, 0);
        return output;
    }
    
    static matrix fromImage(std::string path_str = std::string("/Users/adityadude/Documents/TUSHU.HEIC"), CFDictionaryRef *meta_out = nullptr) {
#if !TARGET_OS_IPHONE
    CFStringRef path = CFStringCreateWithCString(NULL, path_str.c_str(),
                                                    kCFStringEncodingUTF8);
    CFURLRef url =
    CFURLCreateWithFileSystemPath(NULL, path, kCFURLPOSIXPathStyle, false);
    CGImageSourceRef source;
    CGImageRef cgImage;
    for (int i = 0; i < 3; i++) {
        source = CGImageSourceCreateWithURL(url, NULL);
        cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
        if (cgImage) {
            break;
        }
    }
    CFRelease(url);
    CFRelease(path);
    if (meta_out && source)
        *meta_out = CGImageSourceCopyPropertiesAtIndex(source, 0, nullptr);
#endif
    
#if TARGET_OS_IPHONE
    UIImage *image = [UIImage imageNamed:@"IMG_1278"];
    CGImageRef cgImage = image.CGImage;
#endif
    

    
    if (!cgImage) {
        std::cerr << "Failed to create CGImage" << std::endl;
    }
    size_t Imgwidth = CGImageGetWidth(cgImage);
    size_t Imgheight = CGImageGetHeight(cgImage);
    CGBitmapInfo bitmapInfo = CGImageGetBitmapInfo(cgImage);
    bool isFloat = (bitmapInfo & kCGBitmapFloatComponents) != 0;
    if (isFloat) {
        size_t bytesPerRow = 4 * sizeof(float) * Imgwidth;
        float *data = static_cast<float *>(malloc(bytesPerRow * Imgheight));
        
        CGColorSpaceRef space =
        CGColorSpaceCreateWithName(kCGColorSpaceExtendedLinearSRGB);
        CGContextRef ctx = CGBitmapContextCreate(
                                                    data, Imgwidth, Imgheight, 32, bytesPerRow, space,
                                                    kCGBitmapFloatComponents | kCGImageAlphaPremultipliedLast);
        CGColorSpaceRelease(space);
        CGContextDrawImage(ctx, CGRectMake(0, 0, Imgwidth, Imgheight), cgImage);
        CGContextRelease(ctx);
        CGImageRelease(cgImage);
        
        matrix result(3, dtype::Float);
        result.buffer = data;
        result.shape()[0] = Imgheight;
        result.shape()[1] = Imgwidth;
        result.shape()[2] = 4;
        result.calcStrides();
        result.total_size = Imgwidth * Imgheight * 4;
        result.buildMetalBuffer(); // MTLPixelFormatRGBA32Float
        return result;
    }
    std::cout << "Img of Width: " << Imgwidth << "and Height: " << Imgheight
    << "Loaded \n";
    size_t bytesPerRow = 4 * Imgwidth;
    void *data = malloc(bytesPerRow * Imgheight);
    CGContextRef context = CGBitmapContextCreate(
                                                    data, Imgwidth, Imgheight, 8, bytesPerRow,
                                                    CGImageGetColorSpace(cgImage),
                                                    kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGContextDrawImage(context, CGRectMake(0, 0, Imgwidth, Imgheight), cgImage);
    CGContextRelease(context);
    CGImageRelease(cgImage);
    
    uint8_t *pixelData = static_cast<uint8_t *>(data);
    
    matrix result(3, dtype::UInt8);
    result.buffer = pixelData;
    result.shape()[0] = Imgheight;
    result.shape()[1] = Imgwidth;
    result.shape()[2] = 4;
    result.calcStrides();
    result.total_size = Imgwidth * Imgheight * 4;
    result.buildMetalBuffer();
    return result;
}
    
    matrix slice(std::initializer_list<std::optional<std::pair<size_m, size_m>>>
                 slice_range) const;
    matrix slice(std::initializer_list<R> slice_range) const;
    matrix slice(R slice_range, int slice_axis) const;
    matrix slice(array_descriptor slice_range, std::vector<size_m>& slice_indices, const std::vector<size_m>& unsqueeze_axis, size_t offset) const;

    matrix slice_assign(std::initializer_list<std::optional<std::pair<size_m, size_m>>> slice_range, const matrix& rhs);
    matrix slice_assign(std::initializer_list<R> slice_range, const matrix& rhs);
    matrix slice_assign(R slice_range, int slice_axis, const matrix& rhs);
    matrix slice_assign(array_descriptor slice_range, std::vector<size_m>& slice_indices, size_t offset, const matrix& rhs);

    matrix take(const matrix& index, int axis) const;
    void take_backend(const matrix& index, matrix& output, int axis, ExecutionDevice exec_device) const;

    
    inline void set_array_desc(const array_descriptor& new_array_desc);
    inline void set_array_desc(const array_descriptor& new_array_desc, uint32_t new_dims);
    matrix transpose(std::initializer_list<size_m> new_axis_order) const;
    matrix transpose(const std::vector<size_m>& new_axis_order) const;
    matrix transpose(array_descriptor transpose_desc) const;
    matrix T() const;
    
    matrix broadcast_to(const size_m *target_shape, int target_dims) const;
    void SumNoRed(matrix& output, int axis, EvalType eval_type = EvalType::EVAL_AUTO);
    void sum_legacy(matrix& output, int axis,  bool keepdims = false, EvalType eval_type = EvalType::EVAL_AUTO);

    matrix insert_break(std::function<void(matrix&)> lambda, ExecutionDevice exec_device = ExecutionDevice::METAL) const;

    matrix sum_legacy(int axis,  bool keepdims = false);
    matrix sum(int axis, bool keepdims = false) const ;
    void sum(matrix& output, int axis, bool keepdims, ExecutionDevice exec_device = ExecutionDevice::AUTO);
    matrix sum(int start, int end, bool keepdims = false) const;
    matrix sum() const; // global sum
    
    matrix mean(int axis, bool keepdims = false) const;
    matrix mean(int start, int end = -1, bool keepdims = false) const;
    matrix mean() const; // global mean

    matrix rms(int axis, bool keepdims = false) const;
    matrix rms(int start, int end = -1, bool keepdims = false) const;
    matrix rms() const; // global rms

    matrix max(int axis, bool keepdims = false) const;
    void max(matrix& output, int axis, bool keepdims, ExecutionDevice exec_device = ExecutionDevice::AUTO);
    matrix max(int start, int end, bool keepdims = false) const;
    matrix max() const; // global max

    matrix min(int axis, bool keepdims = false) const;
    void min(matrix& output, int axis, bool keepdims, ExecutionDevice exec_device = ExecutionDevice::AUTO);
    matrix min(int start, int end, bool keepdims = false) const;
    matrix min() const; // global min
    void unbrodcast(matrix& output, matrix& target);
    matrix unbroadcast_shape(const size_m* target_shape, int target_dims) const;
    
    matrix operator[](AxisRange range);
    matrix operator[](AxisRange range1, AxisRange range2);
    matrix operator[](AxisRange range1, AxisRange range2, AxisRange range3);
    
    void CopyToTexture(id<MTLTexture> texture,
                       Execution exec = Execution::EncodeAndExecute);
    id<MTLTexture> ToMTLTexture(Execution exec = Execution::EncodeAndExecute);
    
    void save_as_image(std::string path, ImgType img_type);
    
//    static auto jit_gpu(std::function<matrix(matrix&)> func, matrix& sample) {
//        matrix output = func(sample);
//        output.compile_metal();
//        std::cout << [sample.metalBuffer contents] << "\n";
//        return [&sample, output](matrix& input) mutable -> matrix {
//            memcpy(sample.buffer, input.buffer, sample.effectiveBufferSize() * (dtype_size(input.type)));
//            std::cout << [sample.metalBuffer contents] << "\n";
//            output.execute_metal();
//            output.clear_trace_checks();
//            return output;
//        };
//    }
    static std::function<matrix(matrix&)> jit_gpu(std::function<matrix(matrix&)> func, matrix& sample);
    static std::function<matrix(matrix)> grad_gpu(std::function<matrix(matrix)> func, matrix sample);
    static matrix build_grad_graph(matrix& output_node, matrix& sample_input_node);
    static std::function<matrix(const matrix&)> jit_graph_gpu(std::function<matrix(matrix)> func, matrix sample);
    static std::function<matrix(matrix)> grad_graph_gpu(std::function<matrix(matrix)> func, matrix sample);
    
    static std::function<std::vector<matrix>(const std::vector<matrix>&)> multi_jit_graph_gpu(std::function<std::vector<matrix>(std::vector<matrix>)> func, const std::vector<matrix>& sample_inputs);
    static std::function<std::vector<matrix>(std::vector<matrix>)> grad_graph_gpu(std::function<std::vector<matrix>(std::vector<matrix>)> func, const std::vector<matrix>& sample_inputs);
    static std::vector<matrix> build_grad_graph(std::vector<matrix>& output_nodes, std::vector<matrix>& sample_input_nodes);
    // Add this inside the matrix class
    matrix dot(const matrix& b, bool transposeB = false);

    // And these will be our backends
    void dot_cpu(matrix& b_transposed, matrix& result);
    void batched_dot_cpu(matrix& b_transposed, matrix& result);
    void dot_gpu(matrix& b_transposed, matrix& result);
    void batched_dot_gpu(matrix& b_transposed, matrix& result);
    
    static matrix linespace(const matrix& a, const matrix& b, size_t no_of_points);

    template <typename Type>
    static matrix linespace(Type start, Type end, size_t no_of_points, dtype type = dtype_from_type<Type>()) {
        matrix result(1, no_of_points, type);
        result.shape()[0] = no_of_points;
        result.calcStrides();
        Type step = (no_of_points > 1) ? (end - start) / static_cast<Type>(no_of_points - 1) : Type(0);
        
        dispatch_type(type, result.buffer, [&](auto* typed_buffer) {
            using BufT = std::decay_t<decltype(*typed_buffer)>;
            for (size_t i = 0; i < no_of_points; i++) {
                typed_buffer[i] = static_cast<BufT>(start + i * step);
            }
        });
        
        return result;
    }
    
    static void copyGPUinplace(matrix &outMat, const matrix& inMat, int offset,
                               Execution exec = Execution::EncodeAndExecute) {
        if (exec == Execution::EncodeAndExecute) {
            if (inMat.tape) const_cast<matrix&>(inMat).eval_metal();
            if (outMat.tape) outMat.eval_metal();
        }
#ifdef SAFE_MODE
        if (inMat.total_size > outMat.total_size) {
            std::cerr << "MatrixH: CopyInplace operation requires both mats to be of "
            "same size."
            << "\n";
            throw;
        }
        if (inMat.type != outMat.type) {
            throw std::runtime_error("Type mismatch in copy");
        }
#endif
        if (inMat.type != outMat.type) {
            copyGPUinplaceTypeCasted(outMat, inMat, offset, exec);
            return;
        }
        size_t elem_size = dtype_size(inMat.type);
        id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
        //        Blit fast path for the GPU. A compute shader (even a 1D one)
        //        requires the GPU's ALU execution units to run the copy loop. A
        //        Blit command skips the ALUs entirely and uses the GPU's direct
        //        memory access (DMA) engines to blast the bytes across VRAM. It is
        //        drastically faster.
        if (!(inMat.flags & NON_CONTIGUOUS_FLAG) &&
            !(outMat.flags & NON_CONTIGUOUS_FLAG)) {
            GlobalGPUManager.endCommandEncoding();
            id<MTLBlitCommandEncoder> blitEncoder =
            [commandBuffer blitCommandEncoder];
            [blitEncoder copyFromBuffer:inMat.metalBuffer
                           sourceOffset:offset * elem_size
                               toBuffer:outMat.metalBuffer
                      destinationOffset:offset * elem_size
                                   size:inMat.total_size * elem_size];
            [blitEncoder endEncoding];
            if (exec == Execution::EncodeAndExecute) {
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
                GlobalGPUManager.setCommandBuffer(nil);
                GlobalGPUManager.setCommandEncoder(nil);
            }
            return;
        }
        uint8_t typeCode = static_cast<int>(inMat.type);
        
        auto res = collapse_dims(inMat.shape(), outMat.strides(), inMat.strides(),
                                 inMat.dims, INT32_MAX);
        uint32_t cdims = res.out_dims;
        
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(inMat.total_size, 1, 1);
        if (!outMat.metalBuffer && outMat.buffer) outMat.buildMetalBuffer();
        if (!inMat.metalBuffer && inMat.buffer) const_cast<matrix&>(inMat).buildMetalBuffer();
        setBufferOrBytes(commandEncoder, outMat, 0);
        setBufferOrBytes(commandEncoder, inMat, 1);
        if (cdims == 0) {
            size_m ones = 1;
            [commandEncoder setBytes:&ones length: sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:&ones length: sizeof(size_m) atIndex:3];
        } else {
            [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:cdims * sizeof(size_m) atIndex:3];
        }
        [commandEncoder setBytes:&offset length:sizeof(int) atIndex:4];
        
        if (cdims == 1 || cdims == 0) {
            if (!GlobalGPUManager.CopyInplace[typeCode][typeCode][0]) {
                GlobalGPUManager.initCopyInplace(typeCode, typeCode, 0);
            }
            [commandEncoder
             setComputePipelineState:GlobalGPUManager
                .CopyInplace_ComputeState[typeCode]
             [typeCode][0]];
        } else if (cdims == 2) {
            if (!GlobalGPUManager.CopyInplace[typeCode][typeCode][1]) {
                GlobalGPUManager.initCopyInplace(typeCode, typeCode, 1);
            }
            _dispatchExecutionSize = MTLSizeMake(res.shape[1], res.shape[0], 1);
            [commandEncoder
             setComputePipelineState:GlobalGPUManager
                .CopyInplace_ComputeState[typeCode]
             [typeCode][1]];
        } else if (cdims == 3) {
            if (!GlobalGPUManager.CopyInplace[typeCode][typeCode][2]) {
                GlobalGPUManager.initCopyInplace(typeCode, typeCode, 2);
            }
            _dispatchExecutionSize =
            MTLSizeMake(res.shape[2], res.shape[1], res.shape[0]);
            [commandEncoder
             setComputePipelineState:GlobalGPUManager
                .CopyInplace_ComputeState[typeCode]
             [typeCode][2]];
            
        } else {
            if (!GlobalGPUManager.CopyInplace[typeCode][typeCode][3]) {
                GlobalGPUManager.initCopyInplace(typeCode, typeCode, 3);
            }
            size_m acc = 1;
            for (int i = 0; i < cdims - 2; i++) {
                acc *= res.shape[i];
            }
            _dispatchExecutionSize =
            MTLSizeMake(res.shape[cdims - 1], res.shape[cdims - 2], acc);
            [commandEncoder setBytes:res.shape
                              length:cdims * sizeof(size_m)
                             atIndex:5];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:6];
            [commandEncoder
             setComputePipelineState:GlobalGPUManager
                .CopyInplace_ComputeState[typeCode]
             [typeCode][3]];
        }
        
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        if (exec == Execution::EncodeAndExecute) {
            GlobalGPUManager.endCommandEncoding();
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            GlobalGPUManager.setCommandBuffer(nil);
        }
    }
    
    static void copyGPUinplaceTypeCasted(matrix &outMat, const matrix &inMat, int offset,
                             Execution exec = Execution::EncodeAndExecute) {
        const_cast<matrix&>(inMat).update_from_trace();
        outMat.update_from_trace();
        if (exec == Execution::EncodeAndExecute) {
            if (inMat.tape) const_cast<matrix&>(inMat).eval_metal();
            if (outMat.tape) outMat.eval_metal();
        }
#ifdef SAFE_MODE
        if (inMat.total_size > outMat.total_size) {
            std::cerr << "MatrixH: CopyInplace operation requires both mats to be of "
            "same size."
            << "\n";
            throw;
        }
        if (inMat.type != outMat.type) {
            throw std::runtime_error("Type mismatch in copy");
        }
#endif
        uint8_t typeCode = static_cast<int>(inMat.type);
        uint8_t dstTypeCode = static_cast<int>(outMat.type);
        
        auto res = collapse_dims(inMat.shape(), outMat.strides(), inMat.strides(),
                                 inMat.dims, INT32_MAX);
        uint32_t cdims = res.out_dims;
        
        id<MTLCommandBuffer> commandBuffer = GlobalGPUManager.getCommandBuffer();
        id<MTLComputeCommandEncoder> commandEncoder = GlobalGPUManager.getCommandEncoder();
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(inMat.total_size, 1, 1);
        if (!outMat.metalBuffer && outMat.buffer) outMat.buildMetalBuffer();
        if (!inMat.metalBuffer && inMat.buffer) const_cast<matrix&>(inMat).buildMetalBuffer();
        setBufferOrBytes(commandEncoder, outMat, 0);
        setBufferOrBytes(commandEncoder, inMat, 1);
        if (cdims == 0) {
            size_m ones = 1;
            [commandEncoder setBytes:&ones length: sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:&ones length: sizeof(size_m) atIndex:3];
        } else {
            [commandEncoder setBytes:res.stridesA length:cdims * sizeof(size_m) atIndex:2];
            [commandEncoder setBytes:res.stridesB length:cdims * sizeof(size_m) atIndex:3];
        }
        [commandEncoder setBytes:&offset length:sizeof(int) atIndex:4];
        if (cdims == 1 || cdims == 0) {
            if (!GlobalGPUManager.CopyInplace[dstTypeCode][typeCode][0]) {
                GlobalGPUManager.initCopyInplace(dstTypeCode, typeCode, 0);
            }
            [commandEncoder
             setComputePipelineState:GlobalGPUManager
                .CopyInplace_ComputeState[dstTypeCode]
             [typeCode][0]];
        } else if (cdims == 2) {
            if (!GlobalGPUManager.CopyInplace[dstTypeCode][typeCode][1]) {
                GlobalGPUManager.initCopyInplace(dstTypeCode, typeCode, 1);
            }
            _dispatchExecutionSize = MTLSizeMake(res.shape[1], res.shape[0], 1);
            [commandEncoder
             setComputePipelineState:GlobalGPUManager
                .CopyInplace_ComputeState[dstTypeCode]
             [typeCode][1]];
        } else if (cdims == 3) {
            if (!GlobalGPUManager.CopyInplace[dstTypeCode][typeCode][2]) {
                GlobalGPUManager.initCopyInplace(dstTypeCode, typeCode, 2);
            }
            _dispatchExecutionSize =
            MTLSizeMake(res.shape[2], res.shape[1], res.shape[0]);
            [commandEncoder
             setComputePipelineState:GlobalGPUManager
                .CopyInplace_ComputeState[dstTypeCode]
             [typeCode][2]];
            
        } else {
            if (!GlobalGPUManager.CopyInplace[dstTypeCode][typeCode][3]) {
                GlobalGPUManager.initCopyInplace(dstTypeCode, typeCode, 3);
            }
            size_m acc = 1;
            for (int i = 0; i < cdims - 2; i++) {
                acc *= res.shape[i];
            }
            _dispatchExecutionSize =
            MTLSizeMake(res.shape[cdims - 1], res.shape[cdims - 2], acc);
            [commandEncoder setBytes:res.shape
                              length:cdims * sizeof(size_m)
                             atIndex:5];
            [commandEncoder setBytes:&cdims length:sizeof(uint32_t) atIndex:6];
            [commandEncoder
             setComputePipelineState:GlobalGPUManager
                .CopyInplace_ComputeState[dstTypeCode]
             [typeCode][3]];
        }
        
        [commandEncoder dispatchThreads:_dispatchExecutionSize
                  threadsPerThreadgroup:_threadsPerThreadgroup];
        if (exec == Execution::EncodeAndExecute) {
            GlobalGPUManager.endCommandEncoding();
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            GlobalGPUManager.setCommandBuffer(nil);
        }
    }
    
    static void copyCPUinplace(matrix &outMat, const matrix &inMat, int offset) {
        const_cast<matrix&>(inMat).update_from_trace();
        outMat.update_from_trace();

        if (inMat.tape) const_cast<matrix&>(inMat).eval_cpu();
        if (outMat.tape) outMat.eval_cpu();
        
        
#ifdef SAFE_MODE
        if (inMat.total_size > outMat.total_size) {
            std::cerr << "MatrixH: CopyInplace operation requires both mats to be of "
            "same size."
            << "\n";
            throw;
        }
        if (inMat.dims != outMat.dims) {
            throw std::runtime_error("Type mismatch in copy");
        }
#endif
        if (inMat.type != outMat.type) {
            copyCPUinplaceTypeCasted(outMat, inMat, offset);
            return;
        }
        size_t elem_size = dtype_size(inMat.type);
        if (!(inMat.flags & NON_CONTIGUOUS_FLAG) &&
            !(outMat.flags & NON_CONTIGUOUS_FLAG)) {
            size_t bytes = inMat.total_size * elem_size;
            // // Native CPU fast-paths skipping dynamic memcpy linkage (Critical for
            // nested loops plotting 32-bit floats or 4-channel uint8!)
            if (bytes == 4) {
                *reinterpret_cast<uint32_t *>(outMat.buffer) =
                *reinterpret_cast<const uint32_t *>(inMat.buffer);
            } else if (bytes == 1) {
                *reinterpret_cast<uint8_t *>(outMat.buffer) =
                *reinterpret_cast<const uint8_t *>(inMat.buffer);
            } else if (bytes == 8) {
                *reinterpret_cast<uint64_t *>(outMat.buffer) =
                *reinterpret_cast<const uint64_t *>(inMat.buffer);
            } else {
                memcpy(outMat.buffer, inMat.buffer, bytes);
            }
            return;
        }
        
        auto res = collapse_dims(inMat.shape(), outMat.strides(), inMat.strides(),
                                 inMat.dims, INT32_MAX);
        auto cdims = res.out_dims;
        dispatch_type(inMat.type, inMat.buffer, [&](auto *in_data) {
            using T = std::decay_t<decltype(*in_data)>;
            T *out_data = static_cast<T *>(outMat.buffer) + offset;
            // us stands for unsafe and fast subscripting so it doesnt suppor negative
            // indices and is super fast.
            if (cdims == 0) {
                out_data[0] = in_data[0];
            } else if (cdims == 1) {
                for (uint32_t i = 0; i < inMat.total_size; i++) {
                    out_data[i * res.stridesA[0]] = in_data[i * res.stridesB[0]];
                }
            } else if (cdims == 2) {
                if (res.stridesA[cdims - 1] == 1 && res.stridesB[cdims - 1] == 1) {
                    for (uint32_t i = 0; i < res.shape[0]; i++) {
                        memcpy(out_data + res.stridesA[0] * i,
                               in_data + res.stridesB[0] * i, res.shape[1] * elem_size);
                    }
                    return;
                }
                for (uint32_t i = 0; i < res.shape[0]; i++) {
                    for (uint32_t j = 0; j < res.shape[1]; j++) {
                        out_data[i * res.stridesA[0] + j * res.stridesA[1]] =
                        in_data[i * res.stridesB[0] + j * res.stridesB[1]];
                    }
                }
            } else if (cdims == 3) {
                if (res.stridesA[cdims - 1] == 1 && res.stridesB[cdims - 1] == 1) {
                    for (uint32_t i = 0; i < res.shape[0]; i++) {
                        for (uint32_t j = 0; j < res.shape[1]; j++) {
                            memcpy(out_data + res.stridesA[0] * i + res.stridesA[1] * j,
                                   in_data + res.stridesB[0] * i + res.stridesB[1] * j,
                                   res.shape[2] * elem_size);
                        }
                    }
                    return;
                }
                for (uint32_t i = 0; i < res.shape[0]; i++) {
                    for (uint32_t j = 0; j < res.shape[1]; j++) {
                        for (uint32_t k = 0; k < res.shape[2]; k++) {
                            out_data[i * res.stridesA[0] + j * res.stridesA[1] +
                                     k * res.stridesA[2]] =
                            in_data[i * res.stridesB[0] + j * res.stridesB[1] +
                                    k * res.stridesB[2]];
                        }
                    }
                }
                
            } else {
                uint32_t outer_iterations = 1;
                for (uint32_t o = 0; o <= cdims - 4; o++) {
                    outer_iterations *= res.shape[o];
                }
                for (uint32_t o = 0; o < outer_iterations; o++) {
                    uint32_t inMatIndex = 0;
                    uint32_t outMatIndex = 0;
                    uint32_t rem = o;
                    for (int i = cdims - 4; i >= 0; i--) {
                        inMatIndex += (rem % res.shape[i]) * res.stridesB[i];
                        outMatIndex += (rem % res.shape[i]) * res.stridesA[i];
                        rem /= res.shape[i];
                    }
                    if (res.stridesA[cdims - 1] == 1 && res.stridesB[cdims - 1] == 1) {
                        for (uint32_t i = 0; i < res.shape[0]; i++) {
                            for (uint32_t j = 0; j < res.shape[1]; j++) {
                                memcpy(out_data + res.stridesA[cdims - 3] * i +
                                       res.stridesA[cdims - 2] * j,
                                       in_data + res.stridesB[cdims - 3] * i +
                                       res.stridesB[cdims - 2] * j,
                                       res.shape[cdims - 1] * elem_size);
                            }
                        }
                        break;
                    }
                    for (uint32_t i = 0; i < res.shape[cdims - 3]; i++) {
                        for (uint32_t j = 0; j < res.shape[cdims - 2]; j++) {
                            for (uint32_t k = 0; k < res.shape[cdims - 1]; k++) {
                                out_data[outMatIndex + i * res.stridesA[cdims - 3] +
                                         j * res.stridesA[cdims - 2] +
                                         k * res.stridesA[cdims - 1]] =
                                in_data[inMatIndex + i * res.stridesB[cdims - 3] +
                                        j * res.stridesB[cdims - 2] +
                                        k * res.stridesB[cdims - 1]];
                            }
                        }
                    }
                }
            }
        });
    }
    
    static void copyCPUinplaceTypeCasted(matrix &outMat, const matrix &inMat,
                                         int offset) {
        const_cast<matrix&>(inMat).update_from_trace();
        outMat.update_from_trace();
        if (inMat.tape) const_cast<matrix&>(inMat).eval_cpu();
        if (outMat.tape) outMat.eval_cpu();
        
        auto res = collapse_dims(inMat.shape(), outMat.strides(), inMat.strides(),
                                 inMat.dims, INT32_MAX);
        auto cdims = res.out_dims;
        dispatch_type(outMat.type, outMat.buffer, [&](auto *out_data) {
            dispatch_type(inMat.type, inMat.buffer, [&](auto *in_data) {
                using DstT = std::decay_t<decltype(*out_data)>;
                out_data = out_data + offset;
                // us stands for unsafe and fast subscripting so it doesnt suppor
                // negative indices and is super fast.
                if (cdims == 0) {
                    out_data[0] = static_cast<DstT>(in_data[0]);
                } else if (cdims == 1) {
                    for (uint32_t i = 0; i < inMat.total_size; i++) {
                        out_data[i * res.stridesA[0]] = static_cast<DstT>(in_data[i * res.stridesB[0]]);
                    }
                } else if (cdims == 2) {
                    for (uint32_t i = 0; i < res.shape[0]; i++) {
                        for (uint32_t j = 0; j < res.shape[1]; j++) {
                            out_data[i * res.stridesA[0] + j * res.stridesA[1]] = static_cast<DstT>(in_data[i * res.stridesB[0] + j * res.stridesB[1]]);
                        }
                    }
                } else if (cdims == 3) {
                    for (uint32_t i = 0; i < res.shape[0]; i++) {
                        for (uint32_t j = 0; j < res.shape[1]; j++) {
                            for (uint32_t k = 0; k < res.shape[2]; k++) {
                                out_data[i * res.stridesA[0] + j * res.stridesA[1] + k * res.stridesA[2]] = static_cast<DstT>(in_data[i * res.stridesB[0] + j * res.stridesB[1] + k * res.stridesB[2]]);
                            }
                        }
                    }
                    
                } else {
                    uint32_t outer_iterations = 1;
                    for (uint32_t o = 0; o <= cdims - 4; o++) {
                        outer_iterations *= res.shape[o];
                    }
                    for (uint32_t o = 0; o < outer_iterations; o++) {
                        uint32_t inMatIndex = 0;
                        uint32_t outMatIndex = 0;
                        uint32_t rem = o;
                        for (int i = cdims - 4; i >= 0; i--) {
                            inMatIndex += (rem % res.shape[i]) * res.stridesB[i];
                            outMatIndex += (rem % res.shape[i]) * res.stridesA[i];
                            rem /= res.shape[i];
                        }
                        for (uint32_t i = 0; i < res.shape[cdims - 3]; i++) {
                            for (uint32_t j = 0; j < res.shape[cdims - 2]; j++) {
                                for (uint32_t k = 0; k < res.shape[cdims - 1]; k++) {
                                    out_data[outMatIndex + i * res.stridesA[cdims - 3] +
                                             j * res.stridesA[cdims - 2] +
                                             k * res.stridesA[cdims - 1]] =
                                    static_cast<DstT>(in_data[inMatIndex + i * res.stridesB[cdims - 3] + j * res.stridesB[cdims - 2] + k * res.stridesB[cdims - 1]]);
                                }
                            }
                        }
                    }
                }
            });
        });
    }
    
    
    void begin_refcount();
    void update_from_trace();

    int brodcast_shapes(const array_descriptor &shape1,
                        const array_descriptor &shape2, int dims1, int dims2,
                        array_descriptor &outShape);
    matrix flatten(int start_dim = 0, int end_dim = -1) const;
    
    void add( matrix &other, matrix &result,
             EvalType evalType = EvalType::EVAL_AUTO)   ;
    void multiply(const matrix &other, matrix &result,
                  EvalType evalType = EvalType::EVAL_AUTO) const;
    void subtract(const matrix &other, matrix &result,
                  EvalType evalType = EvalType::EVAL_AUTO) const;
    void divide(const matrix &other, matrix &result,
                EvalType evalType = EvalType::EVAL_AUTO) const;
    
    void add_cpu_brodcasted( matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    void multiply_cpu_brodcasted( matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    void subtract_cpu_brodcasted( matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    void divide_cpu_brodcasted(matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);

    void add_gpu_brodcasted(matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    void multiply_gpu_brodcasted(matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    void subtract_gpu_brodcasted(matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    void divide_gpu_brodcasted(matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    
    void add_cpu(matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    void multiply_cpu(matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    void subtract_cpu(matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    void divide_cpu(matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    
    void add_gpu( matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO) ;
    void multiply_gpu( matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO) ;
    void subtract_gpu( matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO) ;
    void divide_gpu( matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO) ;
    
    static matrix cross(const matrix& a, const matrix& b, int axis = -1);
    void cross_impl(matrix &other, matrix &result);
    void cross_cpu_brodcasted(matrix& other, matrix& result);
    void cross_gpu_brodcasted(matrix& other, matrix& result);
    static matrix sin(const matrix& input);
    void sin(matrix& output, ExecutionDevice exec_device);

    static matrix cos(const matrix& input);
    void cos(matrix& output, ExecutionDevice exec_device);

    static matrix tan(const matrix& input);
    void tan(matrix& output, ExecutionDevice exec_device);
    
    static matrix sqrt(const matrix& input);
    void sqrt(matrix& output, ExecutionDevice exec_device);

    static matrix exp(const matrix& input);
    void exp(matrix& output, ExecutionDevice exec_device);

    static matrix abs(const matrix& input);
    static matrix log(const matrix& input);
    
    matrix clamp(double min_val, double max_val) const;
    void clamp(matrix& output, double min_val, double max_val, ExecutionDevice exec_device) const;
    
    static matrix max(const matrix& a, const matrix& b);
    void max(const matrix& other, matrix& output, ExecutionDevice exec_device) const;
    static matrix min(const matrix& a, const matrix& b);
    void min(const matrix& other, matrix& output, ExecutionDevice exec_device) const;

    static matrix conv(const matrix& input, const matrix& kernel, const std::vector<int>& padding, const std::vector<int>& stride, const std::vector<int>& dilation, int groups = 1);
    
    static matrix conv1d(const matrix& input, const matrix& kernel, int padding = 0, int stride = 1, int dilation = 1, int groups = 1);
    static matrix conv2d(const matrix& input, const matrix& kernel, int pad_h = 0, int pad_w = 0, int stride_h = 1, int stride_w = 1, int dilation_h = 1, int dilation_w = 1, int groups = 1);
    static matrix conv3d(const matrix& input, const matrix& kernel, int pad_d = 0, int pad_h = 0, int pad_w = 0, int stride_d = 1, int stride_h = 1, int stride_w = 1, int dilation_d = 1, int dilation_h = 1, int dilation_w = 1, int groups = 1);
    
    void abs(matrix& output, ExecutionDevice exec_device);
    void log(matrix& output, ExecutionDevice exec_device);
    void conv1d_gpu(const matrix& kernel, matrix& output);
    void conv2d_gpu(const matrix& kernel, matrix& output);
    void conv3d_gpu(const matrix& kernel, matrix& output);
    void conv_gpu(const matrix& kernel, matrix& output);

    
    void eval_metal();
    void eval_cpu();
    void eval();
    static constexpr size_t GPU_EXECUTION_THRESHOLD = 10;

    void ensure_evaluated() const;

    template<typename T, typename... Args>
    inline T& at(Args... args) {
        constexpr size_t N = sizeof...(args);
        if (this->dims != N) {
            throw std::runtime_error("Matrix rank mismatch in at().");
        }
        if (this->type != dtype_from_type<T>() && !std::is_same_v<T, simd_float3> && !std::is_same_v<T, simd_float2> && !std::is_same_v<T, simd_float4> && !std::is_same_v<T, simd_float4>) {
            throw std::runtime_error("Type mismatch in at().");
        }
        
        ensure_evaluated();
        
        size_t offset = 0;
        if constexpr (N > 0) {
            size_m indices[N] = { static_cast<size_m>(args)... };
            const size_m* str = this->strides();
            const size_m* shp = this->shape();
            
            for (size_t d = 0; d < N; ++d) {
                if (indices[d] >= shp[d]) {
                    throw std::out_of_range(
                        "at(): index " + std::to_string(indices[d]) +
                        " out of bounds for dimension " + std::to_string(d) +
                        " with shape " + std::to_string(shp[d])
                    );
                }
                offset += indices[d] * str[d];
            }
        }
        return reinterpret_cast<T*>(this->buffer)[offset];
    }
    void compile_cpu();
    void compile_metal();
    void execute_cpu();
    void execute_metal();
    void clear_trace_checks();
    
    void releaseBuffer();
    void releaseTape();
    void destroyInstance();
    
    template <typename Type,
    typename = std::enable_if_t<std::is_arithmetic<Type>::value>>
    matrix &operator=(Type value);
    
    matrix(const matrix &other);
    matrix(matrix &&other) noexcept;
    matrix &operator=(matrix &&other);
    matrix &operator=(const matrix &other);
    
    ~matrix();
};
matrix operator+(const matrix &a, const matrix &b);
matrix operator-(const matrix &a, const matrix &b);
matrix operator*(const matrix &a, const matrix &b);
matrix operator/(const matrix &a, const matrix &b);

// matrix + scalar, scalar + matrix
template <typename T>
matrix operator+(const matrix& m, T scalar) { return m + matrix::scalar(scalar, m.type); }
template <typename T>
matrix operator+(T scalar, const matrix& m) { return matrix::scalar(scalar, m.type) + m; }
// matrix - scalar, scalar - matrix
template <typename T>
matrix operator-(const matrix& m, T scalar) { return m - matrix::scalar(scalar, m.type); }
template <typename T>
matrix operator-(T scalar, const matrix& m) { return matrix::scalar(scalar, m.type) - m; }
// matrix * scalar, scalar * matrix
template <typename T>
matrix operator*(const matrix& m, T scalar) { return m * matrix::scalar(scalar, m.type); }
template <typename T>
matrix operator*(T scalar, const matrix& m) { return matrix::scalar(scalar, m.type) * m; }
// matrix / scalar, scalar / matrix
template <typename T>
matrix operator/(const matrix& m, T scalar) { return m / matrix::scalar(scalar, m.type); }
template <typename T>
matrix operator/(T scalar, const matrix& m) { return matrix::scalar(scalar, m.type) / m; }

bool compare_shapes(const matrix &a, const matrix &b);

#endif // WORLDOF3D_MATRIX_H
