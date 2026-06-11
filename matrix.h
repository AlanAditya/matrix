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

using R = Range;

enum class dtype {
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

struct SharedArrayDescriptor {
    std::atomic<uint32_t> refCount;
    size_m *shape();
    size_m *strides(int dims);
    
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
};

void broadcast_shapes(const array_descriptor &arr_desc1, const array_descriptor &arr_desc2, array_descriptor &out_shape, BroadcastDescriptor *new_desc1, BroadcastDescriptor *new_desc2, int dim1, int dim2);

template <typename Func>
void dispatch_type(dtype type, void *buffer, Func &&function_to_run);

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

class Primitive;

class matrix {
public:
    array_descriptor array_desc;
    
    void *buffer = nullptr;
    size_t total_size;
    id<MTLBuffer> metalBuffer = nil;
    std::atomic<uint32_t> *refCount = nil;
    Primitive *tape = nullptr;
    
    dtype type;
    uint32_t dims = 0;
    uint8_t flags = 0;
    
    template <typename Type> matrix(std::initializer_list<Type> list) {
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
    
    matrix(uint32_t rank, size_t total_size_inp, dtype type_inp);
    
    matrix(uint32_t rank, dtype type_inp);
    
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
    
    // MARK: // --- Unified Initialization Logic ---
    template <int Dims, typename BaseType, typename NestedList>
    void setup_from_list(const NestedList &list);
    
    // --- Recursive Helpers ---
    template <typename ListType>
    void extract_shape(size_m *shape_arr,
                       const std::initializer_list<ListType> &list,
                       int current_dim);
    
    template <typename ListType, typename BaseType>
    void copy_data(BaseType *dest, const std::initializer_list<ListType> &list,
                   size_m &offset);
    
    size_m *shape();
    size_m *strides();
    
    const size_m *shape() const;
    const size_m *strides() const;
    
    void buildMetalBuffer();
    void detach_shape();
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
    
    void print();
    
    friend void setBufferOrBytes(id<MTLComputeCommandEncoder> commandEncoder,
                                 const matrix &tensor, NSUInteger index);
    
    static matrix zeros(std::initializer_list<size_m> shapeI,
                        dtype type = dtype::Float) {
        matrix output((uint32_t)shapeI.size(), type);
        
        if (output.dims > SBO_MAX_DIMS) {
            output.array_desc.shared_arr_desc =
            SharedArrayDescriptor::create(output.dims);
        }
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
    
    static matrix ones(std::initializer_list<size_m> shapeI,
                       dtype type = dtype::Float) {
        matrix output((uint32_t)shapeI.size(), type);
        
        if (output.dims > SBO_MAX_DIMS) {
            output.array_desc.shared_arr_desc =
            SharedArrayDescriptor::create(output.dims);
        }
        memcpy(output.shape(), shapeI.begin(), output.dims * sizeof(size_m));
        output.calcStrides();
        output.total_size = output.accumul(0, output.dims);
        output.buffer = new uint8_t[output.total_size * dtype_size(type)];
        if (output.total_size > 10) {
            output.buildMetalBuffer();
        }
        dispatch_type(type, output.buffer, [&](auto *typed_ptr) {
            std::fill(typed_ptr, typed_ptr + output.total_size, 1);
        });
        return output;
    }
    
    matrix ones() const;
    matrix zeros() const;
    
    static matrix repeating(std::initializer_list<size_m> shapeI,
                            const matrix &pattern) {
#ifdef SAFE_MODE
        if (shapeI.size() + dimsI != dims) {
            std::cerr << "Dimensions Dont Add up, Pattern: " << dimsI
            << " + Repeat:" << shapeI.size() << " != Total Dim" << dims
            << "\n";
            throw std::invalid_argument(
                                        "MatrixH: Repeating shape dimensions mismatch.");
        }
#endif
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
#ifdef SAFE_MODE
        if (shapeI.size() + dimsI != dims) {
            std::cerr << "Dimensions Dont Add up, Pattern: " << dimsI
            << " + Repeat:" << shapeI.size() << " != Total Dim" << dims
            << "\n";
            throw std::invalid_argument(
                                        "Repeating shape dimensions mismatch."); // FIXED
        }
#endif
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
    
    static matrix fromImage(std::string path_str = std::string(
                                                               "/Users/adityadude/Documents/TUSHU.HEIC"),
                            CFDictionaryRef *meta_out = nullptr) {
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
#endif
        
#if TARGET_OS_IPHONE
        UIImage *image = [UIImage imageNamed:@"IMG_1278"];
        CGImageRef cgImage = image.CGImage;
#endif
        
        if (meta_out && source)
            *meta_out = CGImageSourceCopyPropertiesAtIndex(source, 0, nullptr);
        
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
                 slice_range);
    matrix slice(std::initializer_list<R> slice_range);
    
    matrix broadcast_to(const size_m *target_shape, int target_dims) const;
    
    matrix operator[](Range range);
    matrix operator[](Range range1, Range range2);
    matrix operator[](int i, int j) const;
    
    void CopyToTexture(id<MTLTexture> texture,
                       Execution exec = Execution::EncodeAndExecute);
    id<MTLTexture> ToMTLTexture(Execution exec = Execution::EncodeAndExecute);
    
    void save_as_image(std::string path, ImgType img_type);
    
    static void copyGPUinplace(matrix &outMat, const matrix& inMat, int offset,
                               Execution exec = Execution::EncodeAndExecute) {
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
            if (GlobalGPUManager.gCommandEncoder) {
                [GlobalGPUManager.gCommandEncoder endEncoding];
                GlobalGPUManager.gCommandEncoder = nil;
            }
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
                GlobalGPUManager.gCommandBuffer = nil;
                GlobalGPUManager.gCommandEncoder = nil;
            }
            return;
        }
        uint8_t typeCode = static_cast<int>(inMat.type);
        
        auto res = collapse_dims(inMat.shape(), outMat.strides(), inMat.strides(),
                                 inMat.dims, INT32_MAX);
        uint32_t cdims = res.out_dims;
        
        id<MTLComputeCommandEncoder> commandEncoder =
        GlobalGPUManager.getCommandEncoder();
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(inMat.total_size, 1, 1);
        setBufferOrBytes(commandEncoder, outMat, 0);
        setBufferOrBytes(commandEncoder, inMat, 1);
        [commandEncoder setBytes:res.stridesA
                          length:cdims * sizeof(size_m)
                         atIndex:2];
        [commandEncoder setBytes:res.stridesB
                          length:cdims * sizeof(size_m)
                         atIndex:3];
        [commandEncoder setBytes:&offset length:sizeof(int) atIndex:4];
        if (cdims == 1) {
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
            [commandEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            GlobalGPUManager.gCommandBuffer = nil;
            GlobalGPUManager.gCommandEncoder = nil;
        }
    }
    
    static void
    copyGPUinplaceTypeCasted(matrix &outMat, const matrix &inMat, int offset,
                             Execution exec = Execution::EncodeAndExecute) {
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
        id<MTLComputeCommandEncoder> commandEncoder =
        GlobalGPUManager.getCommandEncoder();
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(inMat.total_size, 1, 1);
        setBufferOrBytes(commandEncoder, outMat, 0);
        setBufferOrBytes(commandEncoder, inMat, 1);
        [commandEncoder setBytes:res.stridesA
                          length:cdims * sizeof(size_m)
                         atIndex:2];
        [commandEncoder setBytes:res.stridesB
                          length:cdims * sizeof(size_m)
                         atIndex:3];
        [commandEncoder setBytes:&offset length:sizeof(int) atIndex:4];
        if (cdims == 1) {
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
            [commandEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            GlobalGPUManager.gCommandBuffer = nil;
            GlobalGPUManager.gCommandEncoder = nil;
        }
    }
    
    static void copyCPUinplace(matrix &outMat, const matrix &inMat, int offset) {
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
            T *out_data = static_cast<T *>(outMat.buffer);
            // us stands for unsafe and fast subscripting so it doesnt suppor negative
            // indices and is super fast.
            if (cdims == 1) {
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
        auto res = collapse_dims(inMat.shape(), outMat.strides(), inMat.strides(),
                                 inMat.dims, INT32_MAX);
        auto cdims = res.out_dims;
        dispatch_type(outMat.type, outMat.buffer, [&](auto *out_data) {
            dispatch_type(inMat.type, inMat.buffer, [&](auto *in_data) {
                using DstT = std::decay_t<decltype(*out_data)>;
                // us stands for unsafe and fast subscripting so it doesnt suppor
                // negative indices and is super fast.
                if (cdims == 1) {
                    for (uint32_t i = 0; i < inMat.total_size; i++) {
                        out_data[i * res.stridesA[0]] =
                        static_cast<DstT>(in_data[i * res.stridesB[0]]);
                    }
                } else if (cdims == 2) {
                    for (uint32_t i = 0; i < res.shape[0]; i++) {
                        for (uint32_t j = 0; j < res.shape[1]; j++) {
                            out_data[i * res.stridesA[0] + j * res.stridesA[1]] =
                            static_cast<DstT>(
                                              in_data[i * res.stridesB[0] + j * res.stridesB[1]]);
                        }
                    }
                } else if (cdims == 3) {
                    for (uint32_t i = 0; i < res.shape[0]; i++) {
                        for (uint32_t j = 0; j < res.shape[1]; j++) {
                            for (uint32_t k = 0; k < res.shape[2]; k++) {
                                out_data[i * res.stridesA[0] + j * res.stridesA[1] +
                                         k * res.stridesA[2]] =
                                static_cast<DstT>(
                                                  in_data[i * res.stridesB[0] + j * res.stridesB[1] +
                                                          k * res.stridesB[2]]);
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
                                    static_cast<DstT>(
                                                      in_data[inMatIndex + i * res.stridesB[cdims - 3] +
                                                              j * res.stridesB[cdims - 2] +
                                                              k * res.stridesB[cdims - 1]]);
                                }
                            }
                        }
                    }
                }
            });
        });
    }
    
    static void copyGPUinplace(matrix &outMat, matrix& inMat, int offset, Execution exec = Execution::EncodeAndExecute) {
        inMat.update_from_trace();
        outMat.update_from_trace();
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
            if (GlobalGPUManager.gCommandEncoder) {
                [GlobalGPUManager.gCommandEncoder endEncoding];
                GlobalGPUManager.gCommandEncoder = nil;
            }
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
                GlobalGPUManager.gCommandBuffer = nil;
                GlobalGPUManager.gCommandEncoder = nil;
            }
            return;
        }
        uint8_t typeCode = static_cast<int>(inMat.type);
        
        auto res = collapse_dims(inMat.shape(), outMat.strides(), inMat.strides(),
                                 inMat.dims, INT32_MAX);
        uint32_t cdims = res.out_dims;
        
        id<MTLComputeCommandEncoder> commandEncoder =
        GlobalGPUManager.getCommandEncoder();
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(inMat.total_size, 1, 1);
        setBufferOrBytes(commandEncoder, outMat, 0);
        setBufferOrBytes(commandEncoder, inMat, 1);
        [commandEncoder setBytes:res.stridesA
                          length:cdims * sizeof(size_m)
                         atIndex:2];
        [commandEncoder setBytes:res.stridesB
                          length:cdims * sizeof(size_m)
                         atIndex:3];
        [commandEncoder setBytes:&offset length:sizeof(int) atIndex:4];
        if (cdims == 1) {
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
            [commandEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            GlobalGPUManager.gCommandBuffer = nil;
            GlobalGPUManager.gCommandEncoder = nil;
        }
    }
    
    static void copyGPUinplaceTypeCasted(matrix &outMat, matrix &inMat, int offset, Execution exec = Execution::EncodeAndExecute) {
        inMat.update_from_trace();
        outMat.update_from_trace();
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
        id<MTLComputeCommandEncoder> commandEncoder =
        GlobalGPUManager.getCommandEncoder();
        
        auto _threadsPerThreadgroup = MTLSizeMake(16, 1, 1);
        auto _dispatchExecutionSize = MTLSizeMake(inMat.total_size, 1, 1);
        setBufferOrBytes(commandEncoder, outMat, 0);
        setBufferOrBytes(commandEncoder, inMat, 1);
        [commandEncoder setBytes:res.stridesA
                          length:cdims * sizeof(size_m)
                         atIndex:2];
        [commandEncoder setBytes:res.stridesB
                          length:cdims * sizeof(size_m)
                         atIndex:3];
        [commandEncoder setBytes:&offset length:sizeof(int) atIndex:4];
        if (cdims == 1) {
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
            [commandEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
            GlobalGPUManager.gCommandBuffer = nil;
            GlobalGPUManager.gCommandEncoder = nil;
        }
    }
    
    static void copyCPUinplace(matrix &outMat, matrix &inMat, int offset) {
        inMat.update_from_trace();
        outMat.update_from_trace();
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
            T *out_data = static_cast<T *>(outMat.buffer);
            // us stands for unsafe and fast subscripting so it doesnt suppor negative
            // indices and is super fast.
            if (cdims == 1) {
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
    
    static void copyCPUinplaceTypeCasted(matrix &outMat, matrix &inMat, int offset) {
        inMat.update_from_trace();
        outMat.update_from_trace();
        auto res = collapse_dims(inMat.shape(), outMat.strides(), inMat.strides(),
                                 inMat.dims, INT32_MAX);
        auto cdims = res.out_dims;
        dispatch_type(outMat.type, outMat.buffer, [&](auto *out_data) {
            dispatch_type(inMat.type, inMat.buffer, [&](auto *in_data) {
                using DstT = std::decay_t<decltype(*out_data)>;
                // us stands for unsafe and fast subscripting so it doesnt suppor
                // negative indices and is super fast.
                if (cdims == 1) {
                    for (uint32_t i = 0; i < inMat.total_size; i++) {
                        out_data[i * res.stridesA[0]] =
                        static_cast<DstT>(in_data[i * res.stridesB[0]]);
                    }
                } else if (cdims == 2) {
                    for (uint32_t i = 0; i < res.shape[0]; i++) {
                        for (uint32_t j = 0; j < res.shape[1]; j++) {
                            out_data[i * res.stridesA[0] + j * res.stridesA[1]] =
                            static_cast<DstT>(
                                              in_data[i * res.stridesB[0] + j * res.stridesB[1]]);
                        }
                    }
                } else if (cdims == 3) {
                    for (uint32_t i = 0; i < res.shape[0]; i++) {
                        for (uint32_t j = 0; j < res.shape[1]; j++) {
                            for (uint32_t k = 0; k < res.shape[2]; k++) {
                                out_data[i * res.stridesA[0] + j * res.stridesA[1] +
                                         k * res.stridesA[2]] =
                                static_cast<DstT>(
                                                  in_data[i * res.stridesB[0] + j * res.stridesB[1] +
                                                          k * res.stridesB[2]]);
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
                                    static_cast<DstT>(
                                                      in_data[inMatIndex + i * res.stridesB[cdims - 3] +
                                                              j * res.stridesB[cdims - 2] +
                                                              k * res.stridesB[cdims - 1]]);
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
    void add( matrix &other, matrix &result,
             EvalType evalType = EvalType::EVAL_AUTO)   ;
    void multiply(const matrix &other, matrix &result,
                  EvalType evalType = EvalType::EVAL_AUTO) const;
    void subtract(const matrix &other, matrix &result,
                  EvalType evalType = EvalType::EVAL_AUTO) const;
    void divide(const matrix &other, matrix &result,
                EvalType evalType = EvalType::EVAL_AUTO) const;
    
    void add_cpu_brodcasted( matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    void multiply_cpu_brodcasted(const matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO) const;
    void subtract_cpu_brodcasted(const matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO) const;
    void divide_cpu_brodcasted(const matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO) const;

    void add_gpu_brodcasted(matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    void multiply_gpu_brodcasted(const matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO) const;
    void subtract_gpu_brodcasted(const matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO) const;
    void divide_gpu_brodcasted(const matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO) const;
    
    void add_cpu(matrix &other, matrix &result, EvalType evalType = EvalType::EVAL_AUTO);
    void multiply_cpu(const matrix &other, matrix &result,
                      EvalType evalType = EvalType::EVAL_AUTO) const;
    void subtract_cpu(const matrix &other, matrix &result,
                      EvalType evalType = EvalType::EVAL_AUTO) const;
    void divide_cpu(const matrix &other, matrix &result,
                    EvalType evalType = EvalType::EVAL_AUTO) const;
    
    void add_gpu(const matrix &other, matrix &result,
                 EvalType evalType = EvalType::EVAL_AUTO) const;
    void multiply_gpu(const matrix &other, matrix &result,
                      EvalType evalType = EvalType::EVAL_AUTO) const;
    void subtract_gpu(const matrix &other, matrix &result,
                      EvalType evalType = EvalType::EVAL_AUTO) const;
    void divide_gpu(const matrix &other, matrix &result,
                    EvalType evalType = EvalType::EVAL_AUTO) const;
    
    void eval_metal();
    void eval_cpu();
    void compile_cpu();
    void compile_metal();
    void execute_cpu();
    void execute_metal();
    void clear_trace_checks();
    
    void releaseBuffer();
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
bool compare_shapes(const matrix &a, const matrix &b);

#endif // WORLDOF3D_MATRIX_H
