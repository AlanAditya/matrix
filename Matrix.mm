////
////  Matrix.m
////  AdityaIntelligenceProMax
////
////  Created by Aditya Dudeja on 22/10/25.
////
//
//#import <Foundation/Foundation.h>
//#import <Metal/Metal.h>
//#import <simd/simd.h>
#import <iostream>
#include <vector>
//#import "Matrix.mm"
//
//
//

//class Primitive {
//    mat operand1;
//    mat operand2;
//    
//    std::string MSLCode;
//};

class mat {
public:
    int dims;
    size_t* shape;
    size_t* strides;
    size_t total_size;
    std::vector<std::string> prevCode;
    std::string MSLCode;
    mat(const std::string& name): MSLCode(name) {
        
    }
    mat() : dims(0), shape(nullptr), strides(nullptr), total_size(0), MSLCode("") {}

    
    mat operator +( mat operand2) {
        mat result;
        result.prevCode.reserve(prevCode.size() + operand2.prevCode.size());
        std::copy(prevCode.begin(), prevCode.end(), result.prevCode.begin());
        std::copy(operand2.prevCode.begin(), operand2.prevCode.end(), result.prevCode.begin() + prevCode.size());
        result.MSLCode = MSLCode + " + " + operand2.MSLCode;
        return result;
    }
    
    mat operator -(mat operand2) {
        mat result;
        result.prevCode.reserve(prevCode.size() + operand2.prevCode.size());
        std::copy(prevCode.begin(), prevCode.end(), result.prevCode.begin());
        std::copy(operand2.prevCode.begin(), operand2.prevCode.end(), result.prevCode.begin() + prevCode.size());
        result.MSLCode = MSLCode + " - " + operand2.MSLCode;
        return result;
    }
    
    mat operator *(mat operand2) {
        mat result;
        result.prevCode.reserve(prevCode.size() + operand2.prevCode.size());
        std::copy(prevCode.begin(), prevCode.end(), result.prevCode.begin());
        std::copy(operand2.prevCode.begin(), operand2.prevCode.end(), result.prevCode.begin() + prevCode.size());
        result.MSLCode = MSLCode + " * " + operand2.MSLCode;
        return result;
    }
    
    mat operator /(mat operand2) {
        mat result;
        result.prevCode.reserve(prevCode.size() + operand2.prevCode.size());
        std::copy(prevCode.begin(), prevCode.end(), result.prevCode.begin());
        std::copy(operand2.prevCode.begin(), operand2.prevCode.end(), result.prevCode.begin() + prevCode.size());
        result.MSLCode = MSLCode + " + " + operand2.MSLCode;
        return result;
    }
};


//template <typename Type>
//class matrix {
//public:
//    Type* buffer;
//    Size_m dims;
//    size_m* shape;
//    size_m* strides;
//    size_t total_size;
//    id<MTLBuffer> metalBuffer = nil;
//    uint8_t flags = 0;
//    
//    std::vector<std::shared_ptr<matrix<dims, Type>>> parentNodes;
//    std::function<matrix<dims, Type>(matrix<dims, Type>&)> gradFunc = nullptr;
//    std::conditional_t<
//        !std::is_same<Type, Point3D>::value,
//        Type,
//        std::nullptr_t
//    > grad;
//    
//    
//    using initializer_type = typename nested_initializer_list<Type, dims>::type;
//    matrix(initializer_type nestedList) {
//        total_size = 1;
//        computeShape(nestedList, 0);
//        buffer = new Type[total_size];
//        int k = 0;
//        writeInBuffer(nestedList, k);
//        metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        calcStrides();
//    }
//    
//    matrix()
//      : buffer(nullptr), total_size(0), metalBuffer(nullptr), dims(0) {
//          std::cout << "Created" << "\n";
//      }
//    
//    template <typename T>
//    struct is_initializer_list : std::false_type {};
//
//    template <typename T>
//    struct is_initializer_list<std::initializer_list<T>> : std::true_type {};
//    
//    template <typename T>
//    void computeShape(const T& nestedList, int d) {
//        if constexpr (is_initializer_list<T>::value) {
//            if (d == dims - 1) {
//                shape[d] = nestedList.size();
//                total_size *= nestedList.size();
//            } else {
//                shape[d] = nestedList.size();
//                total_size *= nestedList.size();
//                computeShape(*nestedList.begin(), d+1);
//            }
//        }
//    }
//    
//    template <typename T>
//    void writeInBuffer(const T& nestedList, int& currentIndex) {
//        if constexpr (std::is_same<T, std::initializer_list<Type>>::value) {
//            memcpy(buffer + currentIndex, nestedList.begin(), nestedList.size() * sizeof(Type));
//            currentIndex += nestedList.size();
//        } else {
//            for (auto i: nestedList) {
//                writeInBuffer(i, currentIndex);
//            }
//        }
//        
//    };
//    
//    matrix(Type value) requires (dims == 0) {
//        buffer = new Type[1];
//        *buffer = value;
//        total_size = 1;
//        metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//    }
//    
//    matrix(size_t reserveCapacity) requires (dims != 0) {
//        buffer = new Type[reserveCapacity];
//        shape[0] = reserveCapacity;
//        total_size = reserveCapacity;
//        metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//    }
//    
//
//    
//    static matrix<dims, Type> solid() {
//        matrix<dims, Type> outputM;
//    }
//    
//    matrix<dims, Type> copy() {
//        matrix<dims, Type> result;
//        result.buffer = new Type[total_size];
//        memcpy(buffer, result.buffer, sizeof(Type) * result.total_size);
//        result.total_size = total_size;
//        memcpy(result.shape, shape, sizeof(size_m) * dims);
//        result.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:result.total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        return result;
//    }
//    
//    void buildMetalBuffer() {
//        metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//    }
//    void copyFrom(matrix<dims, Type>& input) {
//        if (!buffer) {
//            buffer = new Type[input.total_size];
//            total_size = input.total_size;
//        }
//        
//        memcpy(buffer, input.buffer, sizeof(Type) * input.total_size);
//        memcpy(shape,input.shape, sizeof(size_m) * dims);
//        metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//    }
//    
//    static size_t accumul(const std::vector<size_t>& shape) {
//        size_t acc = 1;
//        for (int i = 0; i < shape.size(); i ++) {
//            acc *= shape[i];
//        }
//        return acc;
//    }
//    
//    size_t accumul(int start, int end) const {
//        size_t acc = 1;
//        for (int i = start; i < end; i ++) {
//            acc *= shape[i];
//        }
//        return acc;
//    }
//    
//    void calcStrides() {
//        size_t acc = 1;
//        for (int i = dims-1; i >= 0; i--) {
//            strides[i] = acc;
//            acc *= shape[i];
//        }
//    }
//    
//    bool compareShapes(size_m* Othershape) {
//        bool res = true;
//        for (int i = 0; i < dims; i ++) {
//            if (shape[i] != Othershape[i]) {
//                res = false;
//            }
//        }
//        return res;
//    }
//    
//    bool compareShapes(size_m* Othershape, int end) {
//        if (end < 0) {
//            end = dims - end;
//        }
//        bool res = true;
//        for (int i = 0; i < end; i ++) {
//            if (shape[i] != Othershape[i]) {
//                res = false;
//            }
//        }
//        return res;
//    }
//    
//    static matrix<dims, Type> constant(const std::vector<size_t>& shapeI ,Type value) {
//        matrix<dims, Type> result;
//        for (int i = 0; i< dims; i++) {
//            result.shape[i] = shapeI[i];
//        }
//        result.total_size = accumul(shapeI);
//        result.buffer = new Type[result.total_size];
//        std::fill(result.buffer, result.buffer + result.total_size, value);
//        result.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:result.total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        return result;
//    }
//    
//    template <int dimsI>
//    static matrix<dims, Type> repeating(const std::vector<size_t>& shapeI, const matrix<dimsI, Type>& pattern) {
//        matrix<dims, Type> result;
//        if (shapeI.size() + dimsI != dims) {
//            std::cerr << "Dimensions Dont Add up, Pattern: " << dimsI << " + Repeat:" << shapeI.size() << " != Total Dim" << dims << "\n";
//            throw ;
//        }
//        
//        for (int i = 0; i < shapeI.size(); i++) {
//            result.shape[i] = shapeI[i];
//        }
//        for (int i = 0; i < dimsI; i++) {
//            result.shape[shapeI.size() + i] = pattern.shape[i];
//        }
//        
//        result.total_size = result.accumul(0, dims);
//        result.buffer = new Type[result.total_size];
//        
////        for (int i = 0; i < result.accumul(0, shapeI.size()); ++i) {
////            memcpy(result.buffer + i * pattern.total_size, pattern.buffer, pattern.total_size * sizeof(Type));
////        }
//        PatternFill(result.buffer, pattern.buffer, pattern.total_size * sizeof(Type), result.accumul(0, shapeI.size()));
//        result.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:result.total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        return result;
//    }
//    
//    static matrix<dims, Type> Range(Type start, std::initializer_list<size_m> shapeI) {
//        if (shapeI.size() != dims) {std::cerr << "matrix: Shape should not excede dim of matrix"; throw;}
//        matrix<dims, Type> result;
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
//        CGFloat components[] = {colorMat.buffer[0] / 255.0, colorMat.buffer[1] / 255.0, colorMat.buffer[2] / 255.0, colorMat.buffer[3] / 255.0};
//        CGColorRef color = CGColorCreate(colorSpace, components);
//        CGColorSpaceRelease(colorSpace);
//        return color;  // Remember to CFRelease when done using it
//    }
//
//
//    void drawText(char* text, matrix<1, int> point, const matrix<1, int>& colorMat, float fontSize) {
//        if (colorMat.total_size != 4) {
//            std::cerr << "Error: 4 arguments are required for colour" << "\n";
//            return;
//        }
//        
//        if (point.total_size != 2) {
//            std::cerr << "Error: 2 arguments are required for position" << "\n";
//            return;
//        }
//        CGColorRef color = createCGColorFrommatrix(colorMat);
//        
//        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
//        size_t widthsize = total_size / shape[0];
//        CGContextRef context = CGBitmapContextCreate(
//                                                     buffer, shape[1], shape[0], 8 * sizeof(Type), sizeof(Type) * widthsize, colorSpace, kCGImageAlphaPremultipliedLast
//        );
//        
//        if (!context) {
//            fprintf(stderr, "Failed to create bitmap context!\n");
//            CGColorSpaceRelease(colorSpace);
//        }
//
//        CFStringRef stringRef = CFStringCreateWithCString(NULL, text, kCFStringEncodingUTF8);
//        CTFontRef font = CTFontCreateWithName(CFSTR("Helvetica"), fontSize, NULL);
//        
//        NSDictionary *attributes = @{ (__bridge id)kCTFontAttributeName: (__bridge id)font, (__bridge id)kCTForegroundColorAttributeName: (__bridge id)color };
//        NSAttributedString *attributedString = [[NSAttributedString alloc] initWithString:(__bridge NSString *)stringRef attributes:attributes];
//        
//        
//        CTLineRef line = CTLineCreateWithAttributedString((__bridge CFAttributedStringRef)attributedString);
//        
//        CGContextSetTextPosition(context, point.buffer[0], point.buffer[1]);
//        CGContextSetTextDrawingMode(context, kCGTextFillClip);
//        
//        // Draw text
//        CTLineDraw(line, context);
//    }
//
//    
//    static matrix<dims, Type> blend(matrix<dims, Type>& mat1, matrix<dims, Type>& mat2) {
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
//        output.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:output.buffer length:output.total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        if (!GlobalGPUManager.blendInit) {
//            GlobalGPUManager.initBlend();
//        }
//        
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(mat1.shape[0] * mat1.shape[1],1, 1);
//        
//        
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:mat1.buffer length:mat1.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:mat2.buffer length:mat2.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        
//        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:output.buffer length:output.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        
//        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
//        [commandEncoder setBuffer:buffer2 offset:0 atIndex:1];
//        [commandEncoder setBuffer:buffer3 offset:0 atIndex:2];
//        [commandEncoder setComputePipelineState:GlobalGPUManager.blendCompute];
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
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(shape[0] * shape[1],1, 1);
//        
//        
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//
//        
//        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
//        [commandEncoder setBytes:&evenAlpha length:sizeof(bool) atIndex:1];
//        [commandEncoder setComputePipelineState:GlobalGPUManager.invertImgCompute];
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
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(shape[0] * shape[1],1, 1);
//        
//        
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:this->buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//
//        
//        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
//        [commandEncoder setBytes:&key length:sizeof(simd_packed_char3) atIndex:1];
//        [commandEncoder setComputePipelineState:GlobalGPUManager.chromaKeyCompute];
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
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:other.buffer length:other.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(shape[0] * shape[1], 1, 1);
//        
//        [commandEncoder setBuffer:buffer1 offset:0 atIndex:0];
//        [commandEncoder setBuffer:buffer2 offset:0 atIndex:1];
//        [commandEncoder setBuffer:buffer3 offset:0 atIndex:2];
//        [commandEncoder setComputePipelineState:GlobalGPUManager.AddImgCompute];
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
//        result.parentNodes.push_back(std::make_shared<matrix<dims, Type>>(*this));
//        result.gradFunc = [constant](matrix<dims, Type>& selfs) {
//            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
//            return p1 * constant;
//        };
//        
//        if (!GlobalGPUManager.MulAllInit) {
//            GlobalGPUManager.initMulAll();
//        }
//        
//        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:&constant length:sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//        }];
//        
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
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
//        [commandEncoder setComputePipelineState:GlobalGPUManager.MulAllCompute];
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
//        result.parentNodes.push_back(std::make_shared<matrix<dims, Type>>(*this));
//        result.parentNodes.push_back(std::make_shared<matrix<dims, Type>>(other));
//        result.gradFunc = [](matrix<dims, Type>& selfs) {
//            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
//            auto p2 = selfs.parentNodes[1]->gradFunc ? selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :  selfs.parentNodes[1]->ones();
//            return (p1) * *(selfs.parentNodes[1])+
//                       p2 * *(selfs.parentNodes[0]);
//        };
//        
//        if (!GlobalGPUManager.MulAllInit) {
//            GlobalGPUManager.initMulAll();
//        }
//        
////        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
////        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:other.buffer length:other.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
////        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
//        
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
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
//        [commandEncoder setComputePipelineState:GlobalGPUManager.MulAllCompute];
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
//    static matrix<dims, Type> concat(matrix<dims, Type>& Mat1, matrix<dims, Type>& Mat2, int axis) {
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
//            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1, strideMat1 * sizeof(Type));
//            memcpy(output.buffer + strideMat1 + i * stride, Mat2.buffer + i * strideMat2, strideMat2 * sizeof(Type));
//        }
//        
//        return output;
//    }
//    
//    static void concat(matrix<dims, Type>& Mat1, matrix<dims, Type>& Mat2, matrix<dims, Type>& output, int axis) {
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
//            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1, strideMat1 * sizeof(Type));
//            memcpy(output.buffer + strideMat1 + i * stride, Mat2.buffer + i * strideMat2, strideMat2 * sizeof(Type));
//        }
//    }
//    
//    static void concatID(matrix<dims, Type>& Mat1, matrix<dims, Type>& Mat2, matrix<dims+1, Type>& output) {
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
//            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1, strideMat1 * sizeof(Type));
//            memcpy(output.buffer + strideMat1 + i * stride, Mat2.buffer + i * strideMat2, strideMat2 * sizeof(Type));
//        }
//        
//        return output;
//    }
//    
//    static matrix<dims+1, Type> concatID(matrix<dims, Type>& Mat1, matrix<dims, Type>& Mat2) {
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
//            memcpy(output.buffer + i * stride, Mat1.buffer + i * strideMat1, strideMat1 * sizeof(Type));
//            memcpy(output.buffer + strideMat1 + i * stride, Mat2.buffer + i * strideMat2, strideMat2 * sizeof(Type));
//        }
//        
//        return output;
//    }
//    
//    static matrix<dims, Type> zeros(std::initializer_list<size_m> shapeI) {
//        if (shapeI.size() != dims) {std::cerr << "matrix: dimensions in provided shape must match the dims of " << dims << "\n"; throw;}
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
//        if (shapeI.size() != dims) {std::cerr << "matrix: dimensions in provided shape must match the dims of " << dims << "\n"; throw;}
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
//    static matrix<dims, Type> withShape(std::initializer_list<size_m> shapeI) {
//        if (shapeI.size() != dims) {std::cerr << "matrix: dimensions in provided shape must match the dims of " << dims << "\n"; throw;}
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
//                // Moving the Diagnol Left is Same as moving it above as y = (x + k) ==> (y - k) = x
//                output.buffer[(i - k) * output.shape[1] + i] = 1;
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
//                std::cerr << "Error Dimensions not equal at index " << i << "as " << shape[i+2] << " != " << element.shape[i] << "\n";
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
//            memcpy(buffer + Y * widthsize + (X + i) * elementSize, element.buffer, element.total_size * sizeof(Type));
//        }
//        
//        for (int j = Y+1; j < Y + height; j++) {
//            memcpy(buffer + j * widthsize + X * elementSize , buffer + Y * widthsize + X * elementSize, element.total_size * width * sizeof(Type));
//        }
//    }
//    
//    simd_float2 NormaliseShapeToScreen(simd_int2 deviceCoord) {
//        simd_float2 size = simd_make_float2(shape[1], shape[0]);
//        
//        return simd_make_float2((float)deviceCoord.x, (float)deviceCoord.y) / size;
//    }
//    
//    void drawRect(simd_int2 p1, simd_int2 p2, matrix<dims, Type> element) {
//        for (int i = 0; i < shape.size() - 2; i++) {
//            if (shape[i+2] != element.shape[i]) {
//                std::cerr << "Error Dimensions not equal at index " << i << "as " << shape[i+2] << " != " << element.shape[i] << "\n";
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
//            memcpy(buffer + Y * widthsize + (X + i) * elementSize, element.buffer, element.total_size * sizeof(Type));
//        }
//        
//        for (int j = Y+1; j < Y + height; j++) {
//            memcpy(buffer + j * widthsize + X * elementSize , buffer + Y * widthsize + X * elementSize, element.total_size * width * sizeof(Type));
//        }
//    }
//    void drawElipse(const simd_int4& rect, const matrix<dims-2, Type>& element) {
//        int X = rect[0];
//        int Y = rect[1];
//        int width = rect[2];
//        int height = rect[3];
//        for (int i = 0; i < dims - 2; i++) {
//            if (shape[i+2] != element.shape[i]) {
//                std::cerr << "Elipse: Error Dimensions not equal at index " << i << "as " << shape[i+2] << " != " << element.shape[i] << "\n";
//                std::cerr << shape << " != " << element.shape << "\n";
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
//                float S1 = simd_dot(((coord - centre) / rad), ((coord - centre) / rad)) - 1.0;
//                
//                if (S1 < 0.0) {
//                    memcpy(buffer + j * widthsize + i * elementSize , element.buffer, element.total_size * sizeof(Type));
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
//        return simd_make_float4(buffer[offset + 0], buffer[offset + 1], buffer[offset + 2], buffer[offset + 3]);
//    }
//    CVPixelBufferRef createPixelBufferFromMat() const {
//        
//        size_t width = shape[1];
//        size_t height = shape[0];
//        size_t channels = shape[2];
//        NSDictionary *pixelAttributes = @{(id)kCVPixelBufferIOSurfacePropertiesKey : @{}};
//        
//        CVPixelBufferRef pixelBuffer;
//        CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)pixelAttributes, &pixelBuffer);
//
//        CVPixelBufferLockBaseAddress(pixelBuffer, 0);
//        void *bufferAddress = CVPixelBufferGetBaseAddress(pixelBuffer);
//
//        size_t bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer);
//         
//         // Assuming channels is the number of bytes per pixel (should be 4 for BGRA)
//         size_t copyBytesPerRow = width * channels;
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
//        CFStringRef path = CFStringCreateWithCString(NULL, "/Users/adityadude/Documents/IMG_1278.JPG", kCFStringEncodingUTF8);
//        CFURLRef url = CFURLCreateWithFileSystemPath(NULL, path, kCFURLPOSIXPathStyle, false);
//        CGImageSourceRef source = CGImageSourceCreateWithURL(url, NULL);
//        CGImageRef cgImage = CGImageSourceCreateImageAtIndex(source, 0, NULL);
//        CFRelease(url);
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
//        std::cout << "Img of Width: " <<Imgwidth<<"and Height: " << Imgheight << "Loaded \n";
//        size_t bytesPerRow = 4 * Imgwidth;
//        void *data = malloc(bytesPerRow * Imgheight);
//        CGContextRef context = CGBitmapContextCreate(data, Imgwidth, Imgheight, 8, bytesPerRow,
//                                                     CGImageGetColorSpace(cgImage),
//                                                     kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
//        CGContextDrawImage(context, CGRectMake(0, 0, Imgwidth, Imgheight), cgImage);
//        CGContextRelease(context);
//        CGImageRelease(cgImage);
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
//    //    const char* vidPath = "/Users/adityadude/Downloads/WhatsApp Video 2025-01-01 at 14.49.11.mp4";
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
//        [asset loadTracksWithMediaType:AVMediaTypeVideo completionHandler:^(NSArray<AVAssetTrack *> * videoArray, NSError * _Nullable) {
//            videoTrack = videoArray.firstObject;
//        }];
//        AVAssetReader* reader = [[AVAssetReader alloc] initWithAsset:asset error:nil];
//        NSDictionary* outputSettings = @{
//                (NSString*)kCVPixelBufferPixelFormatTypeKey: @(kCVPixelFormatType_32BGRA)  // 4-channel (BGRA)
//        };
//        for (AVAssetTrack *track in asset.tracks) {
//            NSLog(@"Track media type: %@", track.mediaType);
//        }
//        @try {
//            AVAssetReaderTrackOutput* trackOutput = [[AVAssetReaderTrackOutput alloc] initWithTrack:videoTrack outputSettings:outputSettings];
//            // Proceed with using trackOutput
//        }
//        @catch (NSException *exception) {
//            NSLog(@"Exception occurred: %@, %@", exception.name, exception.reason);
//            // Handle the exception appropriately
//        }
//        AVAssetReaderTrackOutput* trackOutput = [[AVAssetReaderTrackOutput alloc] initWithTrack:videoTrack outputSettings:outputSettings];
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
//        [asset loadValuesAsynchronouslyForKeys:@[@"duration"] completionHandler:^{
//            NSError *error = nil;
//            AVKeyValueStatus status = [asset statusOfValueForKey:@"duration" error:&error];
//            if (status == AVKeyValueStatusLoaded) {
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
//        size_t frameCount = (size_t)(CMTimeGetSeconds(duration) * videoTrack.nominalFrameRate);
//        size_t width = (size_t)videoTrack.naturalSize.width;
//        size_t height = (size_t)videoTrack.naturalSize.height;
//        size_t channels = 4; // BGRA format has 4 channels
//        std::cout << "data " << frameCount <<" "<< height <<" " <<width<< " "<<channels;
//        uint8_t* values = new uint8_t[width*height*frameCount*channels];
//        NSLog(@"CMTime %f", CMTimeGetSeconds(duration));
//        
//        size_t frameIndex = 0;
//        while ([reader status] == AVAssetReaderStatusReading) {
//            CMSampleBufferRef sampleBuffer = [trackOutput copyNextSampleBuffer];
//            if (!sampleBuffer) {std::cout << "error "; break; }
//            CVImageBufferRef imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer);
//            CVPixelBufferLockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
//            uint8_t* baseAddress = (uint8_t*)CVPixelBufferGetBaseAddress(imageBuffer);
//            size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
//            for (size_t y = 0; y < height; y++) {
//                memcpy(values + (frameIndex * height * width * channels) + (y * width * channels),
//                       baseAddress + (y * bytesPerRow),
//                       width * channels * sizeof(Type));
//            }
//            frameIndex++;
//            CVPixelBufferUnlockBaseAddress(imageBuffer, kCVPixelBufferLock_ReadOnly);
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
//        MTLRegion region = MTLRegionMake2D(0, 0, (NSUInteger)shape[1], (NSUInteger)shape[0]);
//        NSUInteger bytesPerRow = shape[1] * 4;  // 4 bytes per pixel for BGRA8
//        
////        id<MTLBuffer> Metalbuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
//        
//        id<MTLCommandBuffer> commandBuffer = [GlobalGPUManager.gCommandQueue commandBuffer];
//        id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
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
//        MTLTextureDescriptor* drawableDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm_sRGB
//                                                                                                width:(NSUInteger)shape[1]
//                                                                                               height:(NSUInteger)shape[0]
//                                                                                            mipmapped:NO];
//        drawableDesc.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
//        // Use shared storage so the CPU can read the texture data.
//        drawableDesc.storageMode = MTLStorageModeShared;
//    
//    
//        resultTexture = [GlobalGPUManager.metalDevice newTextureWithDescriptor:drawableDesc];
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
//                        std::cout << buffer[strides[0] * i + strides[1] * j + k] << ", ";
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
//                            std::cout << buffer[strides[0] * l + strides[1] * i + strides[2] * j + k] << " ";
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
//            std::cerr << "Printing only supported for 2D matrices." << std::endl;
//            return;
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
//                        std::cout << buffer[shape[2]*(shape[1] * i + j) + k] << ", ";
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
//                            std::cout << buffer[shape[3]*(shape[2]*(shape[1] * l + i) + j)  + k] << " ";
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
//            std::cerr << "Printing only supported for 2D matrices." << std::endl;
//            return;
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
//        Type* value = buffer + i * (total_size / shape[0]) + j * (total_size / (shape[0] * shape[1]));
//        return value;
//    }
//    
//    Type* operator()(size_t i, size_t j, size_t k) {
//        Type* value = buffer + i * (total_size / shape[0]) + j * (total_size / (shape[0] * shape[1])) + k * (total_size / (shape[0] * shape[1] * shape[2]));
//        return value;
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
//            std::cerr << "matrix: Invalid Dims, cannot convert from (" << total_size <<", " << valuesIn << "(imp)) To (" << output.total_size <<", " << valuesOut << ") \n";
//        }
//        
//        if (!GlobalGPUManager.typeCasting[currentTypeCode][OutTypeCode]) {
//            GlobalGPUManager.initTypeCasting(currentTypeCode, OutTypeCode);
//        }
//        
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(total_size * valuesIn, 1, 1);
//        
//        
////        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:output.buffer length:output.total_size*sizeof(OutType) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
////        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
//        
//        
//        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
////        [commandEncoder setBytes:&type length:sizeof(int) atIndex:2];
//        [commandEncoder setComputePipelineState:GlobalGPUManager.typeCasting[currentTypeCode][OutTypeCode]];
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
//            std::cerr << "matrix: Invalid Dims, cannot convert from (" << total_size <<", " << valuesIn << "(imp)) To (" << output.total_size <<", " << valuesOut << ") \n";
//        }
//        if (!GlobalGPUManager.typeCasting[currentTypeCode][OutTypeCode]) {
//            GlobalGPUManager.initTypeCasting(currentTypeCode, OutTypeCode);
//        }
//        
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(total_size * valuesIn ,1, 1);
//        
//        
////        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:output.buffer length:output.total_size*sizeof(OutType) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
////        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
//        
//        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setComputePipelineState:GlobalGPUManager.typeCasting[currentTypeCode][OutTypeCode]];
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
////        result.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:result.total_size * sizeof(float) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
////        std::memcpy(result.shape, shape, sizeof(size_m) * dims);
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
//                std::cerr << "matrix: For conversion last dim should be " << valuesOut << "\n";
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
////        result.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:result.total_size * sizeof(uint8_t) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
////        std::memcpy(result.shape + (DimsNew - dims) - dimBias, shape, sizeof(size_m) * dims);
////        std::fill(result.shape, result.shape + (DimsNew - dims) - dimBias, 1);
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
////        result.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:result.total_size * sizeof(uint8_t) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
////        std::memcpy(result.shape, shape, sizeof(size_m) * dims);
////        std::fill(result.shape + dims, result.shape + DimsNew, 1);
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
//            std::cerr << "matrix: Axis size should be equal to Dims of " << dims << "\n";
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
//            if (*(axis.begin() + i) >= dims) {std::cerr << "matrix: Rearranged axis should not increase dims" << "\n"; }
//            acc *= shape[*(axis.begin() + i)];
//            output.shape[i] = shape[*(axis.begin() + i)];
//        }
//        
//        int dimensions = dims;
//        
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
//        
//        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBytes:inputStrides length:dims * sizeof(size_m) atIndex:2];
//        [commandEncoder setBytes:outputStrides length:dims * sizeof(size_m) atIndex:3];
//        [commandEncoder setBytes:&dimensions length: sizeof(dims) atIndex:4];
//        [commandEncoder setComputePipelineState:GlobalGPUManager.TransposeComputeState[typeCode]];
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
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(mat.total_size, 1, 1);
//        
//        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:mat.metalBuffer offset:0 atIndex:1];
//
//        [commandEncoder setComputePipelineState:GlobalGPUManager.SinComputeState[typeCode]];
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
//            std::cerr << "matrix: Axis should not excede Dims of " << dims << "\n";
//            throw;
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
//        memcpy(output.shape + axis, shape + axis + 1, (dims-axis) * sizeof(size_m));
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
////        std::cout << "AxStride: " << axisStride << " ElStride: " << ElStride << " noOfOpp: " << noOfOpp << "\n";
////        printArray(inputStrides, dims-1);
////        printArray(maskedStrides, dims-1);
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(output.total_size, 1, 1);
//         
//        size_t outputDims = dims -1;
//        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBytes:&axisStride length: sizeof(size_m) atIndex:2];
//        [commandEncoder setBytes:&ElStride length: sizeof(size_m) atIndex:3];
//        [commandEncoder setBytes:&noOfOpp length: sizeof(size_m) atIndex:4];
//        [commandEncoder setBytes:&inputStrides length: (dims-1) * sizeof(size_m) atIndex:5];
//        [commandEncoder setBytes:&maskedStrides length: (dims-1)* sizeof(size_m) atIndex:6];
//        [commandEncoder setBytes:&outputDims length:  sizeof(size_m) atIndex:7];
//        [commandEncoder setComputePipelineState:GlobalGPUManager.SumComputeState[typeCode]];
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
//            if (*(axis + i) >= dims) {std::cerr << "matrix: Rearranged axis should not increase dims" << "\n"; }
//            acc *= shape[axis[i]];
//            output.shape[i] = shape[axis[i]];
//        }
//        
//        int dimensions = dims;
//        
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(total_size, 1, 1);
//        
//        [commandEncoder setBuffer:output.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBytes:inputStrides length:dims * sizeof(size_t) atIndex:2];
//        [commandEncoder setBytes:outputStrides length:dims * sizeof(size_t) atIndex:3];
//        [commandEncoder setBytes:&dimensions length: sizeof(dims) atIndex:4];
//        [commandEncoder setComputePipelineState:GlobalGPUManager.TransposeComputeState[typeCode]];
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
//    void conv(matrix<dims, Type>& result, const matrix<kDims, Type> &kernel, int sepAxis = kDims, ConvMode mode = ConvMode::Same) {
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
//                memcpy(newShape + sepAxis, shape + sepAxis, (dims - sepAxis) * sizeof(size_m));
//                break;
//            case ConvMode::Full:
//                noOfOpp = 1;
//                for (size_m i = 0; i < sepAxis; ++i) {
//                    newShape[i] = shape[i] + (kernel.shape[i] - 1);
//                    noOfOpp *= newShape[i];
//                }
//                memcpy(newShape + sepAxis, shape + sepAxis, (dims - sepAxis) * sizeof(size_m));
//                break;
//        }
//        
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        MTLLogStateDescriptor *logStateDesc = [MTLLogStateDescriptor new];
//        logStateDesc.bufferSize = 1024*1024;
//        logStateDesc.level = MTLLogLevelInfo;
//        NSError* error = nil;
//        id<MTLLogState> logState = [GlobalGPUManager.metalDevice newLogStateWithDescriptor:logStateDesc error:&error];
//        [logState addLogHandler:^(NSString *substring, NSString *category,
//                                  MTLLogLevel level, NSString *message)
//        {
//        }];
//        
//        
//        MTLCommandBufferDescriptor *cbufDesc = [MTLCommandBufferDescriptor new];
//        cbufDesc.logState = logState;
//        
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBufferWithDescriptor:cbufDesc];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(noOfOpp, 1, 1);
//        
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBuffer:kernel.metalBuffer offset:0 atIndex:2];
//
//        [commandEncoder setBytes:&kernel.total_size length: sizeof(size_t) atIndex:3];
//        [commandEncoder setBytes:shape length: kDims * sizeof(size_m) atIndex:4];
//        [commandEncoder setBytes:kernel.shape length:kDims * sizeof(size_m) atIndex:5];
//        if (sepAxis == dims) {
//            [commandEncoder setBytes:strides length:kDims * sizeof(size_m) atIndex:6];
//        } else {
//            size_m stridesNew[kDims];
//            size_m acc = 1;
//            for (int i = kDims-1; i >= 0; i--) {
//                stridesNew[i] = acc;
//                acc *= shape[i];
//            }
//            [commandEncoder setBytes:stridesNew length:kDims * sizeof(size_m) atIndex:6];
//        }
//        [commandEncoder setBytes:&elementStride length: sizeof(size_m) atIndex:7];
//        [commandEncoder setBytes:&sepAxis length: sizeof(int) atIndex:8];
//        
//        [commandEncoder setComputePipelineState:GlobalGPUManager.ConvolveComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//        
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//    
//    void Dot(matrix<dims, Type>& result, const matrix<dims, Type> &other, bool TransposeB) {
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
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
//        
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        
//        if (!TransposeB) {
//            matrix<dims, Type> B_transposed = other.T();
//            [commandEncoder setBuffer:B_transposed.metalBuffer offset:0 atIndex:2];
//            [commandEncoder setBytes:other.shape length:dims * sizeof(size_m) atIndex:4];
//            
//        } else {
//            size_m* reverseShapeBuffer = new size_m[dims];
//            reverseBuffer(other.shape, reverseShapeBuffer, dims);
//            [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:2];
//            [commandEncoder setBytes:reverseShapeBuffer length:dims * sizeof(size_m) atIndex:4];
//            delete [] reverseShapeBuffer;
//        }
//        
//        [commandEncoder setBytes:shape length:dims * sizeof(size_m) atIndex:3];
//        
//        [commandEncoder setComputePipelineState:GlobalGPUManager.GEMMAComputeState[typeCode]];
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
//            std::cerr << ") not aligned: "<< shape[dims-1]<<" (dim "<< dims-1 <<") != "<< other.shape[0] <<" (dim 0) \n";
//            throw;
//        }
//        
//        matrix<dims, Type> result = matrix<dims, Type>::withShape({shape[0], other.shape[1]});
//        Dot(result, other, false);
//        return result;
//    }
//    
//    matrix<dims, Type> Dot(const matrix<dims, Type> &other, bool TransposeB) {
//        if (shape[dims - 1] != other.shape[0]) {
//            std::cerr << "ValueError: shapes (" ;
//            printShape(false);
//            std::cerr << ") and (";
//            other.printShape(false);
//            std::cerr << ") not aligned: "<< shape[dims-1]<<" (dim "<< dims-1 <<") != "<< other.shape[0] <<" (dim 0) \n";
//            throw;
//        }
//        
//        matrix<dims, Type> result = matrix<dims, Type>::withShape({shape[0], other.shape[1]});
//        Dot(result, other, TransposeB);
//        return result;
//    }
//    
//    template <int dimsB, int resultDims>
//    void BrodcastedAdd(matrix<resultDims, Type>& result, const matrix<dimsB, Type> &other) const {
////        result.parentNodes.push_back(std::make_shared<matrix<dims, Type>>(*this));
////        result.parentNodes.push_back(std::make_shared<matrix<dimsB, Type>>(other));
////        result.gradFunc = [](matrix<dims, Type>& selfs) {
////            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
////            auto p2 = selfs.parentNodes[1]->gradFunc ? selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :  selfs.parentNodes[1]->ones();
////            return p1 + p2;
////        };
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
//        memcpy(strideA + (resultDims -  dims), strides, dims * sizeof(size_m));
//        memcpy(strideB + (resultDims - dimsB), other.strides, dimsB * sizeof(size_m));
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
//                    std::invalid_argument("Incompatible shapes for broadcasting");
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
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
//        int rDims = resultDims;
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:2];
//        [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
//        [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
//        [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
//        [commandEncoder setBytes:&rDims length:sizeof(int) atIndex:6];
//        
//        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedAddComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//        
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//    
//    template <int dimsB, int resultDims>
//    void BrodcastedSub(matrix<resultDims, Type>& result, const matrix<dimsB, Type> &other) const {
////        result.parentNodes.push_back(std::make_shared<matrix<dims, Type>>(*this));
////        result.parentNodes.push_back(std::make_shared<matrix<dimsB, Type>>(other));
////        result.gradFunc = [](matrix<dims, Type>& selfs) {
////            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
////            auto p2 = selfs.parentNodes[1]->gradFunc ? selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :  selfs.parentNodes[1]->ones();
////            return p1 + p2;
////        };
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
//        memcpy(strideA + (resultDims -  dims), strides, dims * sizeof(size_m));
//        memcpy(strideB + (resultDims - dimsB), other.strides, dimsB * sizeof(size_m));
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
//                    std::invalid_argument("Incompatible shapes for broadcasting");
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
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
//        int rDims = resultDims;
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:2];
//        [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
//        [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
//        [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
//        [commandEncoder setBytes:&rDims length:sizeof(int) atIndex:6];
//        
//        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedSubComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//        
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//    
//    template <int dimsB, int resultDims>
//    void BrodcastedMul(matrix<resultDims, Type>& result, const matrix<dimsB, Type> &other) const {
////        result.parentNodes.push_back(std::make_shared<matrix<dims, Type>>(*this));
////        result.parentNodes.push_back(std::make_shared<matrix<dimsB, Type>>(other));
////        result.gradFunc = [](matrix<dims, Type>& selfs) {
////            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
////            auto p2 = selfs.parentNodes[1]->gradFunc ? selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :  selfs.parentNodes[1]->ones();
////            return p1 + p2;
////        };
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
//        memcpy(strideA + (resultDims -  dims), strides, dims * sizeof(size_m));
//        memcpy(strideB + (resultDims - dimsB), other.strides, dimsB * sizeof(size_m));
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
//                    std::invalid_argument("Incompatible shapes for broadcasting");
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
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
//        int rDims = resultDims;
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:2];
//        [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
//        [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
//        [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
//        [commandEncoder setBytes:&rDims length:sizeof(int) atIndex:6];
//        
//        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedMulComputeState[typeCode]];
//        [commandEncoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
//        
//        [commandEncoder endEncoding];
//        [commandBuffer commit];
//        [commandBuffer waitUntilCompleted];
//    }
//    
//    template <int dimsB, int resultDims>
//    void BrodcastedDiv(matrix<resultDims, Type>& result, const matrix<dimsB, Type> &other) const {
////        result.parentNodes.push_back(std::make_shared<matrix<dims, Type>>(*this));
////        result.parentNodes.push_back(std::make_shared<matrix<dimsB, Type>>(other));
////        result.gradFunc = [](matrix<dims, Type>& selfs) {
////            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
////            auto p2 = selfs.parentNodes[1]->gradFunc ? selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :  selfs.parentNodes[1]->ones();
////            return p1 + p2;
////        };
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
//        memcpy(strideA + (resultDims -  dims), strides, dims * sizeof(size_m));
//        memcpy(strideB + (resultDims - dimsB), other.strides, dimsB * sizeof(size_m));
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
//                    std::invalid_argument("Incompatible shapes for broadcasting");
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
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
//        
//        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
//        auto _dispatchExecutionSize =  MTLSizeMake(result.total_size, 1, 1);
//        int rDims = resultDims;
//        [commandEncoder setBuffer:result.metalBuffer offset:0 atIndex:0];
//        [commandEncoder setBuffer:metalBuffer offset:0 atIndex:1];
//        [commandEncoder setBuffer:other.metalBuffer offset:0 atIndex:2];
//        [commandEncoder setBytes:result.strides length:resultDims * sizeof(size_m) atIndex:3];
//        [commandEncoder setBytes:strideA length:resultDims * sizeof(size_m) atIndex:4];
//        [commandEncoder setBytes:strideB length:resultDims * sizeof(size_m) atIndex:5];
//        [commandEncoder setBytes:&rDims length:sizeof(int) atIndex:6];
//        
//        [commandEncoder setComputePipelineState:GlobalGPUManager.BrodcastedDivComputeState[typeCode]];
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
//        result.parentNodes.push_back(std::make_shared<matrix<dims, Type>>(*this));
//        result.parentNodes.push_back(std::make_shared<matrix<dims, Type>>(other));
//        result.gradFunc = [](matrix<dims, Type>& selfs) {
//            auto p1 = selfs.parentNodes[0]->gradFunc ? selfs.parentNodes[0]->gradFunc(*selfs.parentNodes[0]) : selfs.parentNodes[0]->ones();
//            auto p2 = selfs.parentNodes[1]->gradFunc ? selfs.parentNodes[1]->gradFunc(*selfs.parentNodes[1]) :  selfs.parentNodes[1]->ones();
//            return p1 + p2;
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
////        id<MTLBuffer> buffer1 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
////        id<MTLBuffer> buffer2 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:other.buffer length:other.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
////        id<MTLBuffer> buffer3 = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
//        
//        
//        id<MTLCommandQueue> commandQueue = GlobalGPUManager.gCommandQueue;
//        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
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
//    matrix<dims, Type> operator+(const matrix<dimsB, Type> &other) requires (dims > dimsB) {
//        matrix<dims, Type> result;
//        BrodcastedAdd(result, other);
//        return result;
//    }
//    
//    template<int dimsB>
//    matrix<dimsB, Type> operator+(const matrix<dimsB, Type> &other) requires (dims < dimsB) {
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
//    friend matrix<dims, Type> operator+(Type value, const matrix<dims, Type>& mat) {
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
//            result.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
//            }];
//            memcpy(result.shape, shape, sizeof(size_m) * dims);
//            Add(result, other);
//        } else {
//            BrodcastedAdd(result, other);
//        }
//        
////        id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
////
////        id<MTLBuffer> buffer1 = [metalDevice newBufferWithBytesNoCopy:buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
////        id<MTLBuffer> buffer2 = [metalDevice newBufferWithBytesNoCopy:other.buffer length:other.total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
////        id<MTLBuffer> buffer3 = [metalDevice newBufferWithBytesNoCopy:result.buffer length:total_size*sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
////        }];
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
////        id<MTLComputePipelineState> computeState = [metalDevice newComputePipelineStateWithFunction:func1 error:&error];
////
////        if (error) {
////            NSLog(@"Adder: %@", error.localizedDescription);
////        }
////
////        id<MTLCommandQueue> commandQueue = [metalDevice newCommandQueue];
////        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
////        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
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
//    matrix<dims, Type> operator*(const matrix<dimsB, Type> &other) requires (dims > dimsB) {
//        matrix<dims, Type> result;
//        BrodcastedMul(result, other);
//        return result;
//    }
//    
//    template<int dimsB>
//    matrix<dimsB, Type> operator*(const matrix<dimsB, Type> &other) requires (dims < dimsB) {
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
//    friend matrix<dims, Type> operator*(Type value, const matrix<dims, Type>& mat) {
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
//        result.metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:result.buffer length:result.total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
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
//        return buffer[strides[0] * i + strides[1] * j + strides[2] * k + strides[3] * l];
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
//        slicedMat.buffer = buffer + strides[0] * i + strides[1] * j + strides[2] * k;
//        memcpy(slicedMat.strides, strides + 3, (dims-3) *sizeof(size_m));
//        memcpy(slicedMat.shape, shape + 3, (dims-3) *sizeof(size_m));
//        slicedMat.total_size = strides[2];
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
//        // copy constructor doesnt need to delete its buffer as  its called only on uninitlised matricies
////        if () {
//            buffer = new Type[other.total_size];
//            total_size = other.total_size;
//            metalBuffer = [GlobalGPUManager.metalDevice newBufferWithBytesNoCopy:buffer length:total_size * sizeof(Type) options:MTLResourceStorageModeShared deallocator:^(void * _Nonnull pointer, NSUInteger length) {
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
//    matrix<dims, Type> Derivative(matrix<dims, Type>& result, int axis, int loopBack) {
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
//        id<MTLComputeCommandEncoder> commandEncoder = [commandBuffer computeCommandEncoder];
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
//        [commandEncoder setComputePipelineState:GlobalGPUManager.DerivativeAll];
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
//    void SliceBuffer(Type* outBuff, Type* inBuff, size_t stride, size_t size, uint rows) {
//        for (int i = 0; i < rows; i++) {
//            memcpy(outBuff + i * size, inBuff + i * stride, size * sizeof(Type));
//        }
//    }
//    
//    
//    
//    void SliceCopy(matrix<dims, Type>& slicedMat, std::initializer_list<std::optional<std::pair<size_t, size_t>>> slice) {
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
//                    SliceBuffer(MemArena, buffer + offsets[index] * stride, stride, size, noOfOp);
//                    firstNonNullStrideFound = true;
//                    index++;
//                    continue;
//                    
//                }
//                SliceBuffer(MemArena, MemArena + offsets[index] * stride, stride, size, noOfOp);
//            }
//            index++;
//        }
//        slicedMat.calcStrides();
////        slicedMat.total_size = slicedMat.accumul(0, dims);
//        slicedMat.total_size = slicedMat.strides[0] * slicedMat.shape[0];
//        memcpy(slicedMat.buffer, MemArena, sizeof(Type) * slicedMat.total_size);
//    }
//    
//    matrix<dims, Type> SliceCopy(std::initializer_list<std::optional<std::pair<size_t, size_t>>> slice) {
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
//                std::cout << "Stride: " << stride << "size: " << size << "no of Opp: " << noOfOp << "\n";
//                if (!firstNonNullStrideFound) {
//                    MemArena = new Type[slicedMat.accumul(index, dims)];
//                    SliceBuffer(MemArena, buffer + offsets[index] * stride, stride, size, noOfOp);
//                    firstNonNullStrideFound = true;
//                    index++;
//                    continue;
//                    
//                }
//                SliceBuffer(MemArena, MemArena + offsets[index] * stride, stride, size, noOfOp);
//            }
//            index++;
//        }
//        slicedMat.calcStrides();
//        slicedMat.total_size = acc;
//        memcpy(slicedMat.buffer, MemArena, sizeof(Type) * slicedMat.total_size);
//        return slicedMat;
//    }
//    
//    matrix<dims, Type> Slice(std::initializer_list<std::optional<std::pair<size_t, size_t>>> slice) {
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
