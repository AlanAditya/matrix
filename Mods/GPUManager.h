//
//  modTest.mm
//  WorldOf3D
//
//  Created by Aditya Dudeja on 25/11/25.
//
@import Metal;
#include <string>
#include <utility>

//@import std_string;
@import Utils;
@import simd;
#pragma once // <--- 1. Prevents recursive inclusion
template <int dims, typename Type>
class MatrixH;

class GPUManager {
public:
    id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> gCommandQueue = [metalDevice newCommandQueue];
    
    // Thread-local storage to prevent cross-thread race conditions
    inline static thread_local void* _thread_gCommandBuffer = nullptr;
    inline static thread_local void* _thread_gCommandEncoder = nullptr;
    
    id<MTLLibrary> library = [metalDevice newDefaultLibrary];
    
    id<MTLComputePipelineState> blendCompute;
    bool blendInit = false;
    
    id<MTLComputePipelineState> invertImgCompute;
    bool invertInitImg = false;
    
    id<MTLComputePipelineState> chromaKeyCompute;
    bool chromaKeyInit = false;
    
    id<MTLComputePipelineState> AddImgCompute;
    bool AddImgInit = false;
    id<MTLComputePipelineState> AddIntCompute;
    bool AddIntInit = false;
    id<MTLComputePipelineState> AddFloatCompute;
    bool AddFloatInit = false;
    
    id<MTLComputePipelineState> SubImgCompute;
    bool SubImgInit = false;
    id<MTLComputePipelineState> SubIntCompute;
    bool SubIntInit = false;
    id<MTLComputePipelineState> SubFloatCompute;
    bool SubFloatInit = false;
    
    id<MTLComputePipelineState> MulAllCompute;
    bool MulAllInit = false;

    id<MTLComputePipelineState> ConversionAll;
    bool ConversionAllInit = false;
    
    bool typeCastingInit[6][6];
    id<MTLComputePipelineState> typeCasting[6][6];
    
    bool TransposeInit[3];
    id<MTLComputePipelineState> TransposeComputeState[3];
    
    bool GEMMAInit[4];
    id<MTLComputePipelineState> GEMMAComputeState[4];
    
    bool BatchedMatMulInit[4][2];
    id<MTLComputePipelineState> BatchedMatMulComputeState[4][2];
    
    bool SumInit[4];
    id<MTLComputePipelineState> SumComputeState[4];
    
    bool SumInit_nd[10][8];
    id<MTLComputePipelineState> SumComputeState_nd[10][8];
    
    bool MaxInit[10][8];
    id<MTLComputePipelineState> MaxComputeState_nd[10][8];
    
    bool MinInit[10][8];
    id<MTLComputePipelineState> MinComputeState_nd[10][8];
    
    bool SinInit[3];
    id<MTLComputePipelineState> SinComputeState[3];
    
    bool CosInit[3];
    id<MTLComputePipelineState> CosComputeState[3];
    
    bool TanInit[3];
    id<MTLComputePipelineState> TanComputeState[3];

    bool SqrtInit[3];
    id<MTLComputePipelineState> SqrtComputeState[3];
    
    bool ExpInit[3];
    id<MTLComputePipelineState> ExpComputeState[3];
    
    bool AbsInit[4][4];
    id<MTLComputePipelineState> AbsComputeState[4][4];
    
    bool LogInit[4][4];
    id<MTLComputePipelineState> LogComputeState[4][4];
    
    bool ClampInit_nd[4][4];
    id<MTLComputePipelineState> ClampComputeState_nd[4][4];
    
    bool SinInit_nd[4][4];
    id<MTLComputePipelineState> SinComputeState_nd[4][4];
    bool CosInit_nd[4][4];
    id<MTLComputePipelineState> CosComputeState_nd[4][4];
    bool TanInit_nd[4][4];
    id<MTLComputePipelineState> TanComputeState_nd[4][4];
    bool SqrtInit_nd[4][4];
    id<MTLComputePipelineState> SqrtComputeState_nd[4][4];
    bool ExpInit_nd[4][4];
    id<MTLComputePipelineState> ExpComputeState_nd[4][4];
    
    bool TakeInit_nd[4][4];
    id<MTLComputePipelineState> TakeComputeState_nd[4][4];
    
    bool ConvolveInit[3];
    id<MTLComputePipelineState> ConvolveComputeState[3];
    
    bool Conv1dInit[3];
    id<MTLComputePipelineState> Conv1dComputeState[3];
    bool Conv2dInit[3];
    id<MTLComputePipelineState> Conv2dComputeState[3];
    bool Conv3dInit[3];
    id<MTLComputePipelineState> Conv3dComputeState[3];
    bool ConvInit[3];
    id<MTLComputePipelineState> ConvComputeState[3];
    
    bool ConvolveFullInit[3];
    id<MTLComputePipelineState> ConvolveFullComputeState[3];
    
    id<MTLComputePipelineState> BrodcastedAddComputeState[4][4];
    id<MTLComputePipelineState> BrodcastedSubComputeState[4][4];
    id<MTLComputePipelineState> BrodcastedMulComputeState[4][4];
    id<MTLComputePipelineState> BrodcastedDivComputeState[4][4];
    id<MTLComputePipelineState> BrodcastedMaxComputeState[4][4];
    id<MTLComputePipelineState> BrodcastedMinComputeState[4][4];
    id<MTLComputePipelineState> BrodcastedCrossComputeState[4][4];
    bool BrodcastedAddInit[4][4] = {{false}};
    bool BrodcastedSubInit[4][4] = {{false}};
    bool BrodcastedMulInit[4][4] = {{false}};
    bool BrodcastedDivInit[4][4] = {{false}};
    bool BrodcastedMaxInit[4][4] = {{false}};
    bool BrodcastedMinInit[4][4] = {{false}};
    bool BrodcastedCrossInit[4][4] = {{false}};
    
    bool Concat_2M[4];
    id<MTLComputePipelineState> Concat_2M_ComputeState[4];
    
    bool CopyInplace[6][6][6];
    id<MTLComputePipelineState> CopyInplace_ComputeState[6][6][6];
    
    bool PaddingInit[6][4];
    id<MTLComputePipelineState> Padding_ComputeState[6][4];
    
    NSMutableDictionary<NSString*, NSNumber*>* shaderNameToIndex;
    NSMutableArray<id<MTLComputePipelineState>>* customComputeShader;
    
    
    GPUManager() {
        customComputeShader = [[NSMutableArray alloc] init];
        shaderNameToIndex = [[NSMutableDictionary alloc] init];
//        library = getSourceOfMetalFiles(metalDevice);
        
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                typeCastingInit[i][j] = false;
            }
        }
        for (int i = 0; i < 3; i++) {
            TransposeInit[i] = false;
        }
        for (int i = 0; i < 4; i++) {
            GEMMAInit[i] = false;
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 2; j++) {
                BatchedMatMulInit[i][j] = false;
            }
        }
        for (int i = 0; i < 4; i++) {
            SumInit[i] = false;
        }
        for (int i = 0; i < 3; i++) {
            ConvolveInit[i] = false;
            Conv1dInit[i] = false;
            Conv2dInit[i] = false;
            Conv3dInit[i] = false;
        }
        for (int i = 0; i < 3; i++) {
            ConvolveFullInit[i] = false;
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                BrodcastedAddInit[i][j] = false;
            }
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                BrodcastedSubInit[i][j] = false;
            }
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                BrodcastedMulInit[i][j] = false;
            }
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                BrodcastedDivInit[i][j] = false;
            }
        }
        for (int i = 0; i < 3; i++) {
            SinInit[i] = false;
        }
        for (int i = 0; i < 3; i++) {
            CosInit[i] = false;
        }
        for (int i = 0; i < 3; i++) {
            TanInit[i] = false;
            SqrtInit[i] = false;
            ExpInit[i] = false;
        }
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                AbsInit[i][j] = false;
                LogInit[i][j] = false;
                SinInit_nd[i][j] = false;
                CosInit_nd[i][j] = false;
                TanInit_nd[i][j] = false;
                SqrtInit_nd[i][j] = false;
                ExpInit_nd[i][j] = false;
                ClampInit_nd[i][j] = false;
                TakeInit_nd[i][j] = false;
            }
        }
        for (int i = 0; i < 3; i++) {
            ConvolveInit[i] = false;
            Conv1dInit[i] = false;
            Conv2dInit[i] = false;
            Conv3dInit[i] = false;
        }
        for (int i = 0; i < 4; i++) {
            Concat_2M[i] = false;
        }
        for (int i = 0; i < 6; i++) {
            for (int j = 0; j < 6; j++) {
                for (int k = 0; k < 6; k++) {
                    CopyInplace[i][j][k] = false;
                }
            }
        }
        for (int i = 0; i < 6; i++) {
            for (int k = 0; k < 4; k++) {
                PaddingInit[i][k] = false;
            }
        }
    }
    
    bool hasShader(std::string name) {
        NSString* shaderName = [NSString stringWithUTF8String:name.c_str()];
        return shaderNameToIndex[shaderName] != nil;
    }
    
    id<MTLCommandBuffer> getCommandBuffer() {
        if (!_thread_gCommandBuffer) {
            id<MTLCommandBuffer> buf = [gCommandQueue commandBuffer];
            _thread_gCommandBuffer = (__bridge_retained void*)buf;
        }
        return (__bridge id<MTLCommandBuffer>)_thread_gCommandBuffer;
    }
    
    void setCommandBuffer(id<MTLCommandBuffer> buf) {
        if (_thread_gCommandBuffer) {
            CFRelease(_thread_gCommandBuffer);
        }
        if (buf) {
            _thread_gCommandBuffer = (__bridge_retained void*)buf;
        } else {
            _thread_gCommandBuffer = nullptr;
        }
    }
    void commitCommandBuffer() {
        [getCommandBuffer() commit];
        [getCommandBuffer() waitUntilCompleted];
        setCommandBuffer(nil);
    }
    
    id<MTLComputeCommandEncoder> getCommandEncoder() {
        if (!_thread_gCommandEncoder) {
            id<MTLComputeCommandEncoder> enc = [getCommandBuffer() computeCommandEncoder];
            _thread_gCommandEncoder = (__bridge_retained void*)enc;
        }
        return (__bridge id<MTLComputeCommandEncoder>)_thread_gCommandEncoder;
    }
    
    void setCommandEncoder(id<MTLComputeCommandEncoder> enc) {
        if (_thread_gCommandEncoder) {
            CFRelease(_thread_gCommandEncoder);
        }
        if (enc) {
            _thread_gCommandEncoder = (__bridge_retained void*)enc;
        } else {
            _thread_gCommandEncoder = nullptr;
        }
    }
    
    id<MTLBlitCommandEncoder> getNewBlitCommandEncoder() {
        endCommandEncoding();
        return [getCommandBuffer() blitCommandEncoder];
    }
    
    void endCommandEncoding() {
        if (_thread_gCommandEncoder) {
            id<MTLComputeCommandEncoder> enc = (__bridge id<MTLComputeCommandEncoder>)_thread_gCommandEncoder;
            [enc endEncoding];
            setCommandEncoder(nil);
        }
    }
    
    id<MTLComputeCommandEncoder> getNewCommandEncoder() {
        endCommandEncoding();
        return getCommandEncoder();
    }
    
    id<MTLLibrary> getSourceOfMetalFiles(id<MTLDevice> device) {
        NSError *error = nil;
        NSArray<NSString *> *metalPaths = [[NSBundle mainBundle] pathsForResourcesOfType:@"metal" inDirectory:nil];
        for (NSString *path in metalPaths) {
            NSLog(@"%@", path);
        }
        // Read all .metal files and concatenate
        NSMutableString *allSource = [NSMutableString string];
        for (NSString *path in metalPaths) {
            NSString *source = [NSString stringWithContentsOfFile:path
                                                         encoding:NSUTF8StringEncoding
                                                            error:&error];
            if (!source) {
                NSLog(@"Error reading %@: %@", path, error);
                continue;
            }
            [allSource appendString:source];
            [allSource appendString:@"\n"];
        }

        // Compile with logging enabled
        MTLCompileOptions *options = [MTLCompileOptions new];
        options.enableLogging = YES;  // macOS 14+ / iOS 17+ (debug builds)
        options.libraryType = MTLLibraryTypeExecutable;

        auto libraryLocal = [device newLibraryWithSource:allSource
                                                      options:options
                                                        error:&error];
        if (!libraryLocal) {
            NSLog(@"Error compiling Metal source: %@", error);
        } else {
            NSLog(@"Metal library compiled with logging enabled.");
        }
        return libraryLocal;
    }
    
    id<MTLComputePipelineState> DerivativeAll;
    bool DerivativeAllInit = false;
    
    template<int dims, typename Type>
    std::pair<id<MTLCommandBuffer>, id<MTLComputeCommandEncoder>> create_custom(std::string name, MatrixH<dims, Type> output, MatrixH<dims, Type> inp, size_t ExecutionSize) {
        NSString* shaderName = [NSString stringWithUTF8String:name.c_str()];
        id<MTLComputePipelineState> pipelineState = nil;
        
        // Check if pipeline state already exists
        NSNumber* indexNum = shaderNameToIndex[shaderName];
        if (indexNum != nil) {
            // Pipeline exists, retrieve it
            NSUInteger index = [indexNum unsignedIntegerValue];
            pipelineState = customComputeShader[index];
        } else {
            // Pipeline doesn't exist, create it
            NSError* error = nil;
            id<MTLFunction> kernelFunction = [library newFunctionWithName:shaderName];
            
            if (!kernelFunction) {
                NSLog(@"Failed to find function '%@' in default library", shaderName);
                return std::make_pair(nil, nil);
            }
            
            pipelineState = [metalDevice newComputePipelineStateWithFunction:kernelFunction
                                                                  error:&error];
            
            if (error || !pipelineState) {
                NSLog(@"Failed to create pipeline state for '%@': %@",
                      shaderName, error.localizedDescription);
                return std::make_pair(nil, nil);
            }
            
            // Store the pipeline state
            [customComputeShader addObject:pipelineState];
            NSUInteger newIndex = [customComputeShader count] - 1;
            shaderNameToIndex[shaderName] = @(newIndex);
        }
        
        // Create command buffer

        id<MTLCommandBuffer> commandBuffer = [gCommandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        auto _threadsPerThreadgroup = MTLSizeMake(1, 1, 1);
        auto _dispatchExecutionSize =  MTLSizeMake(ExecutionSize, 1, 1);
        
        [encoder setComputePipelineState:pipelineState];
        
        // Set buffers (assuming MatrixH has a method to get MTLBuffer)
        // Adjust based on your actual MatrixH implementation
        [encoder setBuffer:output.metalBuffer offset:0 atIndex:0];
        [encoder setBuffer:inp.metalBuffer offset:0 atIndex:1];
        [encoder setBytes:output.shape length:dims * sizeof(size_m) atIndex:2];
        [encoder setBytes:output.strides length:dims * sizeof(size_m) atIndex:3];
        [encoder setBytes:inp.shape length:dims * sizeof(size_m) atIndex:4];
        [encoder setBytes:inp.strides length:dims * sizeof(size_m) atIndex:5];
        
//        [encoder dispatchThreads:_dispatchExecutionSize
//                  threadsPerThreadgroup:_threadsPerThreadgroup];
        
        return std::make_pair(commandBuffer, encoder);
    }

    
    void initBlend() {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:@"BlendCompute"];
        blendCompute = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        blendInit = true;
    }
    
    void initInvert() {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:@"InvertImgCompute"];
        invertImgCompute = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        invertInitImg = true;
    }
    
    void initchromaKey() {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:@"ChromaKeyCompute"];
        chromaKeyCompute = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        chromaKeyInit = true;
    }
    
    void initAddImg() {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:@"AddGPU_C"];
        AddImgCompute = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        AddImgInit = true;
    }
    
    void initAddInt() {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:@"AddGPU_I"];
        AddIntCompute = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        AddIntInit = true;
    }
    
    void initAddFloat() {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:@"AddGPU_F"];
        AddFloatCompute = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        AddFloatInit = true;
    }
    
    void initSubImg() {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:@"AddGPU_C"];
        AddImgCompute = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        AddImgInit = true;
    }
    
    void initSubInt() {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:@"SubGPU_I"];
        SubIntCompute = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        SubIntInit = true;
    }
    
    void initSubFloat() {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:@"SubGPU_F"];
        SubFloatCompute = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        SubFloatInit = true;
    }
    void initMulAll() {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:@"MulGPU_All"];
        MulAllCompute = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        MulAllInit = true;
    }
    void initConversionAll() {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:@"ConversionGPU_All"];
        ConversionAll = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        ConversionAllInit = true;
    }
    
    void initTypeCasting(int i, int j) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"TypeCastingGPU_%i_%i", i,j]];
        typeCasting[i][j] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        typeCastingInit[i][j] = true;
        
    }
    
    void initDerivativeAll() {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:@"DerivativeGPU_All"];
        DerivativeAll = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        DerivativeAllInit = true;
    }
    
    void initTransposeAll(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"TransposeGPU_%i", i]];
        TransposeComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        TransposeInit[i] = true;
    }
    
    void initGEMMA_All(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"MatMulGPU_%i", i]];
        GEMMAComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        GEMMAInit[i] = true;
    }
    
    void initBatchedMatMul(int type_idx, int spec_idx) {
        NSError* error = nil;
        NSString* funcName;
        if (spec_idx == 0) {
            funcName = [NSString stringWithFormat:@"BatchedMatMul_3Dgg_%i", type_idx];
        } else {
            funcName = [NSString stringWithFormat:@"BatchedMatMul_NDgg_%i", type_idx];
        }
        id<MTLFunction> func = [library newFunctionWithName:funcName];
        BatchedMatMulComputeState[type_idx][spec_idx] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        BatchedMatMulInit[type_idx][spec_idx] = true;
    }
    
    void initSum_All(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"SumGPU_%i", i]];
        SumComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        SumInit[i] = true;
    }
    
    void initSum_nd(int type_code, int cdims) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"SumGPU_%i_%i", type_code, cdims]];
        SumComputeState_nd[type_code][cdims] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        SumInit_nd[type_code][cdims] = true;
    }
    
    void initMax_nd(int type_code, int cdims) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"MaxGPU_%i_%i", type_code, cdims]];
        MaxComputeState_nd[type_code][cdims] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        MaxInit[type_code][cdims] = true;
    }

    void initMin_nd(int type_code, int cdims) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"MinGPU_%i_%i", type_code, cdims]];
        MinComputeState_nd[type_code][cdims] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        MinInit[type_code][cdims] = true;
    }
    
    void initSin(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"SinGPU_%i", i]];
        SinComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        SinInit[i] = true;
    }

    void initCos(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"CosGPU_%i", i]];
        CosComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        CosInit[i] = true;
    }

    void initTan(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"TanGPU_%i", i]];
        TanComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        TanInit[i] = true;
    }

    void initSqrt_All(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"SqrtGPU_%i", i]];
        SqrtComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        SqrtInit[i] = true;
    }
    
    void initConvolve_All(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"ConvolveGPU_%i", i]];
        ConvolveComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        ConvolveInit[i] = true;
    }

    void initConv1d(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"conv1d_gpu_%i", i]];
        Conv1dComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        Conv1dInit[i] = true;
    }
    
    void initConv2d(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"conv2d_gpu_%i", i]];
        Conv2dComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        Conv2dInit[i] = true;
    }

    void initConv3d(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"conv3d_gpu_%i", i]];
        Conv3dComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        Conv3dInit[i] = true;
    }

    void initConv(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"conv_gpu_%i", i]];
        ConvComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        ConvInit[i] = true;
    }
    
    void initConvolve_FULL(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"ConvolveGPU_FULL_%i", i]];
        ConvolveFullComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        ConvolveFullInit[i] = true;
    }
    
    void initBrodcastedAddInit(int typeCode, int dimSpecialiation) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"BrodcastedAddGPU_%i_%i", typeCode, dimSpecialiation]];
        BrodcastedAddComputeState[typeCode][dimSpecialiation] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        BrodcastedAddInit[typeCode][dimSpecialiation] = true;
    }
    
    void initBrodcastedCrossInit(int typeCode, int dimSpecialiation) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"BrodcastedCrossGPU_%i_%i", typeCode, dimSpecialiation]];
        BrodcastedCrossComputeState[typeCode][dimSpecialiation] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        BrodcastedCrossInit[typeCode][dimSpecialiation] = true;
    }
    
    void initBrodcastedSubInit(int typeCode, int dimSpecialiation) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"BrodcastedSubGPU_%i_%i", typeCode, dimSpecialiation]];
        BrodcastedSubComputeState[typeCode][dimSpecialiation] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        BrodcastedSubInit[typeCode][dimSpecialiation] = true;
    }


    void initBrodcastedMulInit(int typeCode, int dimSpecialiation) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"BrodcastedMulGPU_%i_%i", typeCode, dimSpecialiation]];
        BrodcastedMulComputeState[typeCode][dimSpecialiation] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        BrodcastedMulInit[typeCode][dimSpecialiation] = true;
    }

    void initBrodcastedDivInit(int typeCode, int dimSpecialiation) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"BrodcastedDivGPU_%i_%i", typeCode, dimSpecialiation]];
        BrodcastedDivComputeState[typeCode][dimSpecialiation] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        BrodcastedDivInit[typeCode][dimSpecialiation] = true;
    }
    
    void initBrodcastedMaxInit(int typeCode, int dimSpecialiation) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"BrodcastedMaxGPU_%i_%i", typeCode, dimSpecialiation]];
        BrodcastedMaxComputeState[typeCode][dimSpecialiation] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        BrodcastedMaxInit[typeCode][dimSpecialiation] = true;
    }
    
    void initBrodcastedMinInit(int typeCode, int dimSpecialiation) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"BrodcastedMinGPU_%i_%i", typeCode, dimSpecialiation]];
        BrodcastedMinComputeState[typeCode][dimSpecialiation] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        BrodcastedMinInit[typeCode][dimSpecialiation] = true;
    }
    void initExp(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"ExpGPU_%i", i]];
        ExpComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        ExpInit[i] = true;
    }

    void initAbs(int type_code, int cdims) {
        NSError *error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"AbsGPU_%i_%i", type_code, cdims]];
        AbsComputeState[type_code][cdims] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        AbsInit[type_code][cdims] = true;
        if (error) {
            NSLog(@"Error occurred while compiling abs shader for type %i and cdims %i: %@", type_code, cdims, error);
            std::cout << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
        }
    }
    
    void initLog(int type_code, int cdims) {
        NSError *error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"LogGPU_%i_%i", type_code, cdims]];
        LogComputeState[type_code][cdims] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        LogInit[type_code][cdims] = true;
        if (error) {
            NSLog(@"Error occurred while compiling log shader for type %i and cdims %i: %@", type_code, cdims, error);
            std::cout << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
        }
    }

    void initSin_nd(int type_code, int cdims) {
        NSError *error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"SinGPU_nd_%i_%i", type_code, cdims]];
        SinComputeState_nd[type_code][cdims] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        SinInit_nd[type_code][cdims] = true;
        if (error) std::cout << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
    }
    
    void initClamp_nd(int type_code, int cdims) {
        NSError *error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"ClampGPU_nd_%i_%i", type_code, cdims]];
        ClampComputeState_nd[type_code][cdims] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        ClampInit_nd[type_code][cdims] = true;
        if (error) std::cout << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
    }
    
    void initCos_nd(int type_code, int cdims) {
        NSError *error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"CosGPU_nd_%i_%i", type_code, cdims]];
        CosComputeState_nd[type_code][cdims] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        CosInit_nd[type_code][cdims] = true;
        if (error) std::cout << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
    }
    
    void initTan_nd(int type_code, int cdims) {
        NSError *error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"TanGPU_nd_%i_%i", type_code, cdims]];
        TanComputeState_nd[type_code][cdims] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        TanInit_nd[type_code][cdims] = true;
        if (error) std::cout << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
    }
    
    void initSqrt_nd(int type_code, int cdims) {
        NSError *error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"SqrtGPU_nd_%i_%i", type_code, cdims]];
        SqrtComputeState_nd[type_code][cdims] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        SqrtInit_nd[type_code][cdims] = true;
        if (error) std::cout << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
    }
    
    void initExp_nd(int type_code, int cdims) {
        NSError *error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"ExpGPU_nd_%i_%i", type_code, cdims]];
        ExpComputeState_nd[type_code][cdims] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        ExpInit_nd[type_code][cdims] = true;
        if (error) std::cout << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
    }

    void initTake_nd(int type_code, int cdims) {
        NSError *error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"TakeGPU_nd_%i_%i", type_code, cdims]];
        TakeComputeState_nd[type_code][cdims] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        TakeInit_nd[type_code][cdims] = true;
        if (error) std::cout << "Error: " << [[error localizedDescription] UTF8String] << std::endl;
    }
    
    void initConcat_2M_GPU(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"concatGPU_2M_%i", i]];
        Concat_2M_ComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        Concat_2M[i] = true;
    }
    
    void initCopyInplace(int dstTypeCode, int typeCode, int dimSpecialiation) {
        NSError* error = nil;
        NSLog([NSString stringWithFormat:@"CopyInplaceGPU_%i_%i_%i", dstTypeCode, typeCode, dimSpecialiation]);
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"CopyInplaceGPU_%i_%i_%i", dstTypeCode, typeCode, dimSpecialiation]];
        CopyInplace_ComputeState[dstTypeCode][typeCode][dimSpecialiation] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        CopyInplace[dstTypeCode][typeCode][dimSpecialiation] = true;
        
    }
    void initPadding(int typeCode, int dimSpecialiation) {
        NSError* error = nil;
        NSLog([NSString stringWithFormat:@"PaddingGPU_%i_%i", typeCode, dimSpecialiation]);
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"PaddingGPU_%i_%i", typeCode, dimSpecialiation]];
        Padding_ComputeState[typeCode][dimSpecialiation] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        PaddingInit[typeCode][dimSpecialiation] = true;
        
    }
};

//simd_float4x4 DAMM() {
//    simd_float4 row0 = {1.0f, 0.0f, 0.0f, 0.0f};
//    simd_float4 row1 = {0.0f, 1.0f, 0.0f, 0.0f};
//    simd_float4 row2 = {0.0f, 0.0f, 1.0f, 0.0f};
//    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
//    return simd_matrix(row0, row1, row2, row3);
//}

inline GPUManager GlobalGPUManager = GPUManager();
