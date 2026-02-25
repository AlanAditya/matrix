//
//  modTest.mm
//  WorldOf3D
//
//  Created by Aditya Dudeja on 25/11/25.
//
@import Metal;
@import std_private_string_string_fwd;
@import std_string;

using size_m = uint32;
template <int dims, typename Type>
class MatrixH;

class GPUManager {
public:
    id<MTLDevice> metalDevice = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> gCommandQueue = [metalDevice newCommandQueue];
    
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
    
    bool GEMMAInit[3];
    id<MTLComputePipelineState> GEMMAComputeState[3];
    
    bool SumInit[3];
    id<MTLComputePipelineState> SumComputeState[3];
    
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
    
    bool ConvolveInit[3];
    id<MTLComputePipelineState> ConvolveComputeState[3];
    
    bool ConvolveFullInit[3];
    id<MTLComputePipelineState> ConvolveFullComputeState[3];
    
    bool BrodcastedAddInit[4][4];
    id<MTLComputePipelineState> BrodcastedAddComputeState[4][4];
    
    bool BrodcastedSubInit[4][4];
    id<MTLComputePipelineState> BrodcastedSubComputeState[4][4];
    
    bool BrodcastedMulInit[4][4];
    id<MTLComputePipelineState> BrodcastedMulComputeState[4][4];
    
    bool BrodcastedDivInit[4][4];
    id<MTLComputePipelineState> BrodcastedDivComputeState[4][4];
    
    bool Concat_2M[4];
    id<MTLComputePipelineState> Concat_2M_ComputeState[4];
    
    bool CopyInplace[4][4];
    id<MTLComputePipelineState> CopyInplace_ComputeState[4][4];
    
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
        for (int i = 0; i < 3; i++) {
            GEMMAInit[i] = false;
        }
        for (int i = 0; i < 3; i++) {
            SumInit[i] = false;
        }
        for (int i = 0; i < 3; i++) {
            ConvolveInit[i] = false;
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
        }
        for (int i = 0; i < 3; i++) {
            ExpInit[i] = false;
        }
        for (int i = 0; i < 3; i++) {
            SqrtInit[i] = false;
        }
    }
    
    bool hasShader(std::string name) {
        NSString* shaderName = [NSString stringWithUTF8String:name.c_str()];
        return shaderNameToIndex[shaderName] != nil;
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
    
    void initSum_All(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"SumGPU_%i", i]];
        SumComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        SumInit[i] = true;
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
    void initExp(int i) {
        NSError* error = nil;
        id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithFormat:@"ExpGPU_%i", i]];
        ExpComputeState[i] = [metalDevice newComputePipelineStateWithFunction:func error:&error];
        ExpInit[i] = true;
    }
};
