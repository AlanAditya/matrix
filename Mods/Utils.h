//
//  Utils.h
//  WorldOf3D
//
//  Created by Aditya Dudeja on 10/12/25.
//

#ifndef Utils_h
#define Utils_h

@import simd;
#include <vector>

struct Vertex3D {
    simd_float3 position;
    simd_float4 colour;
    simd_float2 textureCoordinates;
    simd_float3 normal ;
};
struct Point3D {
    simd_float3 position;
    simd_float4 colour;
};

enum class RenderPipelineType {
    Predefined = 0,
    Custom = 1,
};

enum class PredefinedRenderPipelineState{
    Mesh = 0,
    PointCloud = 1,
    Billboard = 2,
    GaussianSplat = 3,
};
using size_m = uint32_t;
// Define a sensible maximum rank for your library (e.g., 8 or 16)
constexpr uint32_t MAX_TENSOR_DIMS = 8;
// A lightweight struct to hold the results on the stack
struct CollapsedDims {
    size_m shape[MAX_TENSOR_DIMS];
    size_m strides[MAX_TENSOR_DIMS];
    uint32_t out_dims = 0;
};

struct CollapsedDims_2 {
    size_m shape[MAX_TENSOR_DIMS];
    size_m stridesA[MAX_TENSOR_DIMS];
    size_m stridesB[MAX_TENSOR_DIMS];
    uint32_t out_dims = 0;
};


simd_float4x4 Identity();

simd::float4x4 Translation(simd::float3 dPos);

simd::float4x4 RotationZ(float theta);

simd::float4x4 RotationY(float theta);

simd::float4x4 RotationX(float theta);

simd::float4x4 Scale(simd::float3 scale);

//std::pair<std::vector<size_m>, std::vector<size_m>> collapse_dims(const size_m shape[], const size_m strides[], const uint32_t dims, const size_t SIZE_CAP);

CollapsedDims_2 collapse_dims(const size_m shape[], const size_m stridesA[], const size_m stridesB[], const uint32_t dims, const size_t SIZE_CAP);
CollapsedDims collapse_dims(const size_m shape[], const size_m strides[], const uint32_t dims, const size_t SIZE_CAP);

#endif /* Utils_h */
