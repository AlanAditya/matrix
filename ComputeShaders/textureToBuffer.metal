//
//  textureToBuffer.metal
//  AdityaIntelligenceProMax
//
//  Created by Aditya Dudeja on 15/02/26.
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

template <typename Type>
[[kernel]] void TextureToBuffer(device Type* outBuffer [[buffer(0)]], texture2d<Type, access::sample> inTexture [[texture(0)]], constant uint2& targetSize [[buffer(1)]] ,uint2 gid [[thread_position_in_grid]]) {
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = float2(gid.x + 0.5, gid.y + 0.5) / float2(targetSize.x, targetSize.y);
    
    Type depthValue = inTexture.sample(s, uv).r;
    uint flatIndex = (gid.y * targetSize.x) + gid.x;
    outBuffer[flatIndex] = depthValue;
}

// A dead-simple kernel to scale a texture and dump it into a flat tensor buffer
kernel void convertTextureToBuffer(texture2d<half, access::sample> inTexture [[texture(0)]],
                                   device half* outBuffer [[buffer(0)]],
                                   constant uint2& targetSize [[buffer(1)]],
                                   uint2 gridPos [[thread_position_in_grid]])
{
    if (gridPos.x >= targetSize.x || gridPos.y >= targetSize.y) return;

    // Calculate normalized coordinates (0.0 to 1.0) to trigger free hardware scaling
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = float2(gridPos.x + 0.5, gridPos.y + 0.5) / float2(targetSize.x, targetSize.y);

    // Sample the texture (hardware interpolates this instantly)
    half depthValue = inTexture.sample(s, uv).r;

    // Calculate the flat index for your MatrixH buffer
    uint flatIndex = (gridPos.y * targetSize.x) + gridPos.x;

    // Write directly to your tensor's memory!
    outBuffer[flatIndex] = depthValue;
}
