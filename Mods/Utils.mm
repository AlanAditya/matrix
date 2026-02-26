//
//  Utils.mm
//  WorldOf3D
//
//  Created by Aditya Dudeja on 10/12/25.
//


#include "Utils.h"

#include <vector>
#include <__utility/pair.h>
@import simd;

simd_float4x4 Identity() {
    simd_float4 row0 = {1.0f, 0.0f, 0.0f, 0.0f};
    simd_float4 row1 = {0.0f, 1.0f, 0.0f, 0.0f};
    simd_float4 row2 = {0.0f, 0.0f, 1.0f, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 Translation(simd::float3 dPos) {
    
    simd_float4 row0 = {1.0f, 0.0f, 0.0f, 0.0f};
    simd_float4 row1 = {0.0f, 1.0f, 0.0f, 0.0f};
    simd_float4 row2 = {0.0f, 0.0f, 1.0f, 0.0f};
    simd_float4 row3 = {dPos[0], dPos[1], dPos[2], 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 RotationZ(float theta) {
    theta = theta * M_PI / 180;
    float sin = sinf(theta);
    float cos = cosf(theta);
    simd_float4 row0 = {cos, sin, 0.0f, 0.0f};
    simd_float4 row1 = {-sin, cos, 0.0f, 0.0f};
    simd_float4 row2 = {0.0f, 0.0f, 1.0f, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 RotationY(float theta) {
    theta = theta * M_PI / 180;
    float sin = sinf(theta);
    float cos = cosf(theta);
    simd_float4 row0 = {cos, 0.0f, sin, 0.0f};
    simd_float4 row1 = {0.0f, 1.0f, 0.0f, 0.0f};
    simd_float4 row2 = {-sin, 0.0f, cos, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 RotationX(float theta) {
    theta = theta * M_PI / 180;
    float sin = sinf(theta);
    float cos = cosf(theta);
    simd_float4 row0 = {1.0f, 0.0f, 0.0f, 0.0f};
    simd_float4 row1 = {0.0f, cos, -sin, 0.0f};
    simd_float4 row2 = {0.0f, sin, cos, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}

simd::float4x4 Scale(simd::float3 scale) {
    simd_float4 row0 = {scale.x, 0.0f, 0.0f, 0.0f};
    simd_float4 row1 = {0.0f, scale.y, 0.0f, 0.0f};
    simd_float4 row2 = {0.0f, 0.0f, scale.z, 0.0f};
    simd_float4 row3 = {0.0f, 0.0f, 0.0f, 1.0f};
    return simd_matrix(row0, row1, row2, row3);
}


//std::pair<std::vector<size_m>, std::vector<size_m>> collapse_dims_old(const size_m shape[], const size_m strides[], const uint32_t dims, const size_t SIZE_CAP) {
//    std::vector<size_m> new_shape;
//    std::vector<size_m> new_strides;
//
//    new_shape.push_back(shape[0]);
//    new_strides.push_back(strides[0]);
//    for (int i = 1; i < dims; i++) {
//        if (shape[i] == 1) { continue; }
//        if (new_strides.back() != strides[i] * shape[i] || new_shape.back() * shape[i] > SIZE_CAP) {
//            new_shape.push_back(shape[i]);
//            new_strides.push_back(strides[i]);
//        } else {
//            new_shape.back() *= shape[i];
//            new_strides.back() = strides[i];
//        }
//    }
//    return std::make_pair(new_shape, new_strides);
//}
//
//std::pair<std::vector<size_m>, std::vector<size_m>> collapse_dims_old(const size_m shape[], const size_m strides[], const uint32_t dims, const size_t SIZE_CAP) {
//    std::vector<size_m> new_shape;
//    std::vector<size_m> new_strides;
//
//    new_shape.push_back(shape[0]);
//    new_strides.push_back(strides[0]);
//    for (int i = 1; i < dims; i++) {
//        if (shape[i] == 1) { continue; }
//        if (new_strides.back() != strides[i] * shape[i] || new_shape.back() * shape[i] > SIZE_CAP) {
//            new_shape.push_back(shape[i]);
//            new_strides.push_back(strides[i]);
//        } else {
//            new_shape.back() *= shape[i];
//            new_strides.back() = strides[i];
//        }
//    }
//    return std::make_pair(new_shape, new_strides);
//}



CollapsedDims collapse_dims(const size_m shape[], const size_m strides[], const uint32_t dims, const size_t SIZE_CAP) {
    CollapsedDims result;
    if (dims == 0) {
        result.out_dims = 0;
        return result;
    }

    result.shape[0]   = shape[0];
    result.strides[0] = strides[0];
    uint32_t last_index = 0;

    for (int i = 1; i < dims; i++) {
        if (shape[i] == 1) { continue; }
        if (result.strides[last_index] != strides[i] * shape[i] || result.shape[last_index] * shape[i] > SIZE_CAP) {
            last_index++;
            result.shape[last_index]   = shape[i];
            result.strides[last_index] = strides[i];
        } else {
            result.shape[last_index] *= shape[i];
            result.strides[last_index] = strides[i];
        }
    }
    result.out_dims = last_index+1;
    return result;
}

CollapsedDims_2 collapse_dims(const size_m shape[], const size_m stridesA[], const size_m stridesB[], const uint32_t dims, const size_t SIZE_CAP) {
    CollapsedDims_2 result;
    if (dims == 0) {
        result.out_dims = 0;
        return result;
    }

    result.shape[0]   = shape[0];
    result.stridesA[0] = stridesA[0];
    result.stridesB[0] = stridesB[0];
    uint32_t last_index = 0;

    for (int i = 1; i < dims; i++) {
        if (shape[i] == 1) { continue; }
        if (result.stridesA[last_index] != stridesA[i] * shape[i] || result.stridesB[last_index] != stridesB[i] * shape[i] || result.shape[last_index] * shape[i] > SIZE_CAP) {
            last_index++;
            result.shape[last_index]   = shape[i];
            result.stridesA[last_index] = stridesA[i];
            result.stridesB[last_index] = stridesB[i];
        } else {
            result.shape[last_index] *= shape[i];
            result.stridesA[last_index] = stridesA[i];
            result.stridesB[last_index] = stridesB[i];
        }
    }
    result.out_dims = last_index+1;
    return result;
}
