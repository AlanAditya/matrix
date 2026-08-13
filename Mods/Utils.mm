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

CollapsedDims_3 collapse_dims(const size_m shape[], const size_m stridesA[], const size_m stridesB[], const size_m stridesC[], const uint32_t dims, const size_t SIZE_CAP) {
    CollapsedDims_3 result;
    if (dims == 0) {
        result.out_dims = 0;
        return result;
    }

    result.shape[0]   = shape[0];
    result.stridesA[0] = stridesA[0];
    result.stridesB[0] = stridesB[0];
    result.stridesC[0] = stridesC[0];
    uint32_t last_index = 0;

    for (int i = 1; i < dims; i++) {
        if (shape[i] == 1) { continue; }
        if (result.stridesA[last_index] != stridesA[i] * shape[i] ||
            result.stridesB[last_index] != stridesB[i] * shape[i] ||
            result.stridesC[last_index] != stridesC[i] * shape[i] ||
            result.shape[last_index] * shape[i] > SIZE_CAP) {
            last_index++;
            result.shape[last_index]   = shape[i];
            result.stridesA[last_index] = stridesA[i];
            result.stridesB[last_index] = stridesB[i];
            result.stridesC[last_index] = stridesC[i];
        } else {
            result.shape[last_index] *= shape[i];
            result.stridesA[last_index] = stridesA[i];
            result.stridesB[last_index] = stridesB[i];
            result.stridesC[last_index] = stridesC[i];
        }
    }
    result.out_dims = last_index+1;
    return result;
}

CollapsedDims_3 collapse_dims_matmul(const size_m shape[], const size_m stridesA[], const size_m stridesB[], const size_m stridesC[], const uint32_t dims, const size_t SIZE_CAP) {
    CollapsedDims_3 result;
    // shapeA => [....., M, K]
    // shapeB from B.T() => [....., N, K]
    // shape => [......, M, N]
    result.shape[0]   = shape[0];
    result.stridesA[0] = stridesA[0];
    result.stridesB[0] = stridesB[0];
    result.stridesC[0] = stridesC[0];
    uint32_t last_index = 0;
    // LOGIC IS FOR CONTIGUITY AT AN AXIS WE WANT stride[axis] = stride[axis+1] * shape[axis+1]
    // whenevr two axis are collapsed into one the strides of the inner axis are taken
    // THIS IS BATCHED MATMUL SO WE CAN COLLAPSE ALL DIMS EXCEPT THE LAST TWO
    for (int i = 1; i < dims-2; i++) {
        if (shape[i] == 1) { continue; }
        if (result.stridesA[last_index] != stridesA[i] * shape[i] ||
            result.stridesB[last_index] != stridesB[i] * shape[i] ||
            result.stridesC[last_index] != stridesC[i] * shape[i] ||
            result.shape[last_index] * shape[i] > SIZE_CAP) {
            last_index++;
            result.shape[last_index]   = shape[i];
            result.stridesA[last_index] = stridesA[i];
            result.stridesB[last_index] = stridesB[i];
            result.stridesC[last_index] = stridesC[i];
        } else {
            result.shape[last_index] *= shape[i];
            result.stridesA[last_index] = stridesA[i];
            result.stridesB[last_index] = stridesB[i];
            result.stridesC[last_index] = stridesC[i];
        }
    }

    result.out_dims = last_index+1 + 2;

    // shape => [......, M, N]
    result.shape[result.out_dims-1] = shape[dims-1];
    result.shape[result.out_dims-2] = shape[dims-2];

    // SINCE WE ONLY COLLAPSE AXIS EXCEPT LAST TWO COPY THE INNER TWO STRIDES AS IT IS
    result.stridesC[result.out_dims-2] = stridesC[dims-2];
    result.stridesC[result.out_dims-1] = stridesC[dims-1];


    result.stridesA[result.out_dims-2] = stridesA[dims-2];
    result.stridesA[result.out_dims-1] = stridesA[dims-1];
    result.stridesB[result.out_dims-2] = stridesB[dims-2];
    result.stridesB[result.out_dims-1] = stridesB[dims-1];
    return result;
}

CollapsedDims_2 collapse_dims_reduces(const size_m shape[], const size_m stridesA[], const size_m stridesB[], const uint32_t dims, const int reduce_axis, const size_t SIZE_CAP, bool keepdims) {
    size_m reduced_shape[32];
    size_m reduced_stridesA[32];
    size_m reduced_stridesB[32];
    int idx = 0;
    for (int i = 0; i < dims; i++) {
        if (i != reduce_axis) {
            reduced_shape[idx] = shape[i];
            reduced_stridesA[idx] = keepdims ? stridesA[i] : stridesA[idx];
            reduced_stridesB[idx] = stridesB[i];
            idx++;
        }
    }
    return collapse_dims(reduced_shape, reduced_stridesA, reduced_stridesB, idx, SIZE_CAP);
}

CollapsedDims collapse_dims_reduce(const size_m shape[], const size_m stridesO[], const uint32_t dims, int axis, bool keep_dims, const size_t SIZE_CAP) {
    CollapsedDims result;
    result.out_dims = 0;

    if (dims == 0) {
        return result;
    }

    int last_index = -1;
    bool prevent_collapse = false; // Forces the next dimension to start a new slot

    for (int i = 0; i < dims; i++) {
        // 1. Handle the reduction axis explicitly
        if (i == axis) {
            if (keep_dims) {
                last_index++;
                result.shape[last_index] = shape[i];
                result.strides[last_index] = stridesO[i];
            }
            // Even if keep_dims is false, this axis marks a hard split.
            // The next valid dimension MUST start a new slot and cannot merge backwards.
            prevent_collapse = true;
            continue;
        }

        // 2. Safely skip unit dimensions that aren't the reduction axis
        if (shape[i] == 1) {
            continue;
        }
        // 3. Try to collapse into the previous dimension if allowed
        if (!prevent_collapse &&
            last_index >= 0 &&
            result.strides[last_index] == stridesO[i] * shape[i] &&
            result.shape[last_index] * shape[i] <= SIZE_CAP) {

            result.shape[last_index] *= shape[i];
            result.strides[last_index] = stridesO[i]; // Inner stride takes over
        }
        // 4. Otherwise, start a new uncollapsed dimension slot
        else {
            last_index++;
            result.shape[last_index] = shape[i];
            result.strides[last_index] = stridesO[i]; // Inner stride takes over
            prevent_collapse = false; // Reset the flag once we've established a new slot
        }
    }

    result.out_dims = (uint32_t)(last_index + 1);
    return result;
}

CollapsedDims_2 collapse_dims_reduce(const size_m shape[], const size_m stridesO[], const size_m stridesI[], const uint32_t dims, int axis, bool keep_dims, bool outer_collapsed, const size_t SIZE_CAP) {
    CollapsedDims_2 result;
    result.out_dims = 0;

    if (dims == 0) {
        return result;
    }

    int last_index = -1;
    bool prevent_collapse = false; // Forces the next dimension to start a new slot

    for (int i = 0; i < dims; i++) {
        // 1. Handle the reduction axis explicitly
        if (i == axis) {
            if (keep_dims) {
                last_index++;
                result.shape[last_index] = shape[i];
                result.stridesA[last_index] = stridesO[i];
                result.stridesB[last_index] = stridesI[i];
            }
            // Even if keep_dims is false, this axis marks a hard split.
            // The next valid dimension MUST start a new slot and cannot merge backwards.
            prevent_collapse = true;
            continue;
        }

        // 2. Safely skip unit dimensions that aren't the reduction axis
        if (shape[i] == 1) {
            continue;
        }
        int out_idx = i;
        if (outer_collapsed && axis < i) out_idx-=1;
        // 3. Try to collapse into the previous dimension if allowed
        if (!prevent_collapse &&
            last_index >= 0 &&
            result.stridesA[last_index] == stridesO[out_idx] * shape[i] &&
            result.stridesB[last_index] == stridesI[i] * shape[i] &&
            result.shape[last_index] * shape[i] <= SIZE_CAP) {

            result.shape[last_index] *= shape[i];
            result.stridesA[last_index] = stridesO[out_idx]; // Inner stride takes over
            result.stridesB[last_index] = stridesI[i];
            }
        // 4. Otherwise, start a new uncollapsed dimension slot
        else {
            last_index++;
            result.shape[last_index] = shape[i];
            result.stridesA[last_index] = stridesO[out_idx]; // Inner stride takes over
            result.stridesB[last_index] = stridesI[i];
            prevent_collapse = false; // Reset the flag once we've established a new slot
        }
    }

    result.out_dims = (uint32_t)(last_index + 1);
    return result;
}

CollapsedDims_3 collapse_dims_reduce(const size_m shape[], const size_m stridesO[], const size_m stridesB[], const size_m stridesC[], const uint32_t dims, int axis, bool keep_dims, const size_t SIZE_CAP) {
    CollapsedDims_3 result;
    result.out_dims = 0;

    if (dims == 0) {
        return result;
    }

    int last_index = -1;
    bool prevent_collapse = false; // Forces the next dimension to start a new slot

    for (int i = 0; i < dims; i++) {
        // 1. Handle the reduction axis explicitly
        if (i == axis) {
            if (keep_dims) {
                last_index++;
                result.shape[last_index] = shape[i];
                result.stridesA[last_index] = stridesO[i];
                result.stridesB[last_index] = stridesB[i];
                result.stridesC[last_index] = stridesC[i];
            }
            // Even if keep_dims is false, this axis marks a hard split.
            // The next valid dimension MUST start a new slot and cannot merge backwards.
            prevent_collapse = true;
            continue;
        }

        // 2. Safely skip unit dimensions that aren't the reduction axis
        if (shape[i] == 1) {
            continue;
        }
        // 3. Try to collapse into the previous dimension if allowed
        if (!prevent_collapse &&
            last_index >= 0 &&
            result.stridesA[last_index] == stridesO[i] * shape[i] &&
            result.stridesB[last_index] == stridesB[i] * shape[i] &&
            result.stridesC[last_index] == stridesC[i] * shape[i] &&
            result.shape[last_index] * shape[i] <= SIZE_CAP) {

            result.shape[last_index] *= shape[i];
            result.stridesA[last_index] = stridesO[i]; // Inner stride takes over
            result.stridesB[last_index] = stridesB[i];
            result.stridesC[last_index] = stridesC[i];
        }
        // 4. Otherwise, start a new uncollapsed dimension slot
        else {
            last_index++;
            result.shape[last_index] = shape[i];
            result.stridesA[last_index] = stridesO[i]; // Inner stride takes over
            result.stridesB[last_index] = stridesB[i];
            result.stridesC[last_index] = stridesC[i];
            prevent_collapse = false; // Reset the flag once we've established a new slot
        }
    }

    result.out_dims = (uint32_t)(last_index + 1);
    return result;
}



void PatternFill(void* destination, const void* pattern, size_t patternSize, uint32_t n) {
    uint32_t exp = 0;
    uint32_t pO2 = 1;
    while ((1u << (exp + 1)) <= n) {
        ++exp;
    }
    
    memcpy(destination, pattern, patternSize);
    for (int i = 1; i < exp+1; i++) {
        memcpy((char*)destination + patternSize * pO2, destination, patternSize*pO2);
        // ByteShift To Multiply By 2
        pO2 <<= 1;
    }
    
    memcpy((char*)destination + patternSize * ((int)pO2), destination, (n - pO2) * patternSize);
}

void append_uint32(uint32_t value, std::vector<uint8_t>& header)
{
    uint8_t* ptr = reinterpret_cast<uint8_t*>(&value);

    header.insert(header.end(), ptr, ptr + sizeof(uint32_t));
};


void write_string(const std::string& s, std::vector<uint8_t>& data) {
    size_t old_size = data.size();

    data.resize(old_size + s.size() + 1);

    std::memcpy(data.data() + old_size,
                s.data(),
                s.size());

    data.back() = '\0';
}

void write_attr(std::vector<uint8_t>& data,
                const std::string& name,
                const std::string& type_name,
                const std::vector<uint8_t>& value_bytes)
{
    write_string(name, data);

    write_string(type_name, data);

    append_uint32(
        static_cast<uint32_t>(value_bytes.size()),
        data
    );

    data.insert(data.end(),
                value_bytes.begin(),
                value_bytes.end());
}

void write_box2i(std::vector<uint8_t>& data,
                 int32_t xmin,
                 int32_t ymin,
                 int32_t xmax,
                 int32_t ymax)
{
    data.reserve(data.size() + 16);

    append_raw(xmin, data);
    append_raw(ymin, data);
    append_raw(xmax, data);
    append_raw(ymax, data);
}

std::vector<uint8_t> write_box2i_out(
                 int32_t xmin,
                 int32_t ymin,
                 int32_t xmax,
                 int32_t ymax)
{
    std::vector<uint8_t> data;
    data.reserve(data.size() + 16);

    append_raw(xmin, data);
    append_raw(ymin, data);
    append_raw(xmax, data);
    append_raw(ymax, data);
    return data;
}

void write_v2f(std::vector<uint8_t>& data,
               float x,
               float y)
{
    data.reserve(data.size() + 8);

    append_raw(x, data);
    append_raw(y, data);
}

std::vector<uint8_t> write_v2f_out(
               float x,
               float y)
{
    std::vector<uint8_t> data;
    data.reserve(data.size() + 8);

    append_raw(x, data);
    append_raw(y, data);
    return data;
}

std::ostream& operator<<(std::ostream& os, const std::vector<uint8_t>& vec) {
    os << "bytearray(b'";

    for(uint8_t byte : vec) {
        os << "\\x"
           << std::hex
           << std::setw(2)
           << std::setfill('0')
           << static_cast<int>(byte);
    }

    os << "')";

    os << std::dec;

    return os;
}

std::ostream& operator<<(std::ostream& os, const simd_float4x4& m) {
    // simd_float4x4 is column-major: m.columns[col][row]
    // print as a readable row-major grid
    os << "simd_float4x4(\n";
    for (int row = 0; row < 4; ++row) {
        os << "  [";
        for (int col = 0; col < 4; ++col) {
            os << m.columns[col][row];
            if (col < 3) os << ", ";
        }
        os << "]\n";
    }
    os << ")";
    return os;
}