//
//  Utils.h
//  WorldOf3D
//
//  Created by Aditya Dudeja on 10/12/25.
//

#ifndef Utils_h
#define Utils_h
#include <vector>
#include <iostream>
#include <chrono>
//#include <simd/simd.h>
@import simd;


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

enum class Execution {
    EncodeAndExecute = 0,
    Encode = 1,
};

enum class ExecutionDevice {
    CPU = 0,
    METAL = 1,
    AUTO = 2,
};

enum class ImgType {
    PNG = 0,
    JPG = 1,
    EXR = 2,
};

enum class ConvMode : uint8_t {
    Full = 0,
    Valid = 1,
    Same  = 2  // optional common mode
};

enum class ConvModeFull : uint8_t {
    Normal = 0,
    Extend = 1,
    Mirror  = 2  // optional common mode
};


enum class EvalType {
    EVAL_AUTO = 0,
    EVAL_INSTANTLY = 1,
    COMPILE_TRACE = 2,
    EXEC_TRACE = 3,
};

enum class MouseEvent {
    Drag = 0,
    Hover = 1,
    Zoom = 2,
    Scroll = 3,
    Tap = 4
};

struct ViewerEvent {
    MouseEvent type;
    simd_float2 normalized_pos;
    simd_float2 delta;
    float zoom_magnitude;
    simd_float3 ray_origin;
    simd_float3 ray_dir;
    simd_float3 prev_ray_origin;
    simd_float3 prev_ray_dir;
};

enum Flags : unsigned int {
    NON_OWNERSHIP_FLAG = 1u << 0,  // Bit 0
    NON_CONTIGUOUS_FLAG = 1u << 1,  // Bit 1
    COMPUTE_GRAPH = 1u << 2, // Bit 2
    NON_ROW_CONTIGUOUS_FLAG = 1u << 3
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

struct CollapsedDims_3 {
    size_m shape[MAX_TENSOR_DIMS];
    size_m stridesA[MAX_TENSOR_DIMS];
    size_m stridesB[MAX_TENSOR_DIMS];
    size_m stridesC[MAX_TENSOR_DIMS];
    uint32_t out_dims = 0;
};


simd_float4x4 Identity();

simd::float4x4 Translation(simd::float3 dPos);

simd::float4x4 RotationZ(float theta);

simd::float4x4 RotationY(float theta);

simd::float4x4 RotationX(float theta);

simd::float4x4 Scale(simd::float3 scale);

//std::pair<std::vector<size_m>, std::vector<size_m>> collapse_dims(const size_m shape[], const size_m strides[], const uint32_t dims, const size_t SIZE_CAP);

CollapsedDims_3 collapse_dims(const size_m shape[], const size_m stridesA[], const size_m stridesB[], const size_m stridesC[], const uint32_t dims, const size_t SIZE_CAP);
CollapsedDims_2 collapse_dims(const size_m shape[], const size_m stridesA[], const size_m stridesB[], const uint32_t dims, const size_t SIZE_CAP);
CollapsedDims collapse_dims(const size_m shape[], const size_m strides[], const uint32_t dims, const size_t SIZE_CAP);
CollapsedDims_3 collapse_dims_matmul(const size_m shape[], const size_m stridesA[], const size_m stridesB[], const size_m stridesC[], const uint32_t dims, const size_t SIZE_CAP);
CollapsedDims_2 collapse_dims_reduces(const size_m shape[], const size_m stridesA[], const size_m stridesB[], const uint32_t dims, const int reduce_axis, const size_t SIZE_CAP, bool keepdims);

CollapsedDims collapse_dims_reduce(const size_m shape[], const size_m strides[], const uint32_t dims, int axis, bool keep_dims, const size_t SIZE_CAP);
CollapsedDims_2 collapse_dims_reduce(const size_m shape[], const size_m stridesO[], const size_m stridesI[], const uint32_t dims, int axis, bool keep_dims, bool outer_collapsed, const size_t SIZE_CAP);
CollapsedDims_3 collapse_dims_reduce( const size_m shape[], const size_m stridesA[], const size_m stridesB[], const size_m stridesC[], const uint32_t dims, int axis, bool keep_dims, const size_t SIZE_CAP);
template <typename T>
void TypedPatternFill(T* destination, const T pattern, uint32_t n) {
    uint32_t patternSize = sizeof(T);
    uint32_t exp = 0;
    uint32_t pO2 = 1;
    while ((1u << (exp + 1)) <= n) {
        ++exp;
    }
    
    destination[0] = pattern;
    for (int i = 1; i < exp+1; i++) {
        memcpy((char*)destination + patternSize * pO2, destination, patternSize*pO2);
        // ByteShift To Multiply By 2
        pO2 <<= 1;
    }
    
    memcpy((char*)destination + patternSize * ((int)pO2), destination, (n - pO2) * patternSize);
}

struct Range {
    long long start;
    long long end;

    // 1. Specific range: R(0, 1)
    constexpr Range(long long s, long long e) : start(s), end(e) {}

    // 2. Full range: R()  (Represents all elements in this dimension, like ':' in Python)
    constexpr Range() : start(0), end(LLONG_MAX) {}

    // Helper to check if it's a full range
    constexpr bool is_all() const {
        return end == LLONG_MAX;
    }
};

struct AxisRange {
    long long start;
    long long end;
    bool is_index;

    // 1. Specific range: R(0, 1)
    constexpr AxisRange(long long s, long long e) : start(s), end(e), is_index(false) {}

    // 2. Full range: R() (Represents all elements in this dimension, like ':' in Python)
    constexpr AxisRange() : start(0), end(LLONG_MAX), is_index(false) {}

    // 3. Single index range
    constexpr AxisRange(long long i) : start(i), end(i + 1), is_index(true) {}

    // Helper to check if it's a full range
    constexpr bool is_all() const {
        return end == LLONG_MAX;
    }
};


void PatternFill(void* destination, const void* pattern, size_t patternSize, uint32_t n);
void append_uint32(uint32_t value, std::vector<uint8_t>& header);

template<typename T>
void append_raw(T value, std::vector<uint8_t>& buffer)
{
    size_t pos = buffer.size();

    buffer.resize(pos + sizeof(T));

    std::memcpy(buffer.data() + pos, &value, sizeof(T));
}

template<typename T>
std::vector<uint8_t> append_raw_out(T value)
{
    std::vector<uint8_t> buffer;
    size_t pos = buffer.size();

    buffer.resize(pos + sizeof(T));

    std::memcpy(buffer.data() + pos, &value, sizeof(T));
    return buffer;
}

template void append_raw<uint32_t>(uint32_t, std::vector<uint8_t>&);
template void append_raw<uint16_t>(uint16_t, std::vector<uint8_t>&);

template<typename T>
std::ostream& operator<<(std::ostream& os,
                         const std::vector<T>& vec)
{
    os << "[";

    for(size_t i = 0; i < vec.size(); ++i)
    {
        os << +vec[i];

        if(i + 1 != vec.size())
        {
            os << ", ";
        }
    }

    os << "]";

    return os;
}
#include <iomanip>


std::ostream& operator<<(std::ostream& os,
                         const std::vector<uint8_t>& vec);


std::ostream& operator<<(std::ostream& os, const simd_float4x4& m);

void write_string(const std::string& s, std::vector<uint8_t>& data);
void write_attr(std::vector<uint8_t>& data,
                const std::string& name,
                const std::string& type_name,
                const std::vector<uint8_t>& value_bytes);
void write_v2f(std::vector<uint8_t>& data, float x, float y);
void write_box2i(std::vector<uint8_t>& data,
                 int32_t xmin,
                 int32_t ymin,
                 int32_t xmax,
                 int32_t ymax);

std::vector<uint8_t> write_box2i_out(
                 int32_t xmin,
                 int32_t ymin,
                 int32_t xmax,
                int32_t ymax);
std::vector<uint8_t> write_v2f_out(
               float x,
                float y);

class Timer {
public:
     Timer() {
        m_startTime = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {
        Stop();
    }
    
    void Stop() {
        auto endPointTime = std::chrono::high_resolution_clock::now();
        auto start = std::chrono::time_point_cast<std::chrono::nanoseconds>(m_startTime).time_since_epoch().count();
        auto end = std::chrono::time_point_cast<std::chrono::nanoseconds>(endPointTime).time_since_epoch().count();
        auto duration = end - start;
        double us = duration * 0.001;
        std::cout << "It took " << duration << " ns " << us << " us " <<  "\n";
    }
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
};

#endif /* Utils_h */
