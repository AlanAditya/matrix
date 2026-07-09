# Adding New Matrix Generators

A **Generator** is a static function that creates a new matrix populated with values based on specific mathematical or randomized logic (e.g., `zeros`, `ones`, `gaussian`, `linespace`).

This document outlines the standard pattern for implementing new generators in the engine.

## 1. Defining the Function

Generators should be implemented as `static` methods within the `matrix` struct in `matrix.h`.
They take a shape (typically `std::initializer_list<size_m>`) and any generation-specific parameters.

```cpp
static matrix my_generator(std::initializer_list<size_m> shapeI, float param1) {
    // 1. Initialize output matrix with dimensions and type
    matrix output((uint32_t)shapeI.size(), dtype::Float);
    
    // 2. Set the shape
    memcpy(output.shape(), shapeI.begin(), output.dims * sizeof(size_m));
    
    // 3. Calculate strides and total size
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);
    
    // 4. Allocate buffer memory
    output.buffer = new uint8_t[output.total_size * dtype_size(dtype::Float)];
    
    // 5. Build Metal buffer for GPU operations if size is large enough
    if (output.total_size > 10) {
        output.buildMetalBuffer();
    }
    
    // 6. Populate the buffer using your custom logic
    float* buf = (float*)output.buffer;
    for (size_t i = 0; i < output.total_size; ++i) {
        buf[i] = /* your logic here using param1 */;
    }
    
    return output;
}
```

## 2. Key Requirements

- **Type Flexibility:** If the generator can support multiple types (like `zeros` and `ones`), pass a `dtype type = dtype::Float` parameter and use `dispatch_type` to fill the buffer.
- **Metal Buffers:** Always call `output.buildMetalBuffer()` if `output.total_size > 10` so that subsequent operations on the generated matrix can immediately run on the GPU without synchronization issues.
- **Memory Allocation:** Ensure `output.buffer` is correctly allocated using `new uint8_t[total_size * dtype_size(type)]` before populating it.

## 3. Example: Gaussian Generator

```cpp
static matrix gaussian(std::initializer_list<size_m> shapeI, float std_dev = 1.0f, bool normalize = true) {
    matrix output((uint32_t)shapeI.size(), dtype::Float);
    memcpy(output.shape(), shapeI.begin(), output.dims * sizeof(size_m));
    output.calcStrides();
    output.total_size = output.accumul(0, output.dims);
    output.buffer = new uint8_t[output.total_size * dtype_size(dtype::Float)];
    if (output.total_size > 10) {
        output.buildMetalBuffer();
    }
    
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
    
    return output;
}
```
