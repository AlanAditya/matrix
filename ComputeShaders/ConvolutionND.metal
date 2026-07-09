//
//  ConvolutionND.metal
//  AdityaIntelligenceProMax
//

#include <metal_stdlib>
#include "Utils.h"

using namespace metal;

// -------------------------------------------------------------------------
// Conv1D
// Input Shape:  [Batch, In_L, In_C]
// Kernel Shape: [Out_C, K_L, In_C / groups]
// Output Shape: [Batch, Out_L, Out_C]
// GID: [Batch, X, Out_C]
// -------------------------------------------------------------------------
template <typename T>
kernel void conv1d_gpu(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    device const T* Kernel [[buffer(2)]],
    constant size_m* in_shape [[buffer(3)]],
    constant size_m* kernel_shape [[buffer(4)]],
    constant size_m* out_shape [[buffer(5)]],
    constant int* padding [[buffer(6)]],
    constant int* stride [[buffer(7)]],
    constant int* dilation [[buffer(8)]],
    constant int& groups [[buffer(9)]],
    constant size_m* in_strides [[buffer(10)]],
    constant size_m* kernel_strides [[buffer(11)]],
    constant size_m* out_strides [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]])
{
    int batch_size = in_shape[0];
    int in_l = in_shape[1];
    int in_c = in_shape[2];
    
    int out_c = out_shape[2];
    int out_l = out_shape[1];
    
    int k_l = kernel_shape[1];
    int in_c_per_group = kernel_shape[2];
    
    int pad_l = padding[0];
    int stride_l = stride[0];
    int dilation_l = dilation[0];
    
    
    // Deconstruct 1D Grid ID
    int oc = gid.x;
    int out_x = gid.y;
    int b  = gid.z;
    
    int in_virtually_paddex_x = (out_x) * stride_l + dilation_l * (k_l/2);
    int in_x = in_virtually_paddex_x - pad_l;
    
    int group_idx = oc / (out_c / groups);
    int in_c_start = group_idx * in_c_per_group;
    
    T sum = 0;
    
    for (uint32_t i = 0; i < k_l; i++) {
        // offset is the index of the kernel element relative to COM [-2, -1, 0, 1, 2]
        // gid + offset < 0 means the kerenl element is outside even the padding
        int offset = ((int)i - (k_l / 2)) * dilation_l;
        if (in_x + offset < 0 || in_x + offset >= in_l) {
            sum += 0;
            continue;
        }
        
        for (uint32_t j = 0; j < in_c_per_group; j++) {
            sum += Kernel[oc * kernel_strides[0] + i * kernel_strides[1] + j * kernel_strides[2]] * inMat[b * in_strides[0] + in_strides[1] * (in_x + offset) + (in_c_start + j) * in_strides[2]];
        }

    }
    
    outMat[b * out_strides[0] + out_x * out_strides[1] + oc * out_strides[2]] = sum;
}

// -------------------------------------------------------------------------
// Conv2D
// Input Shape:  [Batch, In_H, In_W, In_C]
// Kernel Shape: [Out_C, K_H, K_W, In_C / groups]
// Output Shape: [Batch, Out_H, Out_W, Out_C]
// GID: [[Batch, Out_H], Out_W, Out_C]
// -------------------------------------------------------------------------
template <typename T>
kernel void conv2d_gpu(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    device const T* Kernel [[buffer(2)]],
    constant size_m* in_shape [[buffer(3)]],
    constant size_m* kernel_shape [[buffer(4)]],
    constant size_m* out_shape [[buffer(5)]],
    constant int2& pad [[buffer(6)]],
    constant int2& stride [[buffer(7)]],
    constant int2& dilation [[buffer(8)]],
    constant int& groups [[buffer(9)]],
    constant size_m* in_strides [[buffer(10)]],
    constant size_m* kernel_strides [[buffer(11)]],
    constant size_m* out_strides [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]])
{
    

    int in_h = in_shape[1];
    int in_w = in_shape[2];
    int out_c = out_shape[3];
    int out_w = out_shape[2];
    int out_h = out_shape[1];
    
    int k_h = kernel_shape[1];
    int k_w = kernel_shape[2];
    int in_c_per_group = kernel_shape[3];
    
    int2 k_shape = int2(k_h, k_w); // x=height, y=width
    // Deconstruct 1D Grid ID
    int oc = gid.x;
    int2 out_index = int2(gid.z % out_h, gid.y); // x=out_h, y=out_w
    int b  = gid.z / out_h;
    
    int group_idx = oc / (out_c / groups);
    int in_c_start = group_idx * in_c_per_group;
    
    int2 in_virtually_padded = out_index * stride + dilation * (k_shape / 2);
    int2 in_index = in_virtually_padded - pad; // x=in_h, y=in_w
    
    int in_base_offset = b * in_strides[0] + in_c_start * in_strides[3];
    int kernel_base_offset = oc * kernel_strides[0];
    
    T sum = 0;
    for (int i = 0; i < k_h; i++) { // x, height, i
        for (int j = 0; j < k_w; j++) { // y, width, j
            
            int2 offset = ( int2(i, j) - k_shape / 2 ) * dilation;
            int2 sample = in_index + offset;
            
            if (sample.x < 0 || sample.x >= in_h || sample.y < 0 || sample.y >= in_w) continue;
            
            int k_spatial_offset = i * kernel_strides[1] + j * kernel_strides[2];
            int in_spatial_offset = sample.x * in_strides[1] + sample.y * in_strides[2];
            
            for (int c = 0; c < in_c_per_group; c++) {
                sum += Kernel[kernel_base_offset + k_spatial_offset + c * kernel_strides[3]] * inMat[in_base_offset + in_spatial_offset + c * in_strides[3]];
            }
        }
    }
    
    // TODO: Write your custom conv2d kernel inner logic here!
    
    outMat[out_strides[0] * b + out_strides[1] * out_index.x + out_strides[2] * out_index.y + out_strides[3] * oc] = sum;
}

// -------------------------------------------------------------------------
// Conv3D
// Input Shape:  [Batch, In_D, In_H, In_W, In_C]
// Kernel Shape: [Out_C, K_D, K_H, K_W, In_C / groups]
// Output Shape: [Batch, Out_D, Out_H, Out_W, Out_C]
// -------------------------------------------------------------------------
template <typename T>
kernel void conv3d_gpu(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    device const T* Kernel [[buffer(2)]],
    constant size_m* in_shape [[buffer(3)]],
    constant size_m* kernel_shape [[buffer(4)]],
    constant size_m* out_shape [[buffer(5)]],
    constant int3& pad [[buffer(6)]],
    constant int3& stride [[buffer(7)]],
    constant int3& dilation [[buffer(8)]],
    constant int& groups [[buffer(9)]],
    constant size_m* in_strides [[buffer(10)]],
    constant size_m* kernel_strides [[buffer(11)]],
    constant size_m* out_strides [[buffer(12)]],
    uint3 gid [[thread_position_in_grid]])
{
    int batch_size = in_shape[0];
    int in_d = in_shape[1];
    int in_h = in_shape[2];
    int in_w = in_shape[3];
    int in_c = in_shape[4];
    
    int out_c = out_shape[4];
    int out_w = out_shape[3];
    int out_h = out_shape[2];
    int out_d = out_shape[1];
    
    int k_d = kernel_shape[1];
    int k_h = kernel_shape[2];
    int k_w = kernel_shape[3];
    int in_c_per_group = kernel_shape[4];
    
    int3 k_shape = int3(k_d, k_h, k_w); // x=depth, y=height, z=width
    
    // Deconstruct 3D Grid ID
    int oc = gid.x;
    
    int rem = gid.z;
    int out_h_idx = rem % out_h;
    rem /= out_h;
    int out_d_idx = rem % out_d;
    int b = rem / out_d;
    
    int3 out_index = int3(out_d_idx, out_h_idx, gid.y); // x=out_d, y=out_h, z=out_w
    
    int group_idx = oc / (out_c / groups);
    int in_c_start = group_idx * in_c_per_group;
    
    int3 in_virtually_padded = out_index * stride + dilation * (k_shape / 2);
    int3 in_index = in_virtually_padded - pad; // x=in_d, y=in_h, z=in_w
    
    int in_base_offset = b * in_strides[0] + in_c_start * in_strides[4];
    int kernel_base_offset = oc * kernel_strides[0];
    
    T sum = 0;
    
    for (int i = 0; i < k_d; i++) { // x, depth, i
        for (int j = 0; j < k_h; j++) { // y, height, j
            for (int k = 0; k < k_w; k++) { // z, width, k
                
                int3 offset = ( int3(i, j, k) - k_shape / 2 ) * dilation;
                int3 sample = in_index + offset;
                
                if (sample.x < 0 || sample.x >= in_d || sample.y < 0 || sample.y >= in_h || sample.z < 0 || sample.z >= in_w) continue;
                
                int k_spatial_offset = i * kernel_strides[1] + j * kernel_strides[2] + k * kernel_strides[3];
                int in_spatial_offset = sample.x * in_strides[1] + sample.y * in_strides[2] + sample.z * in_strides[3];
                
                for (int c = 0; c < in_c_per_group; c++) {
                    sum += Kernel[kernel_base_offset + k_spatial_offset + c * kernel_strides[4]] * inMat[in_base_offset + in_spatial_offset + c * in_strides[4]];
                }
            }
        }
    }
    
    outMat[out_strides[0] * b + out_strides[1] * out_index.x + out_strides[2] * out_index.y + out_strides[3] * out_index.z + out_strides[4] * oc] = sum;
}

instantiate_kernel("conv1d_gpu_0", conv1d_gpu, float);
instantiate_kernel("conv2d_gpu_0", conv2d_gpu, float);
instantiate_kernel("conv3d_gpu_0", conv3d_gpu, float);

// -------------------------------------------------------------------------
// ConvND (Generic)
// Input Shape:  [Batch, S1, S2, ..., SN, In_C]
// Kernel Shape: [Out_C, K1, K2, ..., KN, In_C / groups]
// Output Shape: [Batch, S1', S2', ..., SN', Out_C]
// GID: [Channels, Last Spatical Dim, Batch * Spatical Dims(except last)]
// -------------------------------------------------------------------------
//template <typename T>
//        
//        if (valid) {
//            sum += Kernel[k_gid] * inMat[in_gid];
//        }
//    }
//    outMat[out_gid] = sum;
//    
//    // TODO: Write generic ND Convolution logic here!
//}


template <typename T>
kernel void conv_gpu(
    device T* outMat [[buffer(0)]],
    device const T* inMat [[buffer(1)]],
    device const T* Kernel [[buffer(2)]],
    constant size_m* in_shape [[buffer(3)]],
    constant size_m* kernel_shape [[buffer(4)]],
    constant size_m* out_shape [[buffer(5)]],
    constant int* padding [[buffer(6)]],
    constant int* stride [[buffer(7)]],
    constant int* dilation [[buffer(8)]],
    constant int& groups [[buffer(9)]],
    constant size_m* in_strides [[buffer(10)]],
    constant size_m* kernel_strides [[buffer(11)]],
    constant size_m* out_strides [[buffer(12)]],
    constant int& num_spatial_dims [[buffer(13)]],
    constant int& kernel_dot_size [[buffer(14)]],
    uint3 gid [[thread_position_in_grid]])
{
    
    int out_c = out_shape[num_spatial_dims+1]; // Since total dims = N+2, last index is N+1
    int in_c_per_group = kernel_shape[num_spatial_dims+1];
    
    // Deconstruct 1D Grid ID
    int oc = gid.x;
    int group_idx = oc / (out_c / groups);
    int in_c_start = group_idx * in_c_per_group;
    
    T sum = 0;
    
    int rem = gid.z;
    
    size_m out_index[MAX_TENSOR_RANK];
    size_m in_index[MAX_TENSOR_RANK];
    
    // Asigning Channel
    out_index[num_spatial_dims+1] = gid.x;
    in_index[num_spatial_dims+1] = in_c_start;
    
    // Assigning the trailing most dims
    out_index[num_spatial_dims] = gid.y;
    in_index[num_spatial_dims] = out_index[num_spatial_dims] * stride[num_spatial_dims-1] + dilation[num_spatial_dims-1] * (kernel_shape[num_spatial_dims]/2) - padding[num_spatial_dims-1];
    
    // [Batch, ...., Channel]
    for (int i = num_spatial_dims-1; i >= 1; i--) {
        out_index[i] = rem % out_shape[i];
        in_index[i] = out_index[i] * stride[i-1] + dilation[i-1] * (kernel_shape[i]/2) - padding[i-1];
        rem /= out_shape[i];
    }
    
    int b = rem;
    out_index[0] = b;
    in_index[0] = b;
    
    int out_gid = 0;
    for (int i = 0; i < num_spatial_dims+2; i++) {
        out_gid += out_index[i] * out_strides[i];
    }
    
    for (int i = 0; i < kernel_dot_size; i++) {
        int in_gid = b * in_strides[0];
        int k_gid = oc * kernel_strides[0];
        
        int kernel_idx_along_axis = (i % kernel_shape[num_spatial_dims+1]);
        // navigaing the c_in
        k_gid += kernel_idx_along_axis * kernel_strides[num_spatial_dims+1];
        int rem_k = i / kernel_shape[num_spatial_dims+1];
        
        // Accumulating The channel
        in_gid += (in_c_start + kernel_idx_along_axis) * in_strides[num_spatial_dims+1];
        
        bool valid = true;
        for (int j = num_spatial_dims; j >= 1; j--) {
            kernel_idx_along_axis = rem_k % kernel_shape[j];
            int kernel_offset_along_axis = ( kernel_idx_along_axis - (kernel_shape[j] / 2) ) * dilation[j-1];
            if (in_index[j] + kernel_offset_along_axis < 0 || in_index[j] + kernel_offset_along_axis >= in_shape[j]) {valid = false; break;}
            k_gid += kernel_idx_along_axis * kernel_strides[j];
            in_gid += (in_index[j] + kernel_offset_along_axis) * in_strides[j];
            rem_k /= kernel_shape[j];
        }
        
        if (valid) {
            sum += Kernel[k_gid] * inMat[in_gid];
        }
    }
    outMat[out_gid] = sum;
}
instantiate_kernel("conv_gpu_0", conv_gpu, float);
