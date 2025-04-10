#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include "kernels.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if(err != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void convolution_kernel(const float* input, const float* kernel, float* output,
                                 int input_width, int kernel_size, int output_width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < output_width && col < output_width) {
        float sum = 0.0f;
        for (int i = 0; i < kernel_size; ++i) {
            for (int j = 0; j < kernel_size; ++j) {
                int input_row = row + i;
                int input_col = col + j;
                sum += input[input_row * input_width + input_col] * 
                      kernel[i * kernel_size + j];
            }
        }
        output[row * output_width + col] = sum;
    }
}

__global__ void relu_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = fmaxf(0.0f, input[idx]);
}

__global__ void softmax_kernel(float* input, float* output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; ++i)
        max_val = fmaxf(max_val, input[i]);

    float sum = 0.0f;
    for (int i = 0; i < size; ++i)
        sum += expf(input[i] - max_val);

    for (int i = 0; i < size; ++i)
        output[i] = expf(input[i] - max_val) / sum;
}

__global__ void fc_kernel(const float* input, const float* weights, const float* bias,
                        float* output, int input_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < output_size) {
        float sum = 0.0f;
        for (int i = 0; i < input_size; ++i)
            sum += input[i] * weights[idx * input_size + i];
        output[idx] = sum + bias[idx];
    }
}

// Wrappers
extern "C" {
    void convolution_forward(float* d_input, float* d_kernel, float* d_output,
            int input_width, int kernel_size, int output_width) {
        if (!d_input || !d_kernel || !d_output) {
        printf("Error: Null pointer detected!\n");
        return;
        }

        printf("input_width: %d, kernel_size: %d, output_width: %d\n",
        input_width, kernel_size, output_width);

        dim3 block(16, 16);
        dim3 grid((output_width + block.x - 1) / block.x,
        (output_width + block.y - 1) / block.y);

        printf("Grid: (%d, %d), Block: (%d, %d)\n",
        grid.x, grid.y, block.x, block.y);

        convolution_kernel<<<grid, block>>>(d_input, d_kernel, d_output,
                            input_width, kernel_size, output_width);
        CUDA_CHECK(cudaGetLastError());
    }

    void relu_activation(float* d_input, float* d_output, int size) {
        dim3 block(256);
        dim3 grid((size + block.x - 1) / block.x);
        relu_kernel<<<grid, block>>>(d_input, d_output, size);
        CUDA_CHECK(cudaGetLastError());
    }

    void softmax(float* d_input, float* d_output, int size) {
        softmax_kernel<<<1, 1>>>(d_input, d_output, size);
        CUDA_CHECK(cudaGetLastError());
    }

    void fc_forward(float* d_input, float* d_weights, float* d_bias,
                  float* d_output, int input_size, int output_size) {
        dim3 block(256);
        dim3 grid((output_size + block.x - 1) / block.x);
        fc_kernel<<<grid, block>>>(d_input, d_weights, d_bias, d_output,
                                 input_size, output_size);
        CUDA_CHECK(cudaGetLastError());
    }
}

extern "C" {
    void* cuda_malloc(size_t size) {
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        return ptr;
    }
    
    void cuda_free(void* ptr) {
        CUDA_CHECK(cudaFree(ptr));
    }
    
    void copy_to_device(void* dest, void* src, size_t size) {
        CUDA_CHECK(cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice));
    }
    
    void copy_to_host(void* dest, void* src, size_t size) {
        if (!dest || !src) {
            printf("Error: Null pointer in copy_to_host (dest: %p, src: %p)\n", dest, src);
            return;
        }
        
        // Verify src is device pointer
        cudaPointerAttributes attributes;
        cudaError_t err = cudaPointerGetAttributes(&attributes, src);
        if (err != cudaSuccess) {
            printf("Error: Source pointer is not a valid CUDA pointer\n");
            return;
        }
        

            CUDA_CHECK(cudaDeviceSynchronize()); // Ensure previous operations completed
            CUDA_CHECK(cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost));
    }
}
