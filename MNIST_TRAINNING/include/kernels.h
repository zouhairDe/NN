#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>

extern "C" {
    void convolution_forward(float* input, float* kernel, float* output,
                           int input_width, int kernel_size, int output_width);
    void relu_activation(float* input, float* output, int size);
    void softmax(float* input, float* output, int size);
    void fc_forward(float* input, float* weights, float* bias,
                  float* output, int input_size, int output_size);
}

#endif