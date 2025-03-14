#include <iostream>
#include <vector>
#include <cstdio>
#include "kernels.h"
#include "memory.h"
#include "utils.h"

// Network parameters (must match training)
const int input_size = 28 * 28;
const int hidden_size = 128;
const int num_classes = 10;

// Load model parameters
void load_model(std::vector<float>& fc1_weights, std::vector<float>& fc1_bias,
               std::vector<float>& fc2_weights, std::vector<float>& fc2_bias) {
    FILE* fp = fopen("mnist_model.bin", "rb");
    if (!fp) {
        std::cerr << "Error: Could not open model file" << std::endl;
        exit(1);
    }
    
    fread(fc1_weights.data(), sizeof(float), fc1_weights.size(), fp);
    fread(fc1_bias.data(), sizeof(float), fc1_bias.size(), fp);
    fread(fc2_weights.data(), sizeof(float), fc2_weights.size(), fp);
    fread(fc2_bias.data(), sizeof(float), fc2_bias.size(), fp);
    fclose(fp);
}

// Load and preprocess a custom image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::vector<float> load_image(const char* filename) {
    int width, height, channels;
    unsigned char* img = stbi_load(filename, &width, &height, &channels, 0);
    if (!img) {
        std::cerr << "Error: Could not load image " << filename << std::endl;
        exit(1);
    }
    
    std::vector<float> processed(28 * 28, 0.0f);
    
    // Simple resizing and conversion to grayscale
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            int src_x = x * width / 28;
            int src_y = y * height / 28;
            
            float pixel = 0.0f;
            if (channels == 1) {
                pixel = img[src_y * width + src_x] / 255.0f;
            } else {
                // Convert RGB to grayscale
                unsigned char* p = img + (src_y * width + src_x) * channels;
                pixel = (0.3f * p[0] + 0.59f * p[1] + 0.11f * p[2]) / 255.0f;
            }
            
            // MNIST has white digits on black background (1.0 = white)
            // If your image has black digits on white, invert:
            pixel = 1.0f - pixel;
            
            processed[y * 28 + x] = pixel;
        }
    }
    
    stbi_image_free(img);
    return processed;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    
    const char* image_path = argv[1];
    
    // Load the image
    std::vector<float> image = load_image(image_path);
    
    // Load the model
    std::vector<float> h_fc1_weights(input_size * hidden_size);
    std::vector<float> h_fc1_bias(hidden_size);
    std::vector<float> h_fc2_weights(hidden_size * num_classes);
    std::vector<float> h_fc2_bias(num_classes);
    load_model(h_fc1_weights, h_fc1_bias, h_fc2_weights, h_fc2_bias);
    
    // Allocate device memory
    float* d_input = static_cast<float*>(cuda_malloc(input_size * sizeof(float)));
    float* d_hidden = static_cast<float*>(cuda_malloc(hidden_size * sizeof(float)));
    float* d_output = static_cast<float*>(cuda_malloc(num_classes * sizeof(float)));
    float* d_fc1_weights = static_cast<float*>(cuda_malloc(input_size * hidden_size * sizeof(float)));
    float* d_fc1_bias = static_cast<float*>(cuda_malloc(hidden_size * sizeof(float)));
    float* d_fc2_weights = static_cast<float*>(cuda_malloc(hidden_size * num_classes * sizeof(float)));
    float* d_fc2_bias = static_cast<float*>(cuda_malloc(num_classes * sizeof(float)));
    
    // Copy data to device
    copy_to_device(d_input, image.data(), input_size * sizeof(float));
    copy_to_device(d_fc1_weights, h_fc1_weights.data(), h_fc1_weights.size() * sizeof(float));
    copy_to_device(d_fc1_bias, h_fc1_bias.data(), h_fc1_bias.size() * sizeof(float));
    copy_to_device(d_fc2_weights, h_fc2_weights.data(), h_fc2_weights.size() * sizeof(float));
    copy_to_device(d_fc2_bias, h_fc2_bias.data(), h_fc2_bias.size() * sizeof(float));
    
    // Forward pass
    fc_forward(d_input, d_fc1_weights, d_fc1_bias, d_hidden, input_size, hidden_size);
    relu_activation(d_hidden, d_hidden, hidden_size);
    fc_forward(d_hidden, d_fc2_weights, d_fc2_bias, d_output, hidden_size, num_classes);
    softmax(d_output, d_output, num_classes);
    
    // Get results
    std::vector<float> output(num_classes);
    copy_to_host(output.data(), d_output, num_classes * sizeof(float));
    
    // Find prediction
    int prediction = argmax(output.data(), num_classes);
    
    // Print results
    std::cout << "Predicted digit: " << prediction << std::endl;
    std::cout << "Confidence scores:" << std::endl;
    for (int i = 0; i < num_classes; ++i) {
        std::cout << "  " << i << ": " << (output[i] * 100.0f) << "%" << std::endl;
    }
    
    // Cleanup
    cuda_free(d_input);
    cuda_free(d_hidden);
    cuda_free(d_output);
    cuda_free(d_fc1_weights);
    cuda_free(d_fc1_bias);
    cuda_free(d_fc2_weights);
    cuda_free(d_fc2_bias);
    
    return 0;
}