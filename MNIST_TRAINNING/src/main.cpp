#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include "data_loader.h"
#include "kernels.h"
#include "memory.h"
#include "utils.h"

// Network parameters
const int input_size = 28 * 28;  // MNIST image size
const int hidden_size = 128;
const int num_classes = 10;
const int batch_size = 100;
const float learning_rate = 0.01f;

// Initialize weights and biases
void initialize_weights(float* weights, int size) {
    for (int i = 0; i < size; ++i) {
        weights[i] = (rand() / (float)RAND_MAX) * 0.1f;
    }
}

int main() {
    std::srand(std::time(0));

    // Load MNIST data
    int num_images, rows, cols, num_labels;
    std::vector<float> images = load_mnist_images("data/train-images-idx3-ubyte.gz", num_images, rows, cols);
    std::vector<int> labels = load_mnist_labels("data/train-labels-idx1-ubyte.gz", num_labels);

    // Allocate device memory
    float* d_input = static_cast<float*>(cuda_malloc(batch_size * input_size * sizeof(float)));
    float* d_hidden = static_cast<float*>(cuda_malloc(hidden_size * sizeof(float)));
    float* d_output = static_cast<float*>(cuda_malloc(num_classes * sizeof(float)));

    // Initialize weights and biases
    float* fc1_weights = static_cast<float*>(cuda_malloc(input_size * hidden_size * sizeof(float)));
    float* fc1_bias = static_cast<float*>(cuda_malloc(hidden_size * sizeof(float)));
    float* fc2_weights = static_cast<float*>(cuda_malloc(hidden_size * num_classes * sizeof(float)));
    float* fc2_bias = static_cast<float*>(cuda_malloc(num_classes * sizeof(float)));

    // Temporary host buffers for initialization
    std::vector<float> h_fc1_weights(input_size * hidden_size);
    std::vector<float> h_fc1_bias(hidden_size);
    std::vector<float> h_fc2_weights(hidden_size * num_classes);
    std::vector<float> h_fc2_bias(num_classes);

    initialize_weights(h_fc1_weights.data(), h_fc1_weights.size());
    initialize_weights(h_fc1_bias.data(), h_fc1_bias.size());
    initialize_weights(h_fc2_weights.data(), h_fc2_weights.size());
    initialize_weights(h_fc2_bias.data(), h_fc2_bias.size());

    // Copy initialized weights to device
    copy_to_device(fc1_weights, h_fc1_weights.data(), h_fc1_weights.size() * sizeof(float));
    copy_to_device(fc1_bias, h_fc1_bias.data(), h_fc1_bias.size() * sizeof(float));
    copy_to_device(fc2_weights, h_fc2_weights.data(), h_fc2_weights.size() * sizeof(float));
    copy_to_device(fc2_bias, h_fc2_bias.data(), h_fc2_bias.size() * sizeof(float));

    // Training loop
    for (int epoch = 0; epoch < 10; ++epoch) {
        float total_loss = 0.0f;
        int correct = 0;

        for (int i = 0; i < num_images; i += batch_size) {
            // Copy batch to device
            copy_to_device(d_input, &images[i * input_size], batch_size * input_size * sizeof(float));

            // Forward pass
            // Layer 1: Fully connected + ReLU
            fc_forward(d_input, fc1_weights, fc1_bias, d_hidden, input_size, hidden_size);
            relu_activation(d_hidden, d_hidden, hidden_size);

            // Layer 2: Fully connected + Softmax
            fc_forward(d_hidden, fc2_weights, fc2_bias, d_output, hidden_size, num_classes);
            softmax(d_output, d_output, num_classes);

            // Copy results to host
            std::vector<float> host_output(num_classes * batch_size);
            printf("Attempting to copy %zu bytes from device (%p) to host (%p)\n", 
                num_classes * batch_size * sizeof(float), d_output, host_output.data());
            copy_to_host(host_output.data(), d_output, num_classes * sizeof(float));

            // Calculate loss and accuracy
            total_loss += cross_entropy_loss(host_output.data(), &labels[i], batch_size, num_classes);
            for (int j = 0; j < batch_size; ++j) {
                int pred = argmax(&host_output[j * num_classes], num_classes);
                if (pred == labels[i + j]) correct++;
            }
        }

        std::cout << "Epoch " << epoch + 1 
                  << " Loss: " << total_loss / (num_images / batch_size)
                  << " Accuracy: " << (100.0f * correct / num_images) << "%\n";

        // Copying model parameters from device to host
        copy_to_host(h_fc1_weights.data(), fc1_weights, h_fc1_weights.size() * sizeof(float));
        copy_to_host(h_fc1_bias.data(), fc1_bias, h_fc1_bias.size() * sizeof(float));
        copy_to_host(h_fc2_weights.data(), fc2_weights, h_fc2_weights.size() * sizeof(float));
        copy_to_host(h_fc2_bias.data(), fc2_bias, h_fc2_bias.size() * sizeof(float));

        // Write to file
        FILE* fp = fopen("mnist_model.bin", "wb");
        fwrite(h_fc1_weights.data(), sizeof(float), h_fc1_weights.size(), fp);
        fwrite(h_fc1_bias.data(), sizeof(float), h_fc1_bias.size(), fp);
        fwrite(h_fc2_weights.data(), sizeof(float), h_fc2_weights.size(), fp);
        fwrite(h_fc2_bias.data(), sizeof(float), h_fc2_bias.size(), fp);
        fclose(fp);
        printf("Model saved to mnist_model.bin\n");
    }

    // Cleanup
    cuda_free(d_input);
    cuda_free(d_hidden);
    cuda_free(d_output);
    cuda_free(fc1_weights);
    cuda_free(fc1_bias);
    cuda_free(fc2_weights);
    cuda_free(fc2_bias);

    return 0;
}