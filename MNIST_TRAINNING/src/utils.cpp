#include "utils.h"
#include <cmath>
#include <algorithm>

float cross_entropy_loss(const float* predictions, const int* labels, int batch_size, int num_classes) {
    float loss = 0.0f;
    for (int i = 0; i < batch_size; ++i) {
        int label = labels[i];
        loss += -logf(predictions[i * num_classes + label] + 1e-8f);
    }
    return loss / batch_size;
}

int argmax(const float* output, int size) {
    return std::max_element(output, output + size) - output;
}