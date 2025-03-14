#ifndef UTILS_H
#define UTILS_H

float cross_entropy_loss(const float* predictions, const int* labels, int batch_size, int num_classes);
int argmax(const float* output, int size);

#endif