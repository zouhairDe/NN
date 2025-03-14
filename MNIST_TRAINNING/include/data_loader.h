#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>

std::vector<int> load_mnist_labels(const std::string& path, int& num_labels);
std::vector<float> load_mnist_images(const std::string& path, int& num_images, int& rows, int& cols);

#endif