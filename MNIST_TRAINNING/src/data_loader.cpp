#include <zlib.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <byteswap.h>
#include "data_loader.h"

// Read 4 bytes from gzip file and swap endianness
int32_t read_int32(gzFile file) {
    int32_t value;
    if (gzread(file, &value, sizeof(value)) != sizeof(value)) {
        throw std::runtime_error("Failed to read from file");
    }
    return bswap_32(value);
}

std::vector<float> load_mnist_images(const std::string& path, int& num_images, int& rows, int& cols) {
    gzFile file = gzopen(path.c_str(), "rb");
    if (!file) throw std::runtime_error("Could not open file: " + path);

    try {
        int magic_number = read_int32(file);
        num_images = read_int32(file);
        rows = read_int32(file);
        cols = read_int32(file);

        if (magic_number != 2051) {
            throw std::runtime_error("Invalid MNIST image file format");
        }

        std::vector<float> images(num_images * rows * cols);
        unsigned char pixel;
        for (size_t i = 0; i < images.size(); ++i) {
            if (gzread(file, &pixel, 1) != 1) {
                throw std::runtime_error("Unexpected end of file");
            }
            images[i] = static_cast<float>(pixel) / 255.0f;
        }

        gzclose(file);
        return images;
    } catch (...) {
        gzclose(file);
        throw;
    }
}

std::vector<int> load_mnist_labels(const std::string& path, int& num_labels) {
    gzFile file = gzopen(path.c_str(), "rb");
    if (!file) throw std::runtime_error("Could not open file: " + path);

    try {
        int magic_number = read_int32(file);
        num_labels = read_int32(file);

        if (magic_number != 2049) {
            throw std::runtime_error("Invalid MNIST label file format");
        }

        std::vector<int> labels(num_labels);
        unsigned char label;
        for (int i = 0; i < num_labels; ++i) {
            if (gzread(file, &label, 1) != 1) {
                throw std::runtime_error("Unexpected end of file");
            }
            labels[i] = static_cast<int>(label);
        }

        gzclose(file);
        return labels;
    } catch (...) {
        gzclose(file);
        throw;
    }
}