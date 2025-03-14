#ifndef MEMORY_H
#define MEMORY_H

#include <cstddef>

extern "C" {
    void* cuda_malloc(size_t size);
    void cuda_free(void* ptr);
    void copy_to_device(void* dest, void* src, size_t size);
    void copy_to_host(void* dest, void* src, size_t size);
}

#endif