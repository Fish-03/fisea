#pragma once

#ifdef USE_CUDA

namespace fisea
{
    // template <typename T>
    // __global__ void cudaKernelInt_(T *data, size_t size);
    // template <typename T>
    // __global__ void cudaKernelFloat_(T *data, size_t size);
    // template <typename T>
    // __global__ void cudaKernelInt(T *data, size_t size, int *out);
    // template <typename T>
    // __global__ void cudaKernelFloat(T *data, size_t size, float *out);
    // template <typename T>
    // __global__ void cudaKernelFill(T *data, size_t size, T value);

    template <typename T>
    void cudaInt_(T *data, size_t size);
    template <typename T>
    void cudaFloat_(T *data, size_t size);
    template <typename T>
    void cudaInt(T *data, size_t size, int *out);
    template <typename T>
    void cudaFloat(T *data, size_t size, float *out);
    template <typename T>
    void cudaFill(T *data, size_t size, T value);

}

#endif