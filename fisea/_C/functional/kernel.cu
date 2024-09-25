#include <stdio.h>
#include <iostream>

#include "kernel.cuh"

#include "../handler.cuh"

namespace fisea
{
    template <typename T>
    __global__ void cudaKernelInt_(T *data, size_t size)
    {
        CUDA_KERNEL_LOOP(i, size)
        {
            data[i] = static_cast<int>(data[i]);
        }
    }

    template <typename T>
    __global__ void cudaKernelInt(T *data, size_t size, int *out)
    {
        CUDA_KERNEL_LOOP(i, size)
        {
            out[i] = static_cast<int>(data[i]);
        }
    }

    template <typename T>
    __global__ void cudaKernelFloat_(T *data, size_t size)
    {
        CUDA_KERNEL_LOOP(i, size)
        {
            data[i] = static_cast<float>(data[i]);
        }
    }

    template <typename T>
    __global__ void cudaKernelFloat(T *data, size_t size, float *out)
    {
        CUDA_KERNEL_LOOP(i, size)
        {
            out[i] = static_cast<float>(data[i]);
        }
    }

    // TODO
    template <typename T>
    __global__ void cudaKernelFill(T *data, size_t size, T value)
    {
        CUDA_KERNEL_LOOP(i, size)
        {
            data[i] = value;
        }
    }

    template <typename T>
    void cudaFloat_(T *data, size_t size)
    {
        fisea::cudaKernelFloat_<T><<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size);
    }

    template <typename T>
    void cudaInt(T *data, size_t size, int *out)
    {
        fisea::cudaKernelInt<T><<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size, out);
    }

    template <typename T>
    void cudaFloat(T *data, size_t size, float *out)
    {
        fisea::cudaKernelFloat<T><<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size, out);
    }

    template <typename T>
    void cudaInt_(T *data, size_t size)
    {
        fisea::cudaKernelInt_<T><<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size);
    }

    template <typename T>
    void cudaFill(T *data, size_t size, T value)
    {
        fisea::cudaKernelFill<T><<<CudaGetBlocks(size), kCudaThreadsNum>>>(data, size, value);
    }
    
}

template void fisea::cudaFloat_<int>(int *data, size_t size);
template void fisea::cudaFloat_<float>(float *data, size_t size);

template void fisea::cudaInt<int>(int *data, size_t size, int *out);
template void fisea::cudaInt<float>(float *data, size_t size, int *out);

template void fisea::cudaFloat<int>(int *data, size_t size, float *out);
template void fisea::cudaFloat<float>(float *data, size_t size, float *out);

template void fisea::cudaInt_<int>(int *data, size_t size);
template void fisea::cudaInt_<float>(float *data, size_t size);

template void fisea::cudaFill<int>(int *data, size_t size, int value);
template void fisea::cudaFill<float>(float *data, size_t size, float value);