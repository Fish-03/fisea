
#include <stdio.h>
#include <iostream>

#include "kernel.cuh"
#include "../handler.cuh"

namespace fisea
{
    template <typename T>
    __global__ void _to_int_kernel_(T data, size_t size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size)
            data[i] = static_cast<int>(data[i]);
    }

    template <typename T>
    __global__ void _to_int_kernel(T data, size_t size, int *out)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size)
        {
            out[i] = static_cast<int>(data[i]);
        }
    }

    template <typename T>
    __global__ void _to_float_kernel_(T data, size_t size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size)
        {
            data[i] = static_cast<float>(data[i]);
        }
    }

    template <typename T>
    __global__ void _to_float_kernel(T data, size_t size, float *out)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) 
        {
            out[i] = static_cast<float>(data[i]);
        }
    }

    template <typename T>
    void cudaInt_(T data, size_t size)
    {
        CUDA_KERNEL_LOOP(i, size){
            data[i] = static_cast<int>(data[i]);
        }
    }

    template <typename T>
    void cudaInt(T data, size_t size, int *out)
    {
        CUDA_KERNEL_LOOP(i, size){
            out[i] = static_cast<int>(data[i]);
        }
    }

    template <typename T>
    void cudaFloat_(T data, size_t size)
    {
        CUDA_KERNEL_LOOP(i, size){
            data[i] = static_cast<float>(data[i]);
        }
    }

    template <typename T>
    void cudaFloat(T data, size_t size, float *out)
    {
        CUDA_KERNEL_LOOP(i, size){
            out[i] = static_cast<float>(data[i]);
        }
    }
}