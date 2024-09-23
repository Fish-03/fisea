
#include <cuda_runtime.h>
#include <iostream>
#include "basickr.cuh"

namespace fisea
{
    template <typename T>
    __global__ void _to_int_kernel_(T *data, size_t size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
            data[idx] = std::static_cast<int>(data[idx]);
    }

    template <typename T>
    __global__ void _to_int_kernel(T *data, size_t size, int *out)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            out[idx] = std::static_cast<int>(data[idx]);
        }
    }

    template <typename T>
    __global__ void _to_float_kernel_(T *data, size_t size)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size)
        {
            data[idx] = std::static_cast<float>(data[idx]);
        }
    }

    template <typename T>
    __global__ void _to_float_kernel(T *data, size_t size, float *out)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) 
        {
            out[idx] = std::static_cast<float>(data[idx]);
        }
    }
}