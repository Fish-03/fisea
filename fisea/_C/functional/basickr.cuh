#pragma once

namespace fisea
{
    template <typename T>
    __global__ void _to_int_kernel_(T *data, size_t size);
    template <typename T>
    __global__ void _to_float_kernel_(T *data, size_t size);
    template <typename T>
    __global__ void _to_int_kernel(T *data, size_t size, int *out);
    template <typename T>
    __global__ void _to_float_kernel(T *data, size_t size, float *out);
}