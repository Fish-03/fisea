#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using std::string;

namespace fisea
{
    enum class Device
    {
        CPU,
        CUDA,
    };

    // dtype type
    enum class Dtype
    {
        INT,
        FLOAT,
        DOUBLE,
    };
    
    using Shape = std::vector<int>;

    Device device_from_string(const string &device_str);
    Device device_from_string(Device device);
    
    Dtype dtype_from_string(const string &dtype_str);
    Dtype dtype_from_string(Dtype dtype);

    string device_to_string(Device device);
    string device_to_string(string device);

    string dtype_to_string(Dtype dtype);
    string dtype_to_string(string dtype);

    size_t dtype_size(Dtype dtype);
    size_t dtype_size(string dtype);
}