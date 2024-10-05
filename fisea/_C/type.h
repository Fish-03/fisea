#pragma once

#include <string>
#include <memory>
#include <vector>
#include <tuple>
#include <variant>
#include <map>
#include <stdexcept>

using std::string;

namespace fisea
{
    class FloatTensor;
    class CudaFloatTensor;
    class IntTensor;
    class CudaIntTensor;

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
    
    using FloatTensorPtr = std::shared_ptr<FloatTensor>;
    using CudaFloatTensorPtr = std::shared_ptr<CudaFloatTensor>;
    using IntTensorPtr = std::shared_ptr<IntTensor>;
    using CudaIntTensorPtr = std::shared_ptr<CudaIntTensor>;

    using Tensor = std::variant<FloatTensor, CudaFloatTensor, IntTensor, CudaIntTensor>;
    using TensorPtr = std::variant<FloatTensorPtr, CudaFloatTensorPtr, IntTensorPtr, CudaIntTensorPtr>;
    
    using ctx_t = std::map<std::string, TensorPtr>;
    
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