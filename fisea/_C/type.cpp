#include "type.h"
#include <stdexcept>
#include <string>
#include <iostream>

fisea::Dtype fisea::dtype_from_string(const string &dtype_str)
{
    if (dtype_str == "int" || dtype_str == "INT")
    {
        return fisea::Dtype::INT;
    }
    else if (dtype_str == "float" || dtype_str == "FLOAT")
    {
        return fisea::Dtype::FLOAT;
    }
    else if (dtype_str == "double" || dtype_str == "DOUBLE")
    {
        return fisea::Dtype::DOUBLE;
    }
    else
    {
        throw std::invalid_argument("Invalid dtype string");
    }
}

fisea::Dtype fisea::dtype_from_string(fisea::Dtype dtype)
{
    return dtype;
}

fisea::Device fisea::device_from_string(const string &device_str)
{
    if (device_str == "cpu" || device_str == "CPU")
    {
        return fisea::Device::CPU;
    }
    else if (device_str == "cuda" || device_str == "CUDA")
    {
        return fisea::Device::CUDA;
    }
    else
    {
        throw std::invalid_argument("Invalid device string");
    }
}

fisea::Device fisea::device_from_string(fisea::Device device)
{
    return device;
}

string fisea::device_to_string(Device device)
{
    switch (device)
    {
    case Device::CPU:
        return "cpu";
    case Device::CUDA:
        return "cuda";
    default:
        throw std::invalid_argument("Invalid device");
    }
}

string fisea::device_to_string(string device)
{
    return device;
}

string fisea::dtype_to_string(Dtype dtype)
{
    switch (dtype)
    {
    case Dtype::INT:
        return "int";
    case Dtype::FLOAT:
        return "float";
    case Dtype::DOUBLE:
        return "double";
    default:
        throw std::invalid_argument("Invalid dtype: " + std::to_string(static_cast<int>(dtype)));
    }
}

string fisea::dtype_to_string(string dtype)
{
    return dtype;
}

size_t fisea::dtype_size(Dtype dtype)
{
    switch (dtype)
    {
    case Dtype::INT:
        return sizeof(int);
    case Dtype::FLOAT:
        return sizeof(float);
    case Dtype::DOUBLE:
        return sizeof(double);
    default:
        throw std::invalid_argument("Invalid dtype");
    }
}

size_t fisea::dtype_size(string dtype)
{
    return dtype_size(dtype_from_string(dtype));
}