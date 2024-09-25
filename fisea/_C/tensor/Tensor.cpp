#include <iostream>
#include <string>
#include <memory>
#include <cstring> // for memcpy
#include <random>

#include "../type.h"
#include "../const.h"
#include "../functional/kernel.cuh"

#include "Tensor.h"

namespace fisea
{
    Tensor::Tensor(fisea::Shape shape, fisea::Device device, fisea::Dtype dtype, void *data)
        : shape_(shape), device_(device), dtype_(dtype), data_(nullptr), grad_(nullptr), data_size_(shape.size() * fisea::dtype_size(dtype_))
    {
        if (data != nullptr)
        {
            switch (device_)
            {
            case fisea::Device::CPU:
                this->_write_cpu(data); // 用char (1字節) 來進行多態，然後通過 fisea::Dtype 來決定每個元素的解析方式
            case fisea::Device::CUDA:
#ifdef USE_CUDA
                this->_write_cpu_cuda(data);
                break;
#endif
                throw std::runtime_error("CUDA is not enabled.");
            default:
                throw std::runtime_error("Unknown device type.");
            }
        }
#ifdef _DEBUG
        std::cout << "[DEBUG] Tensor has been created" << std::endl;
#endif
    }

    void Tensor::_write_cpu(void *data)
    {
        if (data_ == nullptr)
        {
            data_ = std::shared_ptr<void>(new char[data_size_], std::default_delete<char[]>());
        }
        memcpy(data_.get(), data, data_size_);
    }

    Tensor::Tensor(fisea::Shape shape, std::string device, fisea::Dtype dtype, void *data)
        : Tensor(shape, fisea::device_from_string(device), dtype, data)
    {
    }

    Tensor::Tensor(fisea::Shape shape, fisea::Device device, std::string dtype, void *data)
        : Tensor(shape, device, fisea::dtype_from_string(dtype), data)
    {
    }

    Tensor::Tensor(fisea::Shape shape, std::string device, std::string dtype, void *data)
        : Tensor(shape, fisea::device_from_string(device), fisea::dtype_from_string(dtype), data)
    {
    }

    // 析构函数，注意 data_ 不需要释放，因为 std::shared_ptr 会自动处理
    Tensor::~Tensor()
    {
#ifdef _DEBUG
        std::cout << "[DEBUG] Tensor has been deleted" << std::endl;
#endif
        // 自动管理资源，无需手动释放?
    }

    Tensor Tensor::cpu()
    {
        if (this->device_ == fisea::Device::CPU)
        {
            return *this;
        }
        else if (this->device_ == fisea::Device::CUDA)
        {
#ifdef USE_CUDA
            Tensor out = Tensor(this->shape_, fisea::Device::CPU, this->dtype_, nullptr);
            out._write_cuda_cpu(this->data_.get());
            return out;
#endif
            throw std::runtime_error("CUDA is not enabled.");
        }
        else
        {
            throw std::runtime_error("Unknown device type.");
        }
    }

    Tensor Tensor::cuda()
    {
        switch (this->device_)
        {
        case fisea::Device::CUDA:
            return *this;
        case fisea::Device::CPU:
        {
#ifdef USE_CUDA
            Tensor out = Tensor(this->shape_, fisea::Device::CUDA, this->dtype_, nullptr);
            out._write_cpu_cuda(this->data_.get());
            return out;
#endif
            throw std::runtime_error("CUDA is not enabled.");
        }
        default:
            throw std::runtime_error("Unknown device type.");
        }
    }

    void Tensor::to_int_()
    {
        switch (this->device_)
        {
        case fisea::Device::CPU:
            this->_to_int_cpu_();
            return;
        case fisea::Device::CUDA:
#ifdef USE_CUDA
            this->_to_int_cuda_();
            return;
#endif
            throw std::runtime_error("CUDA is not enabled.");
        default:
            throw std::runtime_error("Unknown device type.");
        }
    }

    void Tensor::to_float_()
    {
        switch (this->device_)
        {
        case fisea::Device::CPU:
            this->_to_float_cpu_();
            return;
        case fisea::Device::CUDA:
#ifdef USE_CUDA
            this->_to_float_cuda_();
            return;
#endif
            throw std::runtime_error("CUDA is not enabled.");
        default:
            throw std::runtime_error("Unknown device type.");
        }
    }

    // TODO: 這個不好實現，如果是 gpu -> gpu 的話，應寫一個 kernel 來實現
    Tensor Tensor::to_int()
    {
        switch (this->device_)
        {
        case fisea::Device::CPU:
            return this->_to_int_cpu();
        case fisea::Device::CUDA:
#ifdef USE_CUDA
            return this->_to_int_cuda();
#endif
            throw std::runtime_error("CUDA is not enabled.");
        default:
            throw std::runtime_error("Unknown device type.");
        }
    }
    // TODO: 這個不好實現，如果是 gpu -> gpu 的話，應寫一個 kernel 來實現
    Tensor Tensor::to_float()
    {
        switch (this->device_)
        {
        case fisea::Device::CPU:
            return this->_to_float_cpu();
        case fisea::Device::CUDA:
#ifdef USE_CUDA
            return this->_to_float_cuda();
#endif
        default:
            throw std::runtime_error("Unknown device type.");
        }
    }

    Tensor Tensor::copy()
    {
        Tensor out = Tensor(this->shape_, this->device_, this->dtype_, nullptr);
        switch (this->device_)
        {
        case fisea::Device::CPU:
            out._write_cpu(this->data_.get());
            return out;

        case fisea::Device::CUDA:
#ifdef USE_CUDA
            out._write_cuda_cuda(this->data_.get());
            return out;
#endif
            throw std::runtime_error("CUDA is not enabled.");
        default:
            throw std::runtime_error("Unknown device type.");
        }
    }

    Tensor Tensor::from(Tensor other)
    {
        Tensor out = Tensor(other.shape_, other.device_, other.dtype_, other.data_.get());
        return out;
    }
    // TODO: how to access the data pointer of a py::array?
    //  Tensor Tensor::from(const py::array &array)
    //  {
    //  py::dtype dtype = array.dtype();

    // // 用來存儲數據的指針
    // void *data = const_cast<void *>(array.data());
    // fisea::Dtype tensor_dtype;

    // // 檢查是否是 float 或 int 類型，否則轉換為 float
    // if (dtype.is(py::dtype::of<float>()))
    // {
    //     tensor_dtype = fisea::Dtype::FLOAT; // 直接支持 float32
    // }
    // else if (dtype.is(py::dtype::of<double>()))
    // {
    //     // 將 double 轉換為 float
    //     py::array_t<float> float_array = array.cast<py::array_t<float>>();
    //     float* float_data = float_array.mutable_data();
    //     data = static_cast<void *>(float_data);
    //     tensor_dtype = fisea::Dtype::FLOAT;
    // }
    // else if (dtype.is(py::dtype::of<int32_t>()))
    // {
    //     tensor_dtype = fisea::Dtype::INT; // 直接支持 int32
    // }
    // else if (dtype.is(py::dtype::of<int64_t>()) || dtype.is(py::dtype::of<long>()))
    // {
    //     // 將 int64 轉換為 int32
    //     py::array_t<int32_t> int_array = array.cast<py::array_t<int32_t>>();
    //     int* int_data = int_array.mutable_data();
    //     data = static_cast<void *>(int_data);
    //     tensor_dtype = fisea::Dtype::INT;
    // }
    // else
    // {
    //     throw std::runtime_error("Unsupported dtype");
    // }

    // // 將 NumPy 的形狀轉換為 Tensor 的形狀
    // std::vector<size_t> shape;
    // for (py::ssize_t i = 0; i < array.ndim(); ++i)
    // {
    //     shape.push_back(array.shape(i));
    // }

    // std::vector<int> int_shape(shape.begin(), shape.end());
    // fisea::Shape tensor_shape(int_shape);

    // return Tensor(tensor_shape, fisea::Device::CPU, fisea::dtype_from_string(array.dtype().str()), data);
    // }

    Tensor Tensor::_to_int_cpu()
    {
        switch (this->dtype_)
        {
        case fisea::Dtype::INT:
            return this->copy();
        case fisea::Dtype::FLOAT:
        {
            Tensor out = Tensor(this->shape_, this->device_, fisea::Dtype::INT, nullptr);
            int *ptr = (int *)out.data_.get();
            float *this_ptr = (float *)this->data_.get();
            for (size_t i = 0; i < this->shape_.size(); i++)
            {
                ptr[i] = static_cast<int>(this_ptr[i]);
            }
            return out;
        }
        default:
            throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
        }
    }

    Tensor Tensor::_to_float_cpu()
    {
        switch (this->dtype_)
        {
        case fisea::Dtype::INT:
        {
            Tensor out = Tensor(this->shape_, this->device_, fisea::Dtype::FLOAT, nullptr);
            float *ptr = (float *)out.data_.get();
            int *this_ptr = (int *)this->data_.get();
            for (size_t i = 0; i < this->shape_.size(); i++)
            {
                ptr[i] = static_cast<float>(this_ptr[i]);
            }
            return out;
        }
        case fisea::Dtype::FLOAT:
            return this->copy();
        default:
            throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
        }
    }

    void Tensor::_to_int_cpu_()
    {
        switch (this->dtype_)
        {
        case fisea::Dtype::INT:
            return;
        case fisea::Dtype::FLOAT: // 這種方法不能推廣到double, 會有內存對齊問題.
        {
            int *ptr = (int *)this->data_.get();
            float *this_ptr = (float *)this->data_.get();
            for (size_t i = 0; i < this->shape_.size(); i++)
            {
                ptr[i] = static_cast<int>(this_ptr[i]);
            }
            this->dtype_ = fisea::Dtype::INT;
            return;
        }
        default:
            throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
        }
    }

    void Tensor::_to_float_cpu_()
    {
        switch (this->dtype_)
        {
        case fisea::Dtype::INT:
        {
            float *ptr = (float *)this->data_.get();
            int *this_ptr = (int *)this->data_.get();
            for (size_t i = 0; i < this->shape_.size(); i++)
            {
                ptr[i] = static_cast<float>(this_ptr[i]);
            }
            this->dtype_ = fisea::Dtype::FLOAT;
            return;
        }
        case fisea::Dtype::FLOAT:
            return;
        default:
            throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
        }
    }

    void Tensor::fill_(int value)
    {
        // TODO 如果 value 和 Tensor 的 dtype 不匹配，應該先進行類型轉換
        switch (this->device_)
        {
        case fisea::Device::CPU:
        {
            return this->_fill_cpu(value);
        }
        case fisea::Device::CUDA:
        {
#ifdef USE_CUDA
            return this->_fill_cuda(value);
#endif
            throw std::runtime_error("CUDA is not enabled.");
        }
        default:
            throw std::runtime_error("Unknown device type.");
        }
    }

    void Tensor::fill_(float value)
    {
        switch (this->device_)
        {
        case fisea::Device::CPU:
        {
            return this->_fill_cpu(value);
        }
        case fisea::Device::CUDA:
        {
#ifdef USE_CUDA
            return this->_fill_cuda(value);
#endif
            throw std::runtime_error("CUDA is not enabled.");
        }
        default:
            throw std::runtime_error("Unknown device type.");
        }
        // TODO 如果 value 和 Tensor 的 dtype 不匹配，應該先進行類型轉換
    }

    void Tensor::_fill_cpu(float value)
    {
        switch (this->dtype_)
        {
        case fisea::Dtype::INT:
        {
            int v = static_cast<int>(value);
            int *ptr = (int *)this->data_.get();
            for (size_t i = 0; i < this->shape_.size(); i++)
            {
                ptr[i] = v;
            }
            break;
        }

        case fisea::Dtype::FLOAT:
        {
            float *ptr = (float *)this->data_.get();
            for (size_t i = 0; i < this->shape_.size(); i++)
            {
                ptr[i] = value;
            }
            break;
        }

        default:
            throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
        }
    }

    void Tensor::_fill_cpu(int value)
    {
        switch (this->dtype_)
        {
        case fisea::Dtype::INT:
        {
            int *ptr = (int *)this->data_.get();
            for (size_t i = 0; i < this->shape_.size(); i++)
            {
                ptr[i] = value;
            }
            break;
        }
        case fisea::Dtype::FLOAT:
        {
            float v = static_cast<float>(value);
            float *ptr = (float *)this->data_.get();
            for (size_t i = 0; i < this->shape_.size(); i++)
            {
                ptr[i] = v;
            }
            break;
        }
        default:
            throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
        }
    }

    void Tensor::zero_()
    {
        this->fill_(0);
    }

    void Tensor::one_()
    {
        this->fill_(1);
    }

    // TODO 種子設定
    void Tensor::randn_()
    {
        switch (this->device_)
        {
        case fisea::Device::CPU:
        {
            return this->_randn_cpu();
        }
        case fisea::Device::CUDA:
        {
// #ifdef USE_CUDA
//             return this->_randn_cuda();
// #endif
            throw std::runtime_error("NOT IMPLEMENTED");
        }
        default:
            throw std::runtime_error("Unknown device type.");
        }
    }

    void Tensor::_randn_cpu()
    {
        if (this->dtype_ != fisea::Dtype::FLOAT)
        {
            throw std::runtime_error("Function `randn` Not Support for type: " + fisea::dtype_to_string(this->dtype_));
        }

        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<float> distribution(0, 1);

        float *ptr = static_cast<float *>(this->data_.get());

        for (size_t i = 0; i < this->shape_.size(); i++)
        {
            ptr[i] = distribution(generator);
        }
    }

    Tensor Tensor::zeros(fisea::Shape shape, fisea::Device device, fisea::Dtype dtype)
    {
        Tensor out = Tensor(shape, device, dtype, nullptr);
        if (out.device_ == fisea::Device::CPU)
        {
            memset(out.data_.get(), 0, out.data_size_);
            return out;
        }
        else if (device == fisea::Device::CUDA)
        {
#ifdef USE_CUDA
            cudaMemset(out.data_.get(), 0, out.data_size_);
            return out;
#endif
            throw std::runtime_error("CUDA is not enabled.");
        }
        else
        {
            throw std::runtime_error("Unknown device type.");
        }
    }

    Tensor Tensor::zeros(fisea::Shape shape, fisea::Device device, std::string dtype)
    {
        return Tensor::zeros(shape, device, fisea::dtype_from_string(dtype));
    }

    Tensor Tensor::zeros(fisea::Shape shape, std::string device, fisea::Dtype dtype)
    {
        return Tensor::zeros(shape, fisea::device_from_string(device), dtype);
    }

    Tensor Tensor::zeros(fisea::Shape shape, std::string device, std::string dtype)
    {
        return Tensor::zeros(shape, fisea::device_from_string(device), fisea::dtype_from_string(dtype));
    }

    Tensor Tensor::ones(fisea::Shape shape, fisea::Device device, fisea::Dtype dtype)
    {
        Tensor out = Tensor(shape, device, dtype, nullptr);
        if (out.device_ == fisea::Device::CPU)
        {
            memset(out.data_.get(), 1, out.data_size_);
            return out;
        }
        else if (device == fisea::Device::CUDA)
        {
#ifdef USE_CUDA
            cudaMemset(out.data_.get(), 1, out.data_size_);
            return out;
#endif
            throw std::runtime_error("CUDA is not enabled.");
        }
        else
        {
            throw std::runtime_error("Unknown device type.");
        }
    }

    Tensor Tensor::ones(fisea::Shape shape, fisea::Device device, std::string dtype)
    {
        return Tensor::ones(shape, device, fisea::dtype_from_string(dtype));
    }

    Tensor Tensor::ones(fisea::Shape shape, std::string device, fisea::Dtype dtype)
    {
        return Tensor::ones(shape, fisea::device_from_string(device), dtype);
    }

    Tensor Tensor::ones(fisea::Shape shape, std::string device, std::string dtype)
    {
        return Tensor::ones(shape, fisea::device_from_string(device), fisea::dtype_from_string(dtype));
    }

    Tensor Tensor::randn(fisea::Shape shape, fisea::Device device, fisea::Dtype dtype)
    {
        Tensor out = Tensor(shape, device, dtype, nullptr);
        out.randn_();
        return out;
    }

    Tensor Tensor::randn(fisea::Shape shape, fisea::Device device, std::string dtype)
    {
        return Tensor::randn(shape, device, fisea::dtype_from_string(dtype));
    }

    Tensor Tensor::randn(fisea::Shape shape, std::string device, fisea::Dtype dtype)
    {
        return Tensor::randn(shape, fisea::device_from_string(device), dtype);
    }

    Tensor Tensor::randn(fisea::Shape shape, std::string device, std::string dtype)
    {
        return Tensor::randn(shape, fisea::device_from_string(device), fisea::dtype_from_string(dtype));
    }
}
