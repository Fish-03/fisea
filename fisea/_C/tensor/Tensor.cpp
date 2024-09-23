#include <iostream>
#include <string>
#include <memory>
#include <cstring> // for memcpy
#include <random>

#include "../type.h"
#include "../handler.h"
#include "../const.h"
#include "../memory.cuh"

#include "Tensor.h"

namespace fisea
{
    Tensor::Tensor(fisea::Shape shape, void *data, fisea::Device device, fisea::Dtype dtype)
        : shape_(shape), device_(device), dtype_(dtype), data_(nullptr), grad_(nullptr), data_size_(shape.size() * fisea::dtype_size(dtype_))
    {
        if (data != nullptr)
        {
            switch (device_)
            {
            case fisea::Device::CPU:
                this->_write_cpu(data); // 用char (1字節) 來進行多態，然後通過 fisea::Dtype 來決定每個元素的解析方式
            case fisea::Device::CUDA:
#ifdef __CUDACC__
                this->_write_cpu_cuda(data);
                break;
#endif
                throw std::runtime_error("CUDA is not enabled.");
            default:
                throw std::runtime_error("Unknown device type.");
            }
        }
    }

    void Tensor::_write_cpu(void *data)
    {
        if (data_ == nullptr)
        {
            data_ = std::shared_ptr<void>(new char[data_size_], std::default_delete<char[]>());
        }
        memcpy(data_.get(), data, data_size_);
    }

    Tensor::Tensor(fisea::Shape shape, void *data, std::string device, fisea::Dtype dtype)
        : Tensor(shape, data, fisea::device_from_string(device), dtype)
    {
    }

    Tensor::Tensor(fisea::Shape shape, void *data, fisea::Device device, std::string dtype)
        : Tensor(shape, data, device, fisea::dtype_from_string(dtype))
    {
    }

    Tensor::Tensor(fisea::Shape shape, void *data, std::string device, std::string dtype)
        : Tensor(shape, data, fisea::device_from_string(device), fisea::dtype_from_string(dtype))
    {
    }

    // 析构函数，注意 data_ 不需要释放，因为 std::shared_ptr 会自动处理
    Tensor::~Tensor()
    {
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
#ifdef __CUDACC__
            Tensor out = Tensor(this->shape_, nullptr, fisea::Device::CPU, this->dtype_);
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
            CHECK_CUDA_ENABLED();
#ifdef __CUDACC__
            Tensor out = Tensor(this->shape_, nullptr, fisea::Device::CUDA, this->dtype_);
            out._write_cpu_cuda(this->data_.get());
            return out;
#endif
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
#ifdef __CUDACC__
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
#ifdef __CUDACC__
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
#ifdef __CUDACC__
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
#ifdef __CUDACC__
            return this->_to_float_cuda();
#endif
        default:
            throw std::runtime_error("Unknown device type.");
        }
    }

    Tensor Tensor::copy()
    {
        Tensor out = Tensor(this->shape_, nullptr, this->device_, this->dtype_);
        switch (this->device_)
        {
        case fisea::Device::CPU:
            out._write_cpu(this->data_.get());
            return out;

        case fisea::Device::CUDA:
#ifdef __CUDACC__
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
        Tensor out = Tensor(other.shape_, other.data_.get(), other.device_, other.dtype_);
        return out;
    }

    Tensor Tensor::_to_int_cpu()
    {
        switch (this->dtype_)
        {
        case fisea::Dtype::INT:
            return this->copy();
        case fisea::Dtype::FLOAT:
        {
            Tensor out = Tensor(this->shape_, nullptr, this->device_, fisea::Dtype::INT);
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
            Tensor out = Tensor(this->shape_, nullptr, this->device_, fisea::Dtype::FLOAT);
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
#ifdef __CUDACC__
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
#ifdef __CUDACC__
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
            int value = static_cast<int>(value);
            int *ptr = (int *)this->data_.get();
            for (size_t i = 0; i < this->shape_.size(); i++)
            {
                ptr[i] = value;
            }
            break;
        case fisea::Dtype::FLOAT:
            float *ptr = (float *)this->data_.get();
            for (size_t i = 0; i < this->shape_.size(); i++)
            {
                ptr[i] = value;
            }
            break;
        default:
            throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
        }
    }

    void Tensor::_fill_cpu(int value)
    {
        switch (this->dtype_)
        {
        case fisea::Dtype::INT:
            int *ptr = (int *)this->data_.get();
            for (size_t i = 0; i < this->shape_.size(); i++)
            {
                ptr[i] = value;
            }
            break;
        case fisea::Dtype::FLOAT:
            float value = static_cast<float>(value);
            float *ptr = (float *)this->data_.get();
            for (size_t i = 0; i < this->shape_.size(); i++)
            {
                ptr[i] = value;
            }
            break;
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
#ifdef __CUDACC__
            return this->_randn_cuda();
#endif
            throw std::runtime_error("CUDA is not enabled.");
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

        float *ptr = (float *)this->data_.get();

        for (size_t i = 0; i < this->shape_.size(); i++)
        {
            ptr[i] = distribution(generator);
        }
    }

    Tensor Tensor::zeros(fisea::Shape shape, fisea::Device device, fisea::Dtype dtype)
    {
        Tensor out = Tensor(shape, nullptr, device, dtype);
        if (out.device_ == fisea::Device::CPU)
        {
            memset(out.data_.get(), 0, out.data_size_);
        }
        else if (device == fisea::Device::CUDA)
        {
            CHECK_CUDA_ENABLED();
            cudaMemset(out.data_.get(), 0, out.data_size_);
        }
        else
        {
            throw std::runtime_error("Unknown device type.");
        }
        return out;
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
        Tensor out = Tensor(shape, nullptr, device, dtype);
        if (out.device_ == fisea::Device::CPU)
        {
            memset(out.data_.get(), 1, out.data_size_);
        }
        else if (device == fisea::Device::CUDA)
        {
            CHECK_CUDA_ENABLED();
            cudaMemset(out.data_.get(), 1, out.data_size_);
        }
        else
        {
            throw std::runtime_error("Unknown device type.");
        }
        return out;
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
        Tensor out = Tensor(shape, nullptr, device, dtype);
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
