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

            if (device_ == fisea::Device::CPU)
            {
                // 用char (1字節) 來進行多態，然後通過 fisea::Dtype 來決定每個元素的解析方式
                data_ = std::shared_ptr<void>(new char[data_size_], std::default_delete<char[]>());
                memcpy(data_.get(), data, data_size_);
            }
            else if (device_ == fisea::Device::CUDA)
            {
                CHECK_CUDA_ENABLED();
                data_ = std::shared_ptr<void>(cuda::cuMalloc<char>(data_size_), cuda::cuDeleter<char>());
                cudaMemcpy(data_.get(), data, data_size_, cudaMemcpyHostToDevice);
            }
            // else Error
        }
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
            CHECK_CUDA_ENABLED();
            Tensor out = Tensor(this->shape_, nullptr, fisea::Device::CPU, this->dtype_);
            cudaMemcpy(out.data_.get(), this->data_.get(), this->data_size_, cudaMemcpyDeviceToHost);
            return out;
        }
        else
        {
            throw std::runtime_error("Unknown device type.");
        }
    }

    Tensor Tensor::cuda()
    {
        if (this->device_ == fisea::Device::CUDA)
        {
            return *this;
        }
        else if (this->device_ == fisea::Device::CPU)
        {
            CHECK_CUDA_ENABLED();
            Tensor out = Tensor(this->shape_, nullptr, fisea::Device::CUDA, this->dtype_);
            cudaMemcpy(out.data_.get(), this->data_.get(), this->data_size_, cudaMemcpyHostToDevice);
            return out;
        }
        else
        {
            throw std::runtime_error("Unknown device type.");
        }
    }

    //TODO: 這個不好實現，如果是 gpu -> gpu 的話，應寫一個 kernel 來實現
    Tensor Tensor::to_int()
    {
        if (this->dtype_ == fisea::Dtype::INT)
        {
            return *this;
        }
        else if (this->dtype_ == fisea::Dtype::FLOAT)
        {
            Tensor out = Tensor(this->shape_, nullptr, this->device_, fisea::Dtype::INT);
            if (this->device_ == fisea::Device::CPU)
            // TODO: 這種做法效率應該很低?
            {
                for (size_t i = 0; i < this->shape_.size(); i++)
                {
                    ((int *)out.data_.get())[i] = (int)((float *)this->data_.get())[i];
                }
                return out;
            }
            else if (this->device_ == fisea::Device::CUDA)
            {
                cudaMemcpy(out.data_.get(), this->data_.get(), this->data_size_, cudaMemcpyDeviceToDevice);
                return out;
            }
            else
            {
                throw std::runtime_error("Unknown device type.");
            }
        }
        else
        {
            throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
        }
    }

    //TODO: 這個不好實現，如果是 gpu -> gpu 的話，應寫一個 kernel 來實現
    Tensor Tensor::to_float()
    {
        if (this->dtype_ == fisea::Dtype::FLOAT)
        {
            return *this;
        }
        else if (this->dtype_ == fisea::Dtype::INT)
        {
            Tensor out = Tensor(this->shape_, nullptr, this->device_, fisea::Dtype::FLOAT);
            if (this->device_ == fisea::Device::CPU)
            // TODO: 這種做法效率應該很低?
            {
                for (size_t i = 0; i < this->shape_.size(); i++)
                {
                    ((float *)out.data_.get())[i] = (float)((int *)this->data_.get())[i];
                }
                return out;
            }
            else if (this->device_ == fisea::Device::CUDA)
            {
                cudaMemcpy(out.data_.get(), this->data_.get(), this->data_size_, cudaMemcpyDeviceToDevice);
                return out;
            }
            else
            {
                throw std::runtime_error("Unknown device type.");
            }
        }
        else
        {
            throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
        }
    }

    Tensor Tensor::copy()
    {
        if (this->device_ == fisea::Device::CPU)
        {
            Tensor out = Tensor(this->shape_, nullptr, this->device_, this->dtype_);
            memcpy(out.data_.get(), this->data_.get(), this->data_size_);
            return out;
        }
        else if (this->device_ == fisea::Device::CUDA)
        {
            CHECK_CUDA_ENABLED();
            Tensor out = Tensor(this->shape_, nullptr, this->device_, this->dtype_);
            cudaMemcpy(out.data_.get(), this->data_.get(), this->data_size_, cudaMemcpyDeviceToDevice);
            return out;
        }
        else
        {
            throw std::runtime_error("Unknown device type.");
        }
    }

    Tensor Tensor::from(Tensor other)
    {
        Tensor out = Tensor(other.shape_, other.data_.get(), other.device_, other.dtype_);
        return out;
    }

    void Tensor::fill_(float value)
    {
        // TODO 如果 value 和 Tensor 的 dtype 不匹配，應該先進行類型轉換
        if (this->device_ == fisea::Device::CPU)
        {
            switch (this->dtype_)
            {
            case fisea::Dtype::INT:
                memset(this->data_.get(), static_cast<int>(value), this->data_size_);
            case fisea::Dtype::FLOAT:
                memset(this->data_.get(), value, this->data_size_);
            default:
                throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
            }
        }
        else if (this->device_ == fisea::Device::CUDA)
        {
            switch (this->dtype_)
            {
            case fisea::Dtype::INT:
                cudaMemset(this->data_.get(), static_cast<int>(value), this->data_size_);
            case fisea::Dtype::FLOAT:
                cudaMemset(this->data_.get(), value, this->data_size_);
            default:
                throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
            }
        }
    }

    void Tensor::fill_(int value)
    {
        // TODO 如果 value 和 Tensor 的 dtype 不匹配，應該先進行類型轉換
        if (this->device_ == fisea::Device::CPU)
        {
            switch (this->dtype_)
            {
            case fisea::Dtype::INT:
                memset(this->data_.get(), value, this->data_size_);
            case fisea::Dtype::FLOAT:
                memset(this->data_.get(), static_cast<float>(value), this->data_size_);
            default:
                throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
            }
        }
        else if (this->device_ == fisea::Device::CUDA)
        {
            switch (this->dtype_)
            {
            case fisea::Dtype::INT:
                cudaMemset(this->data_.get(), value, this->data_size_);
            case fisea::Dtype::FLOAT:
                cudaMemset(this->data_.get(), static_cast<float>(value), this->data_size_);
            default:
                throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
            }
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
        std::random_device rd;
        std::mt19937 generator(rd());
        std::normal_distribution<float> distribution(0, 1);
        for (size_t i = 0; i < this->shape_.size(); i++)
        {
            ((float *)this->data_.get())[i] = distribution(generator);
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
        // TODO 這個實現是有問題的
        if (out.device_ == fisea::Device::CPU)
        {
            for (size_t i = 0; i < out.shape_.size(); i++)
            {
                ((float *)out.data_.get())[i] = (float)rand() / RAND_MAX;
            }
        }
        else if (device == fisea::Device::CUDA)
        {
            CHECK_CUDA_ENABLED();
            for (size_t i = 0; i < out.shape_.size(); i++)
            {
                ((float *)out.data_.get())[i] = (float)rand() / RAND_MAX;
            }
        }
        else
        {
            throw std::runtime_error("Unknown device type.");
        }
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
