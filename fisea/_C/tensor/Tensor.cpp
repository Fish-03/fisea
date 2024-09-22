#include "Tensor.h"
#include "../handler.h"
#include "../const.h"
#include "../memory.cuh"

#include <memory>
#include <cstring> // for memcpy

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#endif

namespace fisea
{
    template <typename DeviceType = fisea::Device, typename DtypeType = fisea::Dtype>
    Tensor::Tensor(void *data, fisea::Shape shape, DeviceType device, DtypeType dtype)
        : shape_(shape), device_(fisea::device_from_string(device)), dtype_(fisea::dtype_from_string(dtype)), data_(nullptr), grad_(nullptr), data_size_(shape.size() * fisea::dtype_size(dtype_))
    {
        if (data != nullptr)
        {

            if (device_ == fisea::Device::CPU)
            {
                std::shared_ptr data_ = std::shared_ptr<void *>(new void *[data_size_]);
                memcpy(data_.get(), data, data_size_);
            }
            else if (device_ == fisea::Device::CUDA)
            {
                CHECK_CUDA_ENABLED();
                std::shared_ptr data_ = std::shared_ptr<void *>(cuda::cuMalloc<void *>(data_size_), cuda::cuDeleter<void *>());
                cudaMemcpy(data_.get(), data, data_size_, cudaMemcpyHostToDevice);
            }
            // else Error
        }
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
            Tensor out = Tensor(nullptr, this->shape_, fisea::Device::CPU, this->dtype_);
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
            Tensor out = Tensor(nullptr, this->shape_, fisea::Device::CUDA, this->dtype_);
            cudaMemcpy(out.data_.get(), this->data_.get(), this->data_size_, cudaMemcpyHostToDevice);
            return out;
        }
        else
        {
            throw std::runtime_error("Unknown device type.");
        }
    }

    Tensor Tensor::to_int()
    {
        if (this->dtype_ == fisea::Dtype::INT)
        {
            return *this;
        }
        else if (this->dtype_ == fisea::Dtype::FLOAT)
        {
            Tensor out = Tensor(nullptr, this->shape_, this->device_, fisea::Dtype::INT);
            if (this->device_ == fisea::Device::CPU)
            // TODO: 這種做法效率應該很低?
            {
                for (size_t i = 0; i < this->shape_.size(); i++)
                {
                    ((int *)out.data_.get())[i] = (int)((float *)this->data_.get())[i];
                }
            }
            else if (this->device_ == fisea::Device::CUDA)
            {
                cudaMemcpy(out.data_.get(), this->data_.get(), this->data_size_, cudaMemcpyDeviceToDevice);
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

    Tensor Tensor::to_float()
    {
        if (this->dtype_ == fisea::Dtype::FLOAT)
        {
            return *this;
        }
        else if (this->dtype_ == fisea::Dtype::INT)
        {
            Tensor out = Tensor(nullptr, this->shape_, this->device_, fisea::Dtype::FLOAT);
            if (this->device_ == fisea::Device::CPU)
            // TODO: 這種做法效率應該很低?
            {
                for (size_t i = 0; i < this->shape_.size(); i++)
                {
                    ((float *)out.data_.get())[i] = (float)((int *)this->data_.get())[i];
                }
            }
            else if (this->device_ == fisea::Device::CUDA)
            {
                cudaMemcpy(out.data_.get(), this->data_.get(), this->data_size_, cudaMemcpyDeviceToDevice);
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
            Tensor out = Tensor(nullptr, this->shape_, this->device_, this->dtype_);
            memcpy(out.data_.get(), this->data_.get(), this->data_size_);
            return out;
        }
        else if (this->device_ == fisea::Device::CUDA)
        {
            CHECK_CUDA_ENABLED();
            Tensor out = Tensor(nullptr, this->shape_, this->device_, this->dtype_);
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
        Tensor out = Tensor(other.data_.get(), other.shape_, other.device_, other.dtype_);
        return out;
    }

    template <typename DtypeType>
    void Tensor::fill_(DtypeType value)
    {
        //TODO 如果 value 和 Tensor 的 dtype 不匹配，應該先進行類型轉換
        if (this->device_ == fisea::Device::CPU)
        {
            if (this->dtype_ == fisea::Dtype::INT){
                memset(this->data_.get(), (int)value, this->data_size_);
            }
            else if (this->dtype_ == fisea::Dtype::FLOAT){
                memset(this->data_.get(), (float)value, this->data_size_);
            }
            else{
                throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
            }
        }
        else if (this->device_ == fisea::Device::CUDA)
        {
            CHECK_CUDA_ENABLED();
            if (this->dtype_ == fisea::Dtype::INT){
                cudaMemset(this->data_.get(), (int)value, this->data_size_);
            }
            else if (this->dtype_ == fisea::Dtype::FLOAT){
                cudaMemset(this->data_.get(), (float)value, this->data_size_);
            }
            else{
                throw std::runtime_error("Not implemented type: " + fisea::dtype_to_string(this->dtype_));
            }
        }
        else
        {
            throw std::runtime_error("Unknown device type.");
        }
    }

    template <typename DeviceType = fisea::Device, typename DtypeType = fisea::Dtype>
    Tensor Tensor::zeros(fisea::Shape shape, DeviceType device, DtypeType dtype)
    {
        Tensor out = Tensor(nullptr, shape, device, dtype);
        if (out->device_ == fisea::Device::CPU)
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

    template <typename DeviceType = fisea::Device, typename DtypeType = fisea::Dtype>
    Tensor Tensor::ones(fisea::Shape shape, DeviceType device, DtypeType dtype)
    {
        Tensor out = Tensor(nullptr, shape, device, dtype);
        if (out->device_ == fisea::Device::CPU)
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

    template <typename DeviceType = fisea::Device, typename DtypeType = fisea::Dtype>
    Tensor Tensor::randn(fisea::Shape shape, DeviceType device, DtypeType dtype)
    {
        Tensor out = Tensor(nullptr, shape, device, dtype);
        //TODO 這個實現是有問題的
        if (out->device_ == fisea::Device::CPU)
        {
            for (size_t i = 0; i < out->shape_.size(); i++)
            {
                ((float *)out.data_.get())[i] = (float)rand() / RAND_MAX;
            }
        }
        else if (device == fisea::Device::CUDA)
        {
            CHECK_CUDA_ENABLED();
            for (size_t i = 0; i < out->shape_.size(); i++)
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
}
