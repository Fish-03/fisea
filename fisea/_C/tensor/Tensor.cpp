#include "Tensor.h"
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

    Tensor::Tensor(void *data, fisea::Shape shape, fisea::Device device, fisea::Dtype dtype)
        : shape_(shape), device_(device), dtype_(dtype), data_(nullptr), grad_(nullptr)
    {
        size_t dataSize = shape.size() * fisea::dtype_size(dtype_); // shape.size() 返回元素数量，dtype.size() 返回每个元素的大小

        if (data != nullptr)
        {

            if (device_ == fisea::Device::CPU)
            {
                std::shared_ptr data_ = std::shared_ptr<void *>(new void *[dataSize]);
                memcpy(data_.get(), data, dataSize);
            }
            else if (device_ == fisea::Device::CUDA)
            {
                // TODO 此处需要处理从 CPU 到 CUDA 的数据迁移逻辑, 如果對於PINNED MEMORY的話呢?, 怎麼知道是否Pinned Memory呢?
                if (fisea::USE_CUDA)
                {
                    std::shared_ptr data_ = std::shared_ptr<void *>(cuda::cuMalloc<void *>(dataSize), cuda::cuDeleter<void *>());
                    cudaMemcpy(data_.get(), data, dataSize, cudaMemcpyHostToDevice);
                }
                else
                {
                    throw std::runtime_error("CUDA is not enabled.");
                }
            }
            // else Error
        }
    }

    Tensor::Tensor(void *data, fisea::Shape shape, std::string device, std::string dtype)
        : Tensor(data, shape, fisea::device_from_string(device), fisea::dtype_from_string(dtype))
    {
    }

    Tensor::Tensor(void *data, fisea::Shape shape, std::string device, fisea::Dtype dtype)
        : Tensor(data, shape, fisea::device_from_string(device), dtype)
    {
    }

    Tensor::Tensor(void *data, fisea::Shape shape, fisea::Device device, std::string dtype)
        : Tensor(data, shape, device, fisea::dtype_from_string(dtype))
    {
    }

    // 析构函数，注意 data_ 不需要释放，因为 std::shared_ptr 会自动处理
    Tensor::~Tensor()
    {
        // 自动管理资源，无需手动释放
    }

    // 静态方法 - 拷贝当前 Tensor 并返回
    Tensor Tensor::copy()
    {
        Tensor copyTensor;
        copyTensor.shape_ = this->shape_;
        copyTensor.device_ = this->device_;
        copyTensor.dtype_ = this->dtype_;

        size_t dataSize = this->shape_.size() * this->dtype_.size();
        if (data_ != nullptr)
        {
            copyTensor.data_ = std::shared_ptr<void *[]>(new void *[dataSize]);
            memcpy(copyTensor.data_.get(), this->data_.get(), dataSize);
        }

        return copyTensor;
    }

    // 静态方法 - 将当前 Tensor 转移到 CPU 设备上
    Tensor Tensor::cpu()
    {
        if (this->device_ == fisea::Device::CPU)
        {
            return *this;
        }
        else
        {
            Tensor cpuTensor = this->copy();
            cpuTensor.device_ = fisea::Device::CPU;
            // 此处需要处理从 CUDA 到 CPU 的数据迁移逻辑
            return cpuTensor;
        }
    }

    // 静态方法 - 将当前 Tensor 转移到 CUDA 设备上
    Tensor Tensor::cuda()
    {
        if (this->device_ == fisea::Device::CUDA)
        {
            return *this;
        }
        else
        {
            Tensor cudaTensor = this->copy();
            cudaTensor.device_ = fisea::Device::CUDA;
            // 此处需要处理从 CPU 到 CUDA 的数据迁移逻辑
            return cudaTensor;
        }
    }

    // 静态方法 - 将 Tensor 转为 int 类型
    Tensor Tensor::to_int()
    {
        Tensor intTensor = this->copy();
        intTensor.dtype_ = fisea::Dtype::INT;
        // 此处需要实现数据类型转换逻辑
        return intTensor;
    }

    // 静态方法 - 将 Tensor 转为 float 类型
    Tensor Tensor::to_float()
    {
        Tensor floatTensor = this->copy();
        floatTensor.dtype_ = fisea::Dtype::FLOAT;
        // 此处需要实现数据类型转换逻辑
        return floatTensor;
    }
}
