#include "Tensor.h"
#include <cstring> // for memcpy

namespace fisea {

    Tensor::Tensor(void *data, fisea::Shape shape, fisea::Device device, fisea::Dtype dtype)
        : shape_(shape), device_(device), dtype_(dtype), data_(nullptr), grad_(nullptr)
    {
        size_t dataSize = shape.size() * fisea::dtype_size(dtype); // shape.size() 返回元素数量，dtype.size() 返回每个元素的大小
        data_ = std::shared_ptr<void*[]>(new void*[dataSize]);
        if (data != nullptr) {
            memcpy(data_.get(), data, dataSize);
        }
    }

    Tensor::Tensor(void *data, fisea::Shape shape, std::string device, std::string dtype)
        : shape_(shape), device_(fisea::device_from_string(device)), dtype_(fisea::dtype_from_string(dtype)), data_(nullptr), grad_(nullptr)
    {
        size_t dataSize = shape.size() * fisea::dtype_size(dtype_); // shape.size() 返回元素数量，dtype.size() 返回每个元素的大小
        data_ = std::shared_ptr<void*[]>(new void*[dataSize]);
        if (data != nullptr) {
        
            if (device_ == fisea::Device::CPU) {
                memcpy(data_.get(), data, dataSize);
            }
            else if (device_ == fisea::Device::CUDA) {
                //TODO 此处需要处理从 CPU 到 CUDA 的数据迁移逻辑, 如果對於PINNED MEMORY的話呢?, 怎麼知道是否Pinned Memory呢?
                
            }
            // else Error
        }
    }

    // 析构函数，注意 data_ 不需要释放，因为 std::shared_ptr 会自动处理
    Tensor::~Tensor() {
        // 自动管理资源，无需手动释放
    }

    // 静态方法 - 拷贝当前 Tensor 并返回
    Tensor Tensor::copy() {
        Tensor copyTensor;
        copyTensor.shape_ = this->shape_;
        copyTensor.device_ = this->device_;
        copyTensor.dtype_ = this->dtype_;

        size_t dataSize = this->shape_.size() * this->dtype_.size();
        if (data_ != nullptr) {
            copyTensor.data_ = std::shared_ptr<void*[]>(new void*[dataSize]);
            memcpy(copyTensor.data_.get(), this->data_.get(), dataSize);
        }

        return copyTensor;
    }

    // 静态方法 - 将当前 Tensor 转移到 CPU 设备上
    Tensor Tensor::cpu() {
        if (this->device_ == fisea::Device::CPU) {
            return *this;
        } else {
            Tensor cpuTensor = this->copy();
            cpuTensor.device_ = fisea::Device::CPU;
            // 此处需要处理从 CUDA 到 CPU 的数据迁移逻辑
            return cpuTensor;
        }
    }

    // 静态方法 - 将当前 Tensor 转移到 CUDA 设备上
    Tensor Tensor::cuda() {
        if (this->device_ == fisea::Device::CUDA) {
            return *this;
        } else {
            Tensor cudaTensor = this->copy();
            cudaTensor.device_ = fisea::Device::CUDA;
            // 此处需要处理从 CPU 到 CUDA 的数据迁移逻辑
            return cudaTensor;
        }
    }

    // 静态方法 - 将 Tensor 转为 int 类型
    Tensor Tensor::to_int() {
        Tensor intTensor = this->copy();
        intTensor.dtype_ = fisea::Dtype::INT;
        // 此处需要实现数据类型转换逻辑
        return intTensor;
    }

    // 静态方法 - 将 Tensor 转为 float 类型
    Tensor Tensor::to_float() {
        Tensor floatTensor = this->copy();
        floatTensor.dtype_ = fisea::Dtype::FLOAT;
        // 此处需要实现数据类型转换逻辑
        return floatTensor;
    }
}
