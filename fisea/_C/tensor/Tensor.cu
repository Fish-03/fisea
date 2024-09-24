#include "Tensor.h"
#include "../type.h"
#include "../handler.cuh"
#include "../functional/kernel.cuh"

namespace fisea
{
    void Tensor::_write_cpu_cuda(void *data)
    {
        if (data_ == nullptr)
        {
            data_ = std::shared_ptr<void>(cuda::cuMalloc<char>(data_size_), cuda::cuDeleter<char>());
        }

        cudaMemcpy(data_.get(), data, data_size_, cudaMemcpyHostToDevice);
    }

    void Tensor::_write_cuda_cpu(void *data)
    {
        if (data_ == nullptr)
        {
            data_ = std::shared_ptr<void>(new char[data_size_], std::default_delete<char[]>());
        }

        cudaMemcpy(data, data_.get(), data_size_, cudaMemcpyDeviceToHost);
    }

    void Tensor::_write_cuda_cuda(void *data)
    {
        if (data_ == nullptr)
        {
            data_ = std::shared_ptr<void>(cuda::cuMalloc<char>(data_size_), cuda::cuDeleter<char>());
        }

        cudaMemcpy(data_.get(), data, data_size_, cudaMemcpyDeviceToDevice);
    }

    Tensor Tensor::_to_int_cuda()
    {
        switch (this->dtype_)
        {
        case fisea::Dtype::INT:
        {
            return this->copy();
        }
        case fisea::Dtype::FLOAT:
        {
            size_t size = this->data_size_ / sizeof(float);
            Tensor out = Tensor(this->shape_, this->device_, fisea::Dtype::INT, nullptr);

            fisea::cudaInt<float*>(static_cast<float *>(this->data_.get()), size, static_cast<int *>(out.data_.get()));

            return out;
        }
        default:
            throw std::runtime_error("Unknown dtype");
        }
    }

    Tensor Tensor::_to_float_cuda()
    {
        switch (this->dtype_)
        {
        case fisea::Dtype::INT:
        {
            size_t size = this->data_size_ / sizeof(int);
            Tensor out = Tensor(this->shape_, this->device_, fisea::Dtype::FLOAT, nullptr);
            fisea::cudaFloat<int*>(static_cast<int *>(this->data_.get()), size, static_cast<float *>(out.data_.get()));
            return out;
        }
        case fisea::Dtype::FLOAT:
        {
            return this->copy();
        }
        default:
            throw std::runtime_error("Unknown dtype");
        }
    }

    void Tensor::_to_int_cuda_()
    {
        switch (this->dtype_)
        {
        case fisea::Dtype::INT:
            return;
        case fisea::Dtype::FLOAT:
        {
            size_t size = this->data_size_ / sizeof(float);
            fisea::cudaInt_<float*>(static_cast<float *>(this->data_.get()), size);
            this->dtype_ = fisea::Dtype::INT;
            return;
        }
        default:
            throw std::runtime_error("Unknown dtype");
        }
    }

    void Tensor::_to_float_cuda_()
    {
        switch (this->dtype_)
        {
        case fisea::Dtype::INT:
        {
            size_t size = this->data_size_ / sizeof(int);
            fisea::cudaFloat_<int*>(static_cast<int *>(this->data_.get()), size);
            this->dtype_ = fisea::Dtype::FLOAT;
            return;
        }
        case fisea::Dtype::FLOAT:
            return;
        default:
            throw std::runtime_error("Unknown dtype");
        }
    }

}