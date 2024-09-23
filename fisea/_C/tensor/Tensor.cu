#include "Tensor.h"
#include "../type.h"
#include "../handler.cuh"

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
            return *this;
        }
        case fisea::Dtype::FLOAT:
        {
            //todo
        }
        }
    }
}