// 這裡寫有關於 CudaFloatTensor的類
#include <memory>
#include <vector>
#include <iomanip>
#include <cuda.h>


#include "type.h"
#include "FloatTensor.h"

using namespace fisea;

CudaFloatTensor::CudaFloatTensor(std::vector<int> shape, std::vector<int> stride) {
    this->device = Device::CUDA;
    this->dtype = Dtype::FLOAT;

    if (shape.empty()) {
        throw std::invalid_argument("shape must not be empty");
    }
    else {
        this->shape = shape;
    }

    if (stride.empty()) {
        this->stride = std::vector<int>();
        int cur_stride = 1;
        for (int i = shape.size() - 1; i > 0; i--) {
            cur_stride *= shape[i];
            this->stride.insert(this->stride.begin(), cur_stride);
        }
    }
    else {
        this->stride = stride;
    }

    int numel = 1;
    for (int i = 0; i < shape.size(); i++) {
        numel *= shape[i];
    }
    this->numel = numel;

    float *dataPtr;
    cudaMalloc(&dataPtr, numel * sizeof(float));
    this->data = std::shared_ptr<float>(dataPtr, [](float *ptr) { cudaFree(ptr); });
}

std::shared_ptr<CudaFloatTensor> CudaFloatTensor::create(std::vector<int> shape, std::vector<int> stride) {
    return std::make_shared<CudaFloatTensor>(shape, stride);
}

std::shared_ptr<CudaFloatTensor> CudaFloatTensor::create(FloatTensor* t) {
    auto ct = CudaFloatTensor::create(t->get_shape(), t->get_stride());
    cudaMemcpy(ct->get_data().get(), t->get_data().get(), t->get_numel() * sizeof(float), cudaMemcpyHostToDevice);
    return ct;
}

std::shared_ptr<FloatTensor> CudaFloatTensor::cpu() const {
    auto t = fisea::FloatTensor::create(this->shape, this->stride);
    cudaMemcpy(t->get_data().get(), this->get_data().get(), this->numel * sizeof(float), cudaMemcpyDeviceToHost);
    return t;
}

std::shared_ptr<CudaFloatTensor> CudaFloatTensor::cuda() {
    return shared_from_this();
}

void CudaFloatTensor::print(const char* fmt, int depth, int start, int maxWidth, int maxHeight) const
{
    this->cpu()->print(fmt, depth, start, maxWidth, maxHeight);
}