// 這裡定義了有關FloatTensor 和 CudaFloatTensor 的類別
#pragma once

#include <memory>
#include <vector>

#include "type.h"
#include "TensorBase.h"

namespace fisea {
    class FloatTensor: public Tensor {
    private:
        std::shared_ptr<float*> data = nullptr;

    protected:
        fisea::Device device = fisea::Device::CPU;
        fisea::Dtype dtype = fisea::Dtype::FLOAT;

    public:
        FloatTensor(std::shared_ptr<float*> data = nullptr, std::vector<size_t> shape = {}, std::vector<size_t> stride = {});

        static std::shared_ptr<FloatTensor> create(std::shared_ptr<float*> data = nullptr, std::vector<int> shape = {}, std::vector<int> stride = {});

        std::shared_ptr<Tensor> cpu();
        std::shared_ptr<Tensor> cuda();

        void print() const;


    };
}

// Tensor* a = FloatTensor.create(nullptr, {1, 2, 3});