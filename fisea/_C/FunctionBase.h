
#pragma once
#include <iostream>
#include "FloatTensor.h"
namespace fisea
{
    class FunctionBase
    {
    private:
        virtual std::shared_ptr<FloatTensor> forward(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> output=nullptr) = 0;
        virtual void backward(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> grad=nullptr) = 0;

    public:
        virtual ~FunctionBase();
        // virtual std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> grad) = 0;
        std::shared_ptr<FloatTensor> operator()(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> output=nullptr);

    };
    

} // namespace fisea