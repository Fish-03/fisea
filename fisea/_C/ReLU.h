#pragma once
#include "FunctionBase.h"
#include "FloatTensor.h"
namespace fisea
{
class ReLU : public FunctionBase
    {
    private:
        std::shared_ptr<FloatTensor> forward(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> output=nullptr) override;

    public:
        ReLU() {};
        ~ReLU() {};
        void backward(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> grad=nullptr) override;

    };
    
} // namespace fisea