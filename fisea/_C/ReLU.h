#pragma once
#include "FunctionBase.h"
#include "FloatTensor.h"
namespace fisea
{
class ReLU : public FunctionBase
    {
    private:
        std::shared_ptr<FloatTensor> forward(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> output=nullptr) override;
        std::shared_ptr<FloatTensor> backwardcalulate(std::shared_ptr<FloatTensor> grad) override;

    public:
        ReLU() {};
        ~ReLU() {};
    };
    
} // namespace fisea