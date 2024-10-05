#pragma once
#include <memory>
#include "type.h"
#include "FunctionBase.h"
#include "FloatTensor.h"
namespace fisea
{
// class ReLU : public FunctionBase
//     {
//     private:
//         std::shared_ptr<FloatTensor> forward(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> output=nullptr) override;
//         void backward(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> grad=nullptr) override;

//     public:
//         ReLU() {};
//         ~ReLU() {};

//     };

class relu : public FnBase<relu>
{
    public:
        static FloatTensorPtr forward(ctx_t &ctx, FloatTensorPtr x);
        static void backward(ctx_t &ctx, FloatTensorPtr grad);
};

} // namespace fisea