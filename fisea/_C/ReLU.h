#pragma once
#include "FunctionBase.h"
#include "FloatTensor.h"
namespace fisea
{
class relu : public FnBase<relu>
{
    public:
        static FloatTensorPtr forward(ctx_t &ctx, FloatTensorPtr x);
        static std::tuple<FloatTensorPtr> backward(ctx_t &ctx, FloatTensorPtr grad);
};

}