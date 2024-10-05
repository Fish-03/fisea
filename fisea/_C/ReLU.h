#pragma once
#include "FunctionBase.h"
#include "FloatTensor.h"
namespace fisea
{
class relu : public FnBase<relu>
{
    public:
        static FloatTensorPtr forward(ctx_t &ctx, FloatTensorPtr x);
        static void backward(ctx_t &ctx, FloatTensorPtr grad);
};

} // namespace fisea