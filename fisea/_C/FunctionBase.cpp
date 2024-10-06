#include "FunctionBase.h"

using namespace fisea;

FloatTensorPtr Add::forward(ctx_t &ctx, FloatTensorPtr x, FloatTensorPtr y)
{
    auto output = FloatTensor::create(x->get_shape(), {}, x->requires_grad, false);
    auto outdata = output->get_data().get();
    auto xdata = x->get_data().get();
    auto ydata = y->get_data().get();

    for (int i = 0; i < x->get_numel(); i++)
    {
        outdata[i] = xdata[i] + ydata[i];
    }

    ctx["x"] = x;
    ctx["y"] = y;

    return output;
}

std::tuple<FloatTensorPtr, FloatTensorPtr> Add::backward(ctx_t &ctx, FloatTensorPtr grad)
{
    auto x = std::get<FloatTensorPtr>(ctx["x"]);
    auto y = std::get<FloatTensorPtr>(ctx["y"]);

    // x->print();
    // x->fill_(1.0);

    if (grad == nullptr)
    {
        grad = fisea::FloatTensor::create(x->get_shape(), {}, false, false);
        grad->fill_(1);
    }

    return std::make_tuple(grad, grad);
}
