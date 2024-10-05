#include "ReLU.h"

using namespace fisea;

FloatTensorPtr relu::forward(ctx_t &ctx, FloatTensorPtr x){
    auto y = FloatTensor::create(x->get_shape());
    auto x_data = x->get_data().get();
    auto y_data = y->get_data().get();
    for (int i = 0; i < x->get_numel(); i++){
        y_data[i] = x_data[i] > 0 ? x_data[i] : 0;
    }
    y->grad_fn = std::bind(&relu::backward, ctx, std::placeholders::_1);

    ctx["x"] = x;
    
    return y;
}

void relu::backward(ctx_t &ctx, FloatTensorPtr grad){
    auto x = std::get<std::shared_ptr<FloatTensor>>(ctx["x"]);

    auto prev_grad = FloatTensor::create(grad->get_shape());

    auto prev_grad_data = prev_grad->get_data().get();
    auto grad_data = grad->get_data().get();
    auto x_data = x->get_data().get();
    for (int i = 0; i < x->get_numel(); i++){
        prev_grad_data[i] = x_data[i] > 0 ? grad_data[i] : 0;
    }

    x->set_grad(prev_grad);
}