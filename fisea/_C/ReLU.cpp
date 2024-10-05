#include "ReLU.h"

using namespace fisea;
std::shared_ptr<FloatTensor> ReLU::forward(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> output)
{
    auto data = input->get_data();
    auto shape = input->get_shape();
    auto stride = input->get_stride();
    auto numel = input->get_numel();
    auto device = input->get_device();
    auto dtype = input->get_dtype();
    auto dataPtr = data.get();
    auto indices = input->get_indices();
    if (output == nullptr)
    {
        output = FloatTensor::create(shape, stride);
    }
    auto outputData = output->get_data().get();
    for (int i : indices)
    {
        outputData[i] = dataPtr[i] > 0 ? dataPtr[i] : 0;
    }
    output->grad_fn = std::bind(&ReLU::backward, this, std::placeholders::_1, std::placeholders::_2);
    output->grad_fn_records.push_back(input);
    return output;
}
 
void ReLU::backward(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> grad)
{
    auto data = input->get_data();
    auto dataPtr = data.get();
   
    if (input->grad_fn_records.front()->get_grad() == nullptr)
    {
        input->grad_fn_records.front()->set_grad(FloatTensor::create(input->get_shape(), input->get_stride()));
    }
    auto dataGradPtr = input->grad_fn_records.front()->get_grad()->get_data().get();
    auto indices = input->get_indices();
    if (grad == nullptr)
    {
        for (int i : indices)
        {
            dataGradPtr[i] = dataPtr[i] > 0 ? 1 : 0;
        }
    }
    else
    {
        auto gradData = grad->get_data();
        auto gradDataPtr = gradData.get();
        for (int i : indices)
        {
            dataGradPtr[i] = dataPtr[i] > 0 ? gradDataPtr[i] : 0;
        }
    }



}

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