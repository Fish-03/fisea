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
    input->grad_fn = std::bind(&ReLU::backward, this, std::placeholders::_1, std::placeholders::_2);
    return output;
}
 
void ReLU::backward(std::shared_ptr<FloatTensor> input, std::shared_ptr<FloatTensor> grad)
{
    auto data = input->get_data();
    auto dataPtr = data.get();
   
    if (input->grad == nullptr)
    {
        input->grad = FloatTensor::create(input->get_shape(), input->get_stride());
    }
    auto dataGradPtr = input->grad->get_data().get();
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