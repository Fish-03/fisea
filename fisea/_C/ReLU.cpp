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
    return output;
}
std::shared_ptr<FloatTensor> ReLU::backwardcalulate(std::shared_ptr<FloatTensor> grad)
{
    auto data = grad->get_data();
    auto shape = grad->get_shape();
    auto stride = grad->get_stride();
    auto numel = grad->get_numel();
    auto device = grad->get_device();
    auto dtype = grad->get_dtype();
    auto dataPtr = data.get();
    auto indices = grad->get_indices();
    for (int i : indices)
    {
        dataPtr[i] = dataPtr[i] > 0 ? 1 : 0;
    }
    return grad;
}