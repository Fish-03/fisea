// FloatTensor
#include <memory>
#include <vector>
#include <iomanip>

#include "type.h"
#include "FloatTensor.h"

using namespace fisea;

FloatTensor::FloatTensor(std::vector<int> shape, std::vector<int> stride)
{
    this->device = Device::CPU;
    this->dtype = Dtype::FLOAT;

    if (shape.empty())
    {
        throw std::invalid_argument("shape must not be empty");
    }
    else
    {
        this->shape = shape;
    }

    if (stride.empty())
    {
        this->stride = std::vector<int>();
        int cur_stride = int(1);
        this->stride.insert(this->stride.begin(), 1);
        for (int i = this->shape.size() - 1; i > 0; i--)
        {
            cur_stride *= this->shape[i];
            this->stride.insert(this->stride.begin(), cur_stride);
        }
    }
    else
    {
        this->stride = stride;
    }

    int numel = int(1);
    for (int i = 0; i < shape.size(); i++)
    {
        numel *= shape[i];
    }
    this->numel = numel;

    this->data = std::shared_ptr<float>(new float[this->numel], std::default_delete<float[]>());
}

std::shared_ptr<FloatTensor> FloatTensor::create(std::vector<int> shape, std::vector<int> stride)
{
    return std::make_shared<FloatTensor>(shape, stride);
}

std::shared_ptr<FloatTensor> FloatTensor::cpu()
{
    return shared_from_this();
}

std::shared_ptr<CudaFloatTensor> FloatTensor::cuda()
{
    return CudaFloatTensor::create(this);
}

void FloatTensor::print(int maxLength, int precision) const
{
    std::vector<int> indices = this->get_indices(this->shape, this->stride);
    int num_elements = indices.size();

    std::cout << std::fixed << std::setprecision(precision); // 設置固定精度

    int printed = int(0);

    this->printRecursive<float>(0, printed, maxLength, indices, this->shape, this->data);
}