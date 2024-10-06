// FloatTensor
#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <iomanip>

#include "type.h"
#include "FloatTensor.h"
#include "random.h"

using namespace fisea;

FloatTensor::FloatTensor(std::vector<int> shape, std::vector<int> stride, bool requires_grad, bool is_leaf)
{
    this->device = Device::CPU;
    this->dtype = Dtype::FLOAT;
    this->requires_grad = requires_grad;
    this->is_leaf = is_leaf;

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

    std::cout << "[DEBUG] FloatTensor is created" << std::endl;
}

FloatTensorPtr FloatTensor::create(std::vector<int> shape, std::vector<int> stride, bool requires_grad, bool is_leaf)
{
    return std::make_shared<FloatTensor>(shape, stride, requires_grad, is_leaf);
}

FloatTensorPtr FloatTensor::cpu()
{
    return shared_from_this();
}

std::shared_ptr<CudaFloatTensor> FloatTensor::cuda()
{
    return CudaFloatTensor::create(shared_from_this());
}

void FloatTensor::print(const char *fmt, int depth, int start, int maxWidth, int maxHeight) const
{

    float *dataPtr = this->data.get();
    int dims = shape.size();

    if (depth == 0)
    {
        std::cout << "tensor(";
    }
    // 打印左括号
    std::cout << '[';

    if (depth == dims - 1)
    {
        char buff[100];
        snprintf(buff, sizeof(buff), fmt, 0.0);
        int width = this->shape[depth] * (std::string(buff).size() + 2) + depth * 2 + 8;

        if (shape[depth] > 0)
        {

            for (int i = 0; i < this->shape[depth]; i++)
            {
                if (width < maxWidth || i <= 3 || i >= this->shape[depth] - 4)
                {
                    snprintf(buff, sizeof(buff), fmt, dataPtr[start + i * stride[depth]]);
                    std::cout << std::string(buff);
                    if (i < this->shape[depth] - 1)
                    {
                        std::cout << ", ";
                    }
                }
                else
                {
                    std::cout << "..., ";
                }
            }
        }
        std::cout << ']';
    }
    else
    {
        int height = this->shape[depth];
        if (shape[depth] > 0)
        {
            for (int i = 0; i < this->shape[depth]; i++)
            {
                if (height < maxHeight || i <= 3 || i >= this->shape[depth] - 4)
                {
                    this->print(fmt, depth + 1, start + i * stride[depth], maxWidth, maxHeight);
                    if (i < this->shape[depth] - 1)
                    {
                        std::cout << "," << std::string(dims - depth - 1, '\n') << std::string(depth + 8, ' ');
                    }
                }
                else
                {
                    std::cout << "...," << std::string(dims - depth - 1, '\n') << std::string(depth + 8, ' ');
                    i = this->shape[depth] - 4;
                }
            }
        }
        std::cout << ']';
    }
    if (depth == 0)
    {
        std::cout << ")" << std::endl;
    }
}

void FloatTensor::uniform_(float low, float high)
{
    auto indices = this->get_indices();
    float *dataPtr = this->data.get();

    for (int i : indices)
    {
        dataPtr[i] = fisea::rand(low, high);
    }
}

void FloatTensor::normal_(float mean, float std)
{
    auto indices = this->get_indices();
    float *dataPtr = this->data.get();

    for (int i : indices)
    {
        dataPtr[i] = mean + std * fisea::randn();
    }
}

template <typename T>
void FloatTensor::fill_(T value)
{
    auto indices = this->get_indices();
    auto ptr = this->data.get();

    value = static_cast<float>(value);

    for (int i : indices)
    {
        ptr[i] = value;
    }
}

template void FloatTensor::fill_<int>(int value);
template void FloatTensor::fill_<float>(float value);
template void FloatTensor::fill_<double>(double value);

void FloatTensor::backward(std::shared_ptr<FloatTensor> grad, bool retain_graph, bool create_graph)
{
    if (grad == nullptr)
    {
        if (this->grad == nullptr)
        {
            grad = FloatTensor::create(this->shape, {}, requires_grad = false, is_leaf = false);
            grad->ones_();
        }
        else
        {
            grad = this->grad;
        }
    }

    if (this->grad_fn != nullptr)
    {
        this->grad_fn(grad, retain_graph, create_graph);
    }

    else
    {
        // std::cout << "grad_fn is nullptr" << std::endl;
        try
        {
            throw std::invalid_argument("grad_fn is nullptr");
        }
        catch (const std::invalid_argument &e)
        {
            std::cerr << e.what() << '\n';
        }
    }
}