// Tensor的基類, 定義了非常多虛函數方法.

#pragma once
#include <memory>
#include <string>
#include <functional>
#include <iostream>
#include <vector>
#include <iomanip>
#include <map>

#include "type.h"

namespace fisea
{
    template <typename T>
    struct _gradInfo
    {
        std::shared_ptr<T> prev;
        // grad_fn: 接收 *in.grad(ctx, *out.grad)
        std::function<std::shared_ptr<T>(std::map<std::string, std::shared_ptr<T>>, std::shared_ptr<T>)> grad_fn;
        // context for backward, 用於保存反向傳播時的中間結果, 注意保存時需要注意tensor是否需要被copy一份保存.
        std::map<std::string, std::shared_ptr<T>> ctx;
    };

    class Tensor
    {
    protected:
        std::shared_ptr<void> data;
        std::vector<int> shape;
        std::vector<int> stride;
        int numel;

        Device device;
        Dtype dtype;

        bool requires_grad = false;
        const bool is_leaf = false;

        std::shared_ptr<Tensor> grad;
        _gradInfo<Tensor> gradinfo;

    public:
        virtual ~Tensor() {};
        // static TensorBase *create(const std::vector<int> &shap);

        // void print() const {};

        const std::vector<int> &get_shape() { return shape; }
        int ndim() { return shape.size(); }
        const int &get_numel() { return numel; }
        const fisea::Device &get_device() { return device; }
        const fisea::Dtype &get_dtype() { return dtype; }
        const std::vector<int> &get_stride() { return stride; }
        void requires_grad_(bool requires_grad) { this->requires_grad = requires_grad; }

        std::vector<int> get_indices();
    };

}