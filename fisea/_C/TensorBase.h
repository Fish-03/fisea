// Tensor的基類, 定義了非常多虛函數方法.

#pragma once
#include <memory>
#include <iostream>
#include <vector>
#include <iomanip>
#include "type.h"

namespace fisea
{
    class Tensor
    {
    protected:
        std::shared_ptr<void> data;
        std::vector<int> shape;
        std::vector<int> stride;
        int numel;

        fisea::Device device;
        fisea::Dtype dtype;

        bool requires_grad = false;
        // Tensor *grad;
        // grad_fn
        bool is_leaf = false;

    public:
        virtual ~Tensor() {};
        // static TensorBase *create(const std::vector<int> &shap);

        // void print() const {};

        const std::vector<int> &get_shape() { return shape; }
        const int &get_numel() { return numel; }
        const fisea::Device &get_device() { return device; }
        const fisea::Dtype &get_dtype() { return dtype; }
        const std::vector<int> &get_stride() { return stride; }
        void requires_grad_(bool requires_grad) { this->requires_grad = requires_grad; }

        static std::vector<int> get_indices(const std::vector<int> &shape, const std::vector<int> &stride);
    };
}