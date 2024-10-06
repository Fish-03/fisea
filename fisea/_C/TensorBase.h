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
    class TensorBase
    {
    protected:
        std::shared_ptr<void> data;
        std::vector<int> shape;
        std::vector<int> stride;
        int numel;

        Device device;
        Dtype dtype;

        
        

        std::shared_ptr<Tensor> grad;

    public:
        bool requires_grad = false;
        bool is_leaf = true;
        virtual ~TensorBase() {};
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