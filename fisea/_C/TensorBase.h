// Tensor的基類, 定義了非常多虛函數方法.

#pragma once
#include <memory>
#include <iostream>
#include <vector>

#include "type.h"

namespace fisea {
    class Tensor {
    private:

    protected:
        std::vector<size_t> shape;
        std::vector<size_t> stride;
        size_t numel;

        fisea::Device device;
        fisea::Dtype dtype;

        bool requires_grad = false;
        Tensor *grad = nullptr;
        // grad_fn
        bool is_leaf = false;

    public:
        ~Tensor() {
            std::cout << "[DEBUG] Tensor has been removed" << std::endl;
        }
        // static TensorBase *create(const std::vector<int> &shap);
        virtual std::shared_ptr<Tensor> cpu();
        virtual std::shared_ptr<Tensor> cuda();
    
        virtual void print() const;

        // virtual const std::shared_ptr<void> &get_data() { return data; }
        const std::vector<size_t> &get_shape() { return shape; }
        const size_t &get_numel() { return numel; }
        const fisea::Device &get_device() { return device; }
        const fisea::Dtype &get_dtype() { return dtype; }
        void requires_grad_(bool requires_grad) { this->requires_grad = requires_grad; }

    };
}