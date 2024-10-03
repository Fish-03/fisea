// Tensor的基類, 定義了非常多虛函數方法.

#pragma once
#include <memory>
#include <iostream>
#include <vector>

#include "type.h"

namespace fisea {
    class TensorBase {
    private:
        std::vector<int> shape;
        std::vector<int> stride;
        
        fisea::Device device;
        fisea::Dtype dtype;

        std::shared_ptr<void> data = nullptr;

        bool requires_grad = false;
        TensorBase *grad = nullptr;
        // grad_fn
        bool is_leaf = false;
        

    public:
        ~TensorBase() {
            std::cout << "[DEBUG] Tensor has been removed" << std::endl;
        }
        // static TensorBase *create(const std::vector<int> &shap);
        virtual void initialize() = 0;
        virtual void finalize() = 0;
        virtual void reshape(int new_shape) = 0;
        virtual void print() const = 0;
    };
}