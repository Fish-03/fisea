// Tensor的基類, 定義了非常多虛函數方法.

#pragma once
#include <memory>
#include <iostream>
#include <vector>

#include "type.h"

namespace fisea
{
    class Tensor
    {
    protected:
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
        virtual ~Tensor()
        {
            std::cout << "[DEBUG] Tensor has been removed" << std::endl;
        }
        // static TensorBase *create(const std::vector<int> &shap);

        // void print() const {};

        const std::vector<int> &get_shape() { return shape; }
        const int &get_numel() { return numel; }
        const fisea::Device &get_device() { return device; }
        const fisea::Dtype &get_dtype() { return dtype; }
        const std::vector<int> &get_stride() { return stride; }
        void requires_grad_(bool requires_grad) { this->requires_grad = requires_grad; }

        static std::vector<int> get_indices(const std::vector<int> &shape, const std::vector<int> &stride);

        template <typename T>
        static void printRecursive(int depth, int &printed, int maxLength, const std::vector<int> &indices,
                                   const std::vector<int> &shape, std::shared_ptr<T> dataPtr)
        {
            {
                if (depth == shape.size() - 1)
                { // 最後一維，打印數據
                    for (int i = 0; i < shape[depth]; ++i)
                    {
                        if (printed >= maxLength)
                        {
                            std::cout << "..."; // 超過最大長度時顯示 "..."
                            return;
                        }
                        int flat_index = indices[printed];
                        std::cout << dataPtr.get()[flat_index] << " ";
                        printed++;
                    }
                    std::cout << std::endl;
                }
                else
                { // 遞歸打印多維結構
                    for (int i = 0; i < shape[depth]; ++i)
                    {
                        if (printed >= maxLength)
                        {
                            std::cout << "..."; // 超過最大長度時顯示 "..."
                            return;
                        }
                        std::cout << "[";
                        Tensor::printRecursive<T>(depth + 1, printed, maxLength, indices, shape, dataPtr);
                        std::cout << "]" << std::endl;
                    }
                }
            }
        }
    };
}