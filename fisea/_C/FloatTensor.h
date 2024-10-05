// 這裡定義了有關FloatTensor 和 CudaFloatTensor 的類別
#pragma once

#include <memory>
#include <vector>
#include <functional>
#include "type.h"
#include "TensorBase.h"

namespace fisea
{
    class CudaFloatTensor;

    class FloatTensor : public TensorBase, public std::enable_shared_from_this<FloatTensor> {
    protected:
        std::shared_ptr<float> data;
        std::shared_ptr<FloatTensor> grad;

    public:
        bool requires_grad = true;
        FloatTensor(std::vector<int> shape = {}, std::vector<int> stride = {});
        ~FloatTensor()
        {
            std::cout << "[DEBUG] FloatTensor is deleted" << std::endl;
        };
        static std::shared_ptr<FloatTensor> create(std::vector<int> shape = {}, std::vector<int> stride = {});

        std::function<void(std::shared_ptr<FloatTensor>)> grad_fn = nullptr;
        
        void backward(std::shared_ptr<FloatTensor> grad = nullptr);

        std::shared_ptr<FloatTensor> cpu();
        std::shared_ptr<CudaFloatTensor> cuda();

        

        const std::shared_ptr<float> &get_data() const { return data; }
        void set_data(std::shared_ptr<float> data) { this->data = data; }
        const std::shared_ptr<FloatTensor> &get_grad() const { return grad; }
        void set_grad(std::shared_ptr<FloatTensor> grad) { this->grad = grad; }

        void print(const char *fmt = "%6.3f", int depth = 0, int start = 0, int maxWidth = 100, int maxHeight = 10) const;

        template <typename T>
        void fill_(T value)
        {
            auto indices = this->get_indices();
            auto ptr = this->data.get();

            value = static_cast<float>(value);

            for (int i : indices)
            {
                ptr[i] = value;
            }
        }

        void ones_() { fill_(1.0); }
        void zeros_() { fill_(0.0); }
        void uniform_(float low = 0.0, float high = 1.0);
        void normal_(float mean = 0.0, float std = 1.0);
    };

    class CudaFloatTensor : public TensorBase, public std::enable_shared_from_this<CudaFloatTensor>
    {
    protected:
        std::shared_ptr<float> data;
        std::shared_ptr<CudaFloatTensor> grad;

    public:
        bool requires_grad = true;
        CudaFloatTensor(std::vector<int> shape = {}, std::vector<int> stride = {});
        ~CudaFloatTensor()
        {
            std::cout << "[DEBUG] CudaFloatTensor is deleted" << std::endl;
        };
        static std::shared_ptr<CudaFloatTensor> create(std::vector<int> shape = {}, std::vector<int> stride = {});
        static std::shared_ptr<CudaFloatTensor> create(FloatTensor *tensor);

        std::shared_ptr<FloatTensor> cpu() const;
        std::shared_ptr<CudaFloatTensor> cuda();

        const std::shared_ptr<float> &get_data() const { return data; }
        void set_data(std::shared_ptr<float> data) { this->data = data; }
        const std::shared_ptr<CudaFloatTensor> &get_grad() const { return grad; }
        void set_grad(std::shared_ptr<CudaFloatTensor> grad) { this->grad = grad; }

        void print(const char *fmt = "%6.3f", int depth = 0, int start = 0, int maxWidth = 100, int maxHeight = 10) const;

        template <typename T>
        void fill_(T value);

        void ones_() { fill_(1.0); }
        void zeros_() { fill_(0.0); }
        void uniform_();
        void normal_(float mean = 0.0, float std = 1.0);
    };
}

// Tensor* a = FloatTensor.create(nullptr, {1, 2, 3});