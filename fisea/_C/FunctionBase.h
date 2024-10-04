#include <iostream>
#pragma once

namespace fisea {
    template <typename Dtype>
    class Function
    {
    protected:
        virtual void forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output = nullptr) = 0;
        virtual void backward_cal(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> gradient, std::shared_ptr<Tensor> output = nullptr) = 0;

    public:
        virtual std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> gradient) = 0;
        virtual void backward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> gradient, std::shared_ptr<Tensor> output) = 0;
        virtual std::shared_ptr<Tensor> operator()(std::shared_ptr<Tensor> input) {
            forward(input);
            return input;
        }
        virtual void operator()(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output){
            forward(input, output);
        }
    };
    
    template <typename Dtype>
    class ReLU : public Function<Dtype>
    {
    protected:
        void forward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> output = nullptr);
        void backward_cal(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> gradient, std::shared_ptr<Tensor> output = nullptr);

    public:
        ReLU();
        ~ReLU();
        static Function<Dtype>* create();
        std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> gradient);
        void backward(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> gradient, std::shared_ptr<Tensor> output);
    };
    
    template <typename Dtype>
    ReLU<Dtype>::ReLU(){}
    template <typename Dtype>
    ReLU<Dtype>::~ReLU(){}
    template <typename Dtype>
    Function<Dtype>* ReLU<Dtype>::create(){
        return new ReLU<Dtype>();
    }



} // namespace fisea