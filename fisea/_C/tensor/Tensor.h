#pragma once

#include <string>
#include <memory>

#include "../type.h"
#include "../memory.cuh"

namespace fisea
{
    class Tensor
    {
    public:
        // 普通的建構需要將 data_ 複製一份, 這樣可以避免 data_ 的生命週期問題
        Tensor(void *data = nullptr, fisea::Shape shape, fisea::Device device = fisea::Device::CPU, fisea::Dtype dtype = fisea::Dtype::FLOAT);
        Tensor(void *data = nullptr, fisea::Shape shape, std::string device = "cpu", std::string dtype = "float");
        Tensor(void *data = nullptr, fisea::Shape shape, std::string device = "cpu", fisea::Dtype dtype = fisea::Dtype::FLOAT);
        Tensor(void *data = nullptr, fisea::Shape shape, fisea::Device device = fisea::Device::CPU, std::string dtype = "float");
        ~Tensor(); // 不用釋放 data_ 

        void from(void *data); //TODO 利用這個函數可以不複制地建構 data_ 但是要注意 data_ 的生命週期, 比如from numpy array, 這個可以不用實現先
        static Tensor copy();
        static Tensor cpu();
        static Tensor cuda();

        static Tensor to_int();     //TODO 這個函數可以建立一個 int 的 Tensor
        static Tensor to_float();   //TODO 這個函數可以建立一個 float 的 Tensor
        void to_int_();             //TODO 這個函數可以將 Tensor 轉換為 int
        void to_float_();           //TODO 這個函數可以將 Tensor 轉換為 float

        static Tensor to(fisea::Device device, fisea::Dtype dtype); //TODO 這個函數可以建立一個指定 device 和 dtype 的 Tensor

        //TODO 在日後的 autograd 中, 可能需要新增其他函數修改 data_, grad_ 的指向.

    private:
        // Internal input check

    protected:
        fisea::Shape shape_;
        fisea::Device device_;
        fisea::Dtype dtype_ = fisea::Dtype::FLOAT;
        std::shared_ptr<void*[]> data_;
        Tensor *grad_ = nullptr;
    };

}
