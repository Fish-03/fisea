#pragma once

#include <memory>
#include "type.h"

namespace fisea
{
    class FloatTensorDeleter
    {
    public:
        // 構造函數，接受一個設備
        FloatTensorDeleter(fisea::Device dev) : device(dev) {}

        // 刪除器：根據不同的設備釋放內存
        void operator()(float *ptr) const
        {
            if (!ptr)
                return;

            if (device == fisea::Device::CPU)
            {
                delete[] ptr;
            }
            else if (device == fisea::Device::CUDA)
            {
                cudaFree(ptr);
            }
        }

    private:
        const fisea::Device device; // 儲存設備類型以決定釋放方式
    };

    using FloatPtr = std::shared_ptr<float[]>;

}
