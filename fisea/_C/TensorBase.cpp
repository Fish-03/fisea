#include "TensorBase.h"

using namespace fisea;

std::vector<int> Tensor::get_indices(const std::vector<int> &shape, const std::vector<int> &stride)
{
    std::vector<int> indices(shape.size(), 0);
    for (int i = 0; i < shape.size(); i++)
    {
        if (shape.size() != stride.size())
        {
            throw std::invalid_argument("Shape and stride must have the same length.");
        }

        // 計算張量的總元素數量
        int num_elements = 1;
        for (int dim : shape)
        {
            num_elements *= dim;
        }

        // 結果存儲索引
        std::vector<int> indices(num_elements);

        // 遞歸遍歷各個維度以計算每個元素的位置
        std::vector<int> current_index(shape.size(), 0); // 用來追踪當前的多維索引
        for (int i = 0; i < num_elements; ++i)
        {
            int flat_index = 0; // 扁平化數據中的索引

            // 根據當前的多維索引計算扁平化索引
            for (int d = 0; d < shape.size(); ++d)
            {
                flat_index += current_index[d] * stride[d];
            }

            indices[i] = flat_index;

            // 更新多維索引 (模擬多維數組中的進位)
            for (int d = shape.size(); d > 0; --d)
            {
                if (++current_index[d - 1] < shape[d - 1])
                {
                    break;
                }
                current_index[d - 1] = 0;
            }
        }

        return indices;
    }
    return indices;
}