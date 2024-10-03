// FloatTensor
#include <memory>
#include <vector>

#include "type.h"
#include "FloatTensor.h"

using namespace fisea;

FloatTensor::FloatTensor(std::shared_ptr<float*> data, std::vector<size_t> shape, std::vector<size_t> stride) {
    this->data = data;
    if (shape.empty()) {
        throw std::invalid_argument("shape must not be empty");
    }
    else {
        this->shape = shape;
    }

    if (stride.empty()) {
        this->stride = std::vector<size_t>(1);
        size_t cur_stride = 1;
        this->stride.insert(this->stride.begin(), 1);
        for (size_t i = shape.size() -2 ; i >= 0; i--) {
            cur_stride *= shape[i];
            this->stride.insert(this->stride.begin(), cur_stride);
        }
    }
    else {
        this->stride = stride;
    }    
}
