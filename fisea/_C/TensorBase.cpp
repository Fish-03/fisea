#include <vector>
#include "TensorBase.h"
#include "Helper.h"

using namespace fisea;

namespace fisea
{
    std::vector<int> TensorBase::get_indices()
    {
        std::vector<int> indices(this->get_numel());
        __grapIdx(indices, 0, 0, this);
        return indices;
    }
}