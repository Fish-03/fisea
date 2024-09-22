#pragma once
#include "const.h"

#define CHECK_CUDA_ENABLED() \
    if (!USE_CUDA) { throw std::runtime_error("CUDA is not enabled."); }
