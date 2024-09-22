#pragma once

#define CHECK_CUDA_ENABLED()                                \
    if (!fisea::USE_CUDA) {                                 \
        throw std::runtime_error("CUDA is not enabled.");   \
    }
