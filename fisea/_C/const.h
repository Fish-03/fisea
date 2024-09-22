#pragma once

namespace fisea {
    #ifdef USE_CUDA
    const bool USE_CUDA = true;
    #else
    const bool USE_CUDA = false;
    #endif
}
