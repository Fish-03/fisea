#! /bin/bash

set -e
set -x

CUDA_ROOT="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2"
curl -L -nv -o cuda.exe https://developer.download.nvidia.com/compute/cuda/12.2.2/local_installers/cuda_12.2.2_537.13_windows.exe
./cuda.exe -s nvcc_12.2 cudart_12.2 cublas_dev_12.2 curand_dev_12.2
rm cuda.exe

# CUDNN_ROOT="C:/Program Files/NVIDIA/CUDNN/v8.8"
# curl -L -nv -o cudnn.exe https://developer.download.nvidia.com/compute/redist/cudnn/v8.8.0/local_installers/12.0/cudnn_8.8.0.121_windows.exe
# ./cudnn.exe -s
# sleep 10
# cp -r "$CUDNN_ROOT"/* "$CUDA_ROOT"
# rm cudnn.exe

pip install -r requirements.txt

mkdir build
cd build
cmake ..
cmake --build . --config Release --target install --parallel 6 --verbose
cd ..
rm -r build

# cp "$CUDA_ROOT/bin/cudnn64_8.dll" python/ctranslate2/