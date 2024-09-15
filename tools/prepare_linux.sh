#! /bin/bash

set -e
set -x

pip install "cmake==3.22.*"

# Install CUDA 12.2:

if [ "$CIBW_ARCHS" == "aarch64" ]; then
    sleep 0.01
else
    yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
    # error mirrorlist.centos.org doesn't exists anymore.
    sed -i s/mirror.centos.org/vault.centos.org/g /etc/yum.repos.d/*.repo
    sed -i s/^#.*baseurl=http/baseurl=http/g /etc/yum.repos.d/*.repo
    sed -i s/^mirrorlist=http/#mirrorlist=http/g /etc/yum.repos.d/*.repo
    yum install --setopt=obsoletes=0 -y \
        cuda-nvcc-12-2-12.2.140-1 \
        cuda-cudart-devel-12-2-12.2.140-1 \
        libcurand-devel-12-2-10.3.3.141-1 \
        libcudnn8-devel-8.9.7.29-1.cuda12.2 \
        libcublas-devel-12-2-12.2.5.6-1 \
        libnccl-devel-2.19.3-1+cuda12.2

    ln -s cuda-12.2 /usr/local/cuda

    export PATH="/usr/local/cuda/bin:$PATH"
    export CPATH="/usr/local/cuda/include:$CPATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

    nvcc -V
fi

# pip install -r requirements.txt

# mkdir build && cd build

# cmake ..

# VERBOSE=1 make -j$(nproc) install

# cd ..

# rm -r build