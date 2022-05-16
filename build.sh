#!/bin/bash

set -x
set -e

git submodule update --init --recursive .
pip3 install -r requirements.txt

# GRPC build for python
python -m grpc_tools.protoc -Icsrc/protos --python_out=. --grpc_python_out=. csrc/protos/runtime.proto

# FastNICS start
export Torch_DIR=$HOME/debug/libtorch
export NCCL_DIR=$HOME/nccl/build

export NCCL_ROOT_DIR=$NCCL_DIR
export NCCL_INCLUDE_DIR=$NCCL_DIR/include
export NCCL_LIB_DIR=$NCCL_DIR/lib
export NCCL_VERSION="2.11.4"

export CUDA_HOME=/usr/local/cuda

export CUDAToolkit_ROOT=$CUDA_HOME
export CUDACXX=$CUDA_HOME/bin/nvcc
# FastNICS end

pushd bench
# FastNICs start
#python setup.py install &
#setuppid=$!
python setup.py install
# FastNICs end
popd

# CPP runtime build.
mkdir -p csrc/build
pushd csrc/build/
# FastNICs start
#cmake ..
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=$Torch_DIR -DCMAKE_PREFIX_PATH=$NCCL_DIR ..
# FastNICs end
make -j
popd

# FastNICs start
#wait $setuppid
# FastNICs end
