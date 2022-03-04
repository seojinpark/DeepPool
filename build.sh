#!/bin/bash

set -x
set -e

# GRPC build for python
python -m grpc_tools.protoc -Icsrc/protos --python_out=. --grpc_python_out=. csrc/protos/runtime.proto

# assumes pytorch installed in anaconda environment
export Torch_DIR=$CONDA_PREFIX/lib/python3.9/site-packages/torch/share/cmake/Torch/
export matplotlib_cpp_INCLUDE_DIRS=/matplotlib-cpp/build/
export BUILD_ANVIL=1

export CUDAToolkit_ROOT=$CUDA_HOME
export CUDACXX=$CUDA_HOME/bin/nvcc

pushd bench
python setup.py install &
setuppid=$!
popd

# CPP runtime build.
mkdir -p csrc/build
pushd csrc/build/

if (($# == 2)); then
    cmake -DCMAKE_BUILD_TYPE=$1 -DCMAKE_PREFIX_PATH=$2 -DBUILD_ANVIL=$BUILD_ANVIL ..
elif (($# == 1)); then
    cmake -DCMAKE_BUILD_TYPE=$1 -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DBUILD_ANVIL=$BUILD_ANVIL ..
else
    # cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_LIBRARY=/root/miniconda3/envs/pt-test/lib/python3.9 ..
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DBUILD_ANVIL=$BUILD_ANVIL ..
fi

# cmake -DCMAKE_BUILD_TYPE=Debug -O0 -DCMAKE_PREFIX_PATH=/libtorch ..
# cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/libtorch ..

# cmake Debug -DCMAKE_PREFIX_PATH=/libtorch ..

make -j
popd

wait $setuppid