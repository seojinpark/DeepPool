#!/bin/bash

set -x

# GRPC build for python
python -m grpc_tools.protoc -Icsrc/protos --python_out=. --grpc_python_out=. csrc/protos/runtime.proto

# CPP runtime build.
mkdir -p csrc/build
pushd csrc/build/
# cmake -DCMAKE_BUILD_TYPE=Debug -O0 -DCMAKE_PREFIX_PATH=/libtorch ..
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/libtorch ..
make -j
popd
