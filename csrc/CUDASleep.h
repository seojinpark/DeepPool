#pragma once

#include <cuda_runtime.h>

void gpu_nsleep(unsigned nanos, cudaStream_t stream);
