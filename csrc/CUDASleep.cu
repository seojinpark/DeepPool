
#include "CUDASleep.h"

__global__ void nanosleepX(unsigned nanos) {
  __nanosleep(nanos);
}

void gpu_nsleep(unsigned nanos, cudaStream_t stream) {
  nanosleepX<<<1,1,0,stream>>>(nanos);
}
