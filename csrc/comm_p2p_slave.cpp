#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "communication.h"

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

int comm_p2p_slave(void)
{
  ncclComm_t comms[1];

  //managing 1 device(s) per node & 2 rank(s) in total
  int nDev = 1;
  int size = 32*1024*1024;
  int devs[1] = { 0 };
  int nRanks = 2;
  int myRank = 1; // hard-coded rank 1 for slave

  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  for (int i = 0; i < nDev; ++i) {
    printf("Setting device %d\n", i);
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  //initializing NCCL
  //NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
  ncclUniqueId cliqueId;
  if (myRank == 0) ncclGetUniqueId(&cliqueId);
  else {
    memset(cliqueId.internal, 0, sizeof(cliqueId));
    cliqueId.internal[0] = 0x2;
    cliqueId.internal[1] = 0x0;
    cliqueId.internal[2] = 0xFFFFFFE3;
    cliqueId.internal[3] = 0x5F;
    cliqueId.internal[4] = 0xFFFFFFAC;
    cliqueId.internal[5] = 0x1F;
    cliqueId.internal[6] = 0x5C;
    cliqueId.internal[7] = 0x27;
  }
  printf("ncclUniqueId = \n");
  for (int i = 0; i < 8; i++) printf("\t 0x%X\n", cliqueId.internal[i]);

#if 0
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    NCCLCHECK(ncclCommInitRank(comms, nRanks, cliqueId, myRank));
  }
  NCCLCHECK(ncclGroupEnd());
#endif

  NCCLCHECK(ncclCommInitRank(comms, nRanks, cliqueId, myRank));
  printf("ncclCommInitRank done!\n");

   //calling NCCL communication API. Group API is required when using
   //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    printf("rank %d sending to rank %d\n", myRank, (myRank+1)%2);
    NCCLCHECK(ncclSend((void*)sendbuff[i], size, ncclFloat, (myRank+1)%2, comms[i], s[i]));
    printf("rank %d recving from rank %d\n", myRank, (myRank+1)%2);
    NCCLCHECK(ncclRecv((void*)recvbuff[i], size, ncclFloat, (myRank+1)%2, comms[i], s[i]));
    /*NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
    */
  }
  NCCLCHECK(ncclGroupEnd());

  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }


  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);


  printf("Success \n");
  return 0;
}
