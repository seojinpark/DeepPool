#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <torch/torch.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include "communication.h"
#include <iostream>

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

int comm_p2p_master(void)
{
  ncclComm_t comms[1];

  //managing 1 device(s) per node & 2 rank(s) in total
  int nDev = 1;
  int size = 9; // 32*1024*1024;
  int devs[1] = { 0 };
  int nRanks = 2;
  int myRank = 0; // hard-coded rank 0 for master

  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  for (int i = 0; i < nDev; ++i) {
    printf("Setting device %d\n", i);
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(sendbuff[i], 0x44, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0, size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  //initializing NCCL
  //NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
  ncclUniqueId cliqueId;
  if (myRank == 0) ncclGetUniqueId(&cliqueId);
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

  torch::Tensor send_tensor = torch::ones({3,3}, torch::Device(torch::kCUDA, 0));
  torch::Tensor recv_tensor = torch::zeros({9}, torch::Device(torch::kCUDA, 0));

  CUDACHECK(cudaMemcpy(send_tensor.data_ptr(), sendbuff[0], size*sizeof(float), cudaMemcpyDeviceToDevice));
  CUDACHECK(cudaMemcpy(recv_tensor.data_ptr(), recvbuff[0], size*sizeof(float), cudaMemcpyDeviceToDevice));

  std::cout << send_tensor << std::endl;
  std::cout << send_tensor.data_ptr() << std::endl;

  std::cout << recv_tensor << std::endl;
  std::cout << recv_tensor.data_ptr() << std::endl;

#if 0
  float *h_sendbuff = (float *)malloc(size * sizeof(float));
  float *h_recvbuff = (float *)malloc(size * sizeof(float));

  CUDACHECK(cudaMemcpy(h_sendbuff, sendbuff[0], size*sizeof(float), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_recvbuff, recvbuff[0], size*sizeof(float), cudaMemcpyDeviceToHost));

  printf("CUDA send tensor before comm:\n");
  for (int i = 0; i < size; i++) printf("%f\n", h_sendbuff[i]);
  printf("CUDA recv tensor before comm:\n");
  for (int i = 0; i < size; i++) printf("%f\n", h_recvbuff[i]);
#endif

   //calling NCCL communication API. Group API is required when using
   //multiple devices per thread
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i) {
    printf("rank %d sending to rank %d\n", myRank, (myRank+1)%2);
    NCCLCHECK(ncclSend((void*)send_tensor.data_ptr(), size, ncclFloat, (myRank+1)%2, comms[i], s[i]));
    printf("rank %d recving from rank %d\n", myRank, (myRank+1)%2);
    NCCLCHECK(ncclRecv((void*)recv_tensor.data_ptr(), size, ncclFloat, (myRank+1)%2, comms[i], s[i]));
    /*NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], s[i]));
    */
  }
  NCCLCHECK(ncclGroupEnd());

#if 0
  CUDACHECK(cudaMemcpy(h_sendbuff, sendbuff[0], size*sizeof(float), cudaMemcpyDeviceToHost));
  CUDACHECK(cudaMemcpy(h_recvbuff, recvbuff[0], size*sizeof(float), cudaMemcpyDeviceToHost));

  printf("CUDA sendbuff after comm:\n");
  for (int i = 0; i < size; i++) printf("%f\n", h_sendbuff[i]);
  printf("CUDA recvbuff after comm:\n");
  for (int i = 0; i < size; i++) printf("%f\n", h_recvbuff[i]);
#endif

  std::cout << send_tensor << std::endl;
  std::cout << send_tensor.data_ptr() << std::endl;

  std::cout << recv_tensor << std::endl;
  std::cout << recv_tensor.data_ptr() << std::endl;

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
