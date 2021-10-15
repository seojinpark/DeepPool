#include <torch/torch.h>
#include <ATen/TensorIndexing.h>
#include <atomic>
#include <iostream>
#include <memory>
#include <string>
#include <getopt.h>
#include <unistd.h>    // For homedir
#include <sys/types.h> // For homedir
#include <pwd.h>       // For homedir
#include "nccl.h"
// #include "runtime.h"
// #include "taskManager.h"
// #include "utils.h"
// #include "logger.h"
// #include "rpcService.h"
// #include "json.hpp"
// #include "communication.h"



#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    printf("Test NCCL failure %s:%d '%s'\n",        \
        __FILE__,__LINE__,ncclGetErrorString(res)); \
  }                                                 \
} while(0)                                          \

typedef enum {
  testSuccess = 0,
  testInternalError = 1,
  testCudaError = 2,
  testNcclError = 3,
  testCuRandError = 4
} testResult_t;

// Relay errors up and trace
#define TESTCHECK(cmd) do {                         \
  testResult_t r = cmd;                             \
  if (r!= testSuccess) {                            \
    printf(" .. pid %d: Test failure %s:%d\n",      \
         getpid(),                                  \
        __FILE__,__LINE__);                         \
  }                                                 \
} while(0)                                          \

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


// Define a new Module.
struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module("fc1", torch::nn::Linear(784, 64));
    fc2 = register_module("fc2", torch::nn::Linear(64, 32));
    fc3 = register_module("fc3", torch::nn::Linear(32, 10));
  }

  // Implement the Net's algorithm.
  torch::Tensor forward(torch::Tensor x) {
    // Use one of many tensor manipulation functions.
    x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
    x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
    x = torch::relu(fc2->forward(x));
    x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main() {
  // Create a new Net.
  auto net = std::make_shared<Net>();
  (*net).to(torch::kCUDA);
  torch::Tensor test_tensor = torch::ones({32,28,28,1}, torch::Device(torch::kCUDA, 0));
  torch::Tensor tar_tensor = torch::ones({32}, torch::Device(torch::kCUDA, 0)).to(torch::kLong);
//   tar_tensor.slice(torch::indexing::None, 0); //index({ torch::indexing::Slice(None, 0)}) = 1;
//   tar_tensor.index({at::indexing::Slice(), 0}) = 0;

//   // Create a multi-threaded data loader for the MNIST dataset.
//   auto data_loader = torch::data::make_data_loader(
//       torch::data::datasets::MNIST("./data").map(
//           torch::data::transforms::Stack<>()),
//       /*batch_size=*/64);

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  torch::optim::SGD optimizer(net->parameters(), /*lr=*/0.01);

  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    std::cout << "epoch " << epoch << std::endl;
    size_t batch_index = 0;
    // // Iterate the data loader to yield batches from the dataset.
    // for (auto& batch : *data_loader) {
      // Reset gradients.
      optimizer.zero_grad();
      // Execute the model on the input data.
      torch::Tensor prediction = net->forward(test_tensor);
      // Compute a loss value to judge the prediction of our model.
      torch::Tensor loss = torch::nll_loss(prediction, tar_tensor);
      // Compute gradients of the loss w.r.t. the parameters of our model.
      loss.backward();
      // Update the parameters based on the calculated gradients.
      optimizer.step();
      // Output the loss and checkpoint every 100 batches.
    //   if (++batch_index % 10 == 0) {
        std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                  << " | Loss: " << loss.item<float>() << std::endl;
        // Serialize your model periodically as a checkpoint.
        // torch::save(net, "net.pt");
    //   }
    // }
  }

  torch::Tensor send_tensor = torch::ones({3,3}, torch::Device(torch::kCUDA, 0));
  torch::Tensor recv_tensor = torch::zeros({3,3}, torch::Device(torch::kCUDA, 1));

  ncclComm_t *comms = (ncclComm_t*)malloc(sizeof(ncclComm_t)*2);
  cudaStream_t streams[2];
  const int devs[2] = {0,1};
  NCCLCHECK(ncclCommInitAll(comms, 2, devs));  
//   cudaStream_t *s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*8);
  // rtctx->cudaStream = (cudaStream_t*)malloc(sizeof(cudaStream_t));
//   for (int i=0; i<8; i++) {
//     cudaSetDevice(i);
//     CUDACHECK(cudaStreamCreate(s[i]));
//   }

  for (int i=0; i<2; i++) {
    CUDACHECK(cudaSetDevice(i));
    // TESTCHECK(AllocateBuffs(sendbuffs+i, sendBytes, recvbuffs+i, recvBytes, expected+i, (size_t)maxBytes, nProcs*nThreads*nGpus));
    CUDACHECK(cudaStreamCreate(streams+i));
  }
//   cudaSetDevice(0);
//   CUDACHECK(cudaStreamCreate(s));

  int dest = 1;
  int src = 0;

  std::cout << send_tensor << std::endl;
  std::cout << recv_tensor << std::endl;

  ncclGroupStart();
  ncclSend((void*)send_tensor.data_ptr(), send_tensor.nbytes()/send_tensor.itemsize(), ncclFloat, dest, comms[0], streams[0]);
  ncclRecv((void*)recv_tensor.data_ptr(), recv_tensor.nbytes()/recv_tensor.itemsize(), ncclFloat, src, comms[1], streams[1]);
  ncclGroupEnd();

  std::cout << send_tensor << std::endl;
  std::cout << recv_tensor << std::endl;
  sleep(5);
}