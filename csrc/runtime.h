// Copyright (c) 2021 MIT
//
// Permission to use, copy, modify, and distribute this software for any
// purpose with or without fee is hereby granted, provided that the above
// copyright notice and this permission notice appear in all copies.
//
// THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR(S) DISCLAIM ALL WARRANTIES
// WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL AUTHORS BE LIABLE FOR
// ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
// WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
// ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
// OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

#ifndef RUNTIME_H
#define RUNTIME_H

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>
#include <map>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/csrc/cuda/nccl.h>

#define VERBOSE 0

/**
 * Forward declarations
 */
class CommunicationHandler;
class RuntimeServiceImpl;
class RunnableModule;
class TaskManager;
namespace grpc {
  class Server;
};
namespace torch {
  namespace optim {
    class Optimizer;
  }
}

struct RuntimeContext_params {
  RuntimeContext_params() : coordinatorAddr(0), myAddr(0), device(0), c10dBackend(0),
      c10dMasterPort(0), rank(), worldSize(), logdir(), be_batch_size(0),
      profile(true), profile_comms(false), debug(false), verify(false)
      {
      }

  ~RuntimeContext_params(){}; // Defined in cpp file because of incomplete unique_ptrs.

  /**
   * Populated by commandline arguments
   */
  char* coordinatorAddr;  // includes port number.
  char* myAddr;           // includes port number.
  int device;
  char* c10dBackend;
  int c10dMasterPort;
  int rank;
  int worldSize;
  char* logdir;
  int be_batch_size;
  bool profile, profile_comms;
  bool debug;
  bool verify;

  int samplePerKernel{32};
  int use_fg_graph{1};
  int use_be_graph{1};
  size_t iters_per_capture{4};
  std::string be_jit_file{"/DeepPool/beModules/resnet.jit"};
  size_t min_layer_sync{8};
  size_t sync_bucket_size{10 * 1000 * 1000};
  std::string bg_json_file {}; //{"/home/seojin/DeepPoolRuntime/beModules/wrnBgJobB32.json"};
};

/**
 * Context holding data for Runtime.
 */

/**
 * Context holding data for Runtime.
 */
struct RuntimeContext {
  
  RuntimeContext(const RuntimeContext_params params) : 
      coordinatorAddr(params.coordinatorAddr), myAddr(params.myAddr), device(params.device), c10dBackend(params.c10dBackend),
      c10dMasterPort(params.c10dMasterPort), rank(params.rank), worldSize(params.worldSize), logdir(params.logdir), be_batch_size(params.be_batch_size),
      profile(params.profile), profile_comms(params.profile_comms), debug(params.debug), verify(params.verify), homedir(0), c10dev(c10::DeviceType::CUDA, params.device),
      grpcService(), grpcServer(), taskManager(), shutdownRequested(),
      commHandlerMap(), rankToIpAndPort(), grpcCommReady(), 
      ncclGroupId(), ncclGroupSize(), ranks(), ncclCommReady(), ncclCommObj(),
      xfer_stream(c10::cuda::getStreamFromPool(true, params.device)),
      torch_stream(c10::cuda::getStreamFromPool(true, params.device)), 
      grad_sync_stream(c10::cuda::getStreamFromPool(true, params.device)) {

        c10::cuda::CUDAGuard(params.device);// ::CUDAGuard(params.device);

        assert(xfer_stream != NULL);
        assert(torch_stream != NULL);
        c10::cuda::setCurrentCUDAStream(torch_stream);
      }

  ~RuntimeContext(); // Defined in cpp file because of incomplete unique_ptrs.

  /**
   * Populated by commandline arguments
   */
  char* coordinatorAddr;  // includes port number.
  char* myAddr;           // includes port number.
  int device;
  char* c10dBackend;
  int c10dMasterPort;
  int rank;
  int worldSize;
  char* logdir;
  int be_batch_size;
  bool profile, profile_comms;
  bool debug;
  bool verify;
  char *homedir;
  c10::Device c10dev;
  int samplePerKernel{32};
  int use_fg_graph{1};
  int use_be_graph{1};
  size_t iters_per_capture{4};
  std::string be_jit_file{"/home/seojin/DeepPoolRuntime/beModules/resnet.jit"};
  size_t min_layer_sync{8};
  size_t sync_bucket_size{10 * 1000 * 1000};
  std::string bg_json_file {}; //{"/home/seojin/DeepPoolRuntime/beModules/wrnBgJobB32.json"};

  /**
   *  additional variables.
   */
  RuntimeServiceImpl* grpcService;
  grpc::Server* grpcServer;
  TaskManager* taskManager;
  std::atomic<bool> shutdownRequested;  // Set to true when coordinator shuts down.
  std::map< std::string, CommunicationHandler* > commHandlerMap;
  std::vector<std::string> rankToIpAndPort;
  std::atomic<bool> grpcCommReady;

  /**
   * variables to maintain per NCCL comm group
   * need to be expanded if one node participates in more than one comm group
   */

  torch::cuda::nccl::ncclUniqueId ncclGroupId;
  int ncclGroupSize;
  std::vector<int> ranks;
  std::atomic<bool> ncclCommReady{false};
  torch::cuda::nccl::ncclComm_t ncclCommObj;
  c10::cuda::CUDAStream xfer_stream;
  c10::cuda::CUDAStream torch_stream;
  c10::cuda::CUDAStream grad_sync_stream;

};

extern RuntimeContext *rtctx;


#endif // RUNTIME_H