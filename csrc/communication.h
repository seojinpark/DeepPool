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

#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <torch/torch.h>
#include <c10d/Types.hpp>
#include <cuda_runtime.h>
#include "json.hpp"
#include "rpcService.h"

#include "utils.h"

#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#include <torch/csrc/cuda/nccl.h>


using json = nlohmann::json;

/**
 * Forward declarations. Do not include headers unless necessary.
 */
class RuntimeContext;

class CommunicationHandler {
 public:
  
  CommunicationHandler(int worldSize, json tensorTags, int rank,
      json jobRankToGlobalRank, c10::Device device, bool tensorInCuda = true);
  virtual ~CommunicationHandler() { }

  /**
   * Changes from Python runtime.
   * - Compute tensor dimension from json spec.
   * - recv takes empty tensor that is ready to be filled.
   * - No separate async/sync methods.
   * 
   * Undecided: take tensorName or tag? maybe just take tag? It may not be
   * that difficult to save the tag in runnableModule's layer..?
   */
  virtual void send(const torch::Tensor& tensor, int tag, int dest,
                    bool async = false) = 0;
  virtual void recv(torch::Tensor& tensor, int tag, int src,
                    bool async = false) = 0;

  virtual void all_reduce(torch::Tensor& tensor, c10d::ReduceOp op, bool async = false) = 0;

  /* block until all outstanding send/recvs have completed */
  virtual void sync(c10::optional<c10::cuda::CUDAStream> stream = {}) = 0;

  /* used to prepare streams for cuda graph capture, WIP */
  virtual void precapture() = 0;
  virtual void postcapture() = 0;
  virtual void comm_start(c10::optional<c10::cuda::CUDAStream> stream = {}) = 0;
  virtual void comm_end() = 0;

  /**
   * Returns the tag for p2p communication send/recv.
   *
   * \param xferName  Transfer name specificed in spec. Sender and receiver
   *                  should use the same xferName.
   */
  int getTag(const std::string& xferName);

  inline c10::Device getDev() { return device; }
  
 protected:
  int worldSize;
  json tensorTags;
  int rank;
  json jobRankToGlobalRank;
  c10::Device device;
  bool tensorInCuda;
};

class CommunicationHandlerNCCL : public CommunicationHandler {
 public:
  CommunicationHandlerNCCL(RuntimeContext* rtctx, std::string taskName,
      int worldSize, json tensorTags, int rank, json jobRankToGlobalRank,
      c10::Device dev, bool tensorInCuda = true);
  ~CommunicationHandlerNCCL();

  void send(const torch::Tensor& tensor, int tag, int dest,
            bool async = false);
  void recv(torch::Tensor& tensor, int tag, int src,
            bool async = false);
  void sync(c10::optional<c10::cuda::CUDAStream> stream = {});
  void precapture();
  void postcapture();
  void comm_start(c10::optional<c10::cuda::CUDAStream> stream = {});
  void comm_end();

  void all_reduce(torch::Tensor& tensor, c10d::ReduceOp op, bool async = false);
  void testRingP2P();
  void testAllReduce();

 private:
  RuntimeContext* rtctx;
  std::string taskName;
  std::mutex _mutex;                // Monitor lock.
  std::unordered_map<int, std::string> receivedData;
  std::unordered_map<int, std::unique_ptr<RuntimeClient> > clientPool;

  at::cuda::CUDAEvent sync_event;

  c10::cuda::CUDAStream default_comm_stream;

  c10::optional<c10::cuda::CUDAStream> group_call_stream;
  bool in_group_call{false};
};

class CommunicationHandlerGRPC : public CommunicationHandler {
 public:
  CommunicationHandlerGRPC(RuntimeContext* rtctx, std::string taskName,
      int worldSize, json tensorTags, int rank, json jobRankToGlobalRank,
      c10::Device dev, bool tensorInCuda = true);
  
  void saveData(const std::string& tensorData, int tag);
  void send(const torch::Tensor& tensor, int tag, int dest,
            bool async = false);
  void recv(torch::Tensor& tensor, int tag, int src,
            bool async = false);
  void testRingP2P();
  void sync(c10::optional<c10::cuda::CUDAStream> stream = {}) { UNUSED(stream); };
  void precapture() {};
  void postcapture() {};
  void comm_start(c10::optional<c10::cuda::CUDAStream> stream = {}) { UNUSED(stream); };
  void comm_end() {};

  void all_reduce(torch::Tensor& tensor, c10d::ReduceOp op, bool async = false);


 private:
  RuntimeContext* rtctx;
  std::string taskName;
  std::mutex _mutex;                // Monitor lock.
  std::unordered_map<int, std::string> receivedData;
  std::unordered_map<int, std::unique_ptr<RuntimeClient> > clientPool;
};

#endif
