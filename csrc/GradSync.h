#pragma once

#include <c10/cuda/CUDAStream.h>
#include <torch/torch.h>

#include <vector>

#include "runtime.h"

class CommunicationHandler;

class GradientSyncGroup {
  static constexpr size_t kBytesPerParam = 4;

 public:
  /* return true if more than sync_bucket_size bytes have been registered in
   * this group */
  bool RegisterParameter(torch::Tensor param) {
    param_group.push_back(param);
    nelem += param.mutable_grad().numel();
    return rtctx->sync_bucket_size &&
           nelem * kBytesPerParam >= rtctx->sync_bucket_size;
  }
  void Coalesce();
  void Sync(size_t key, c10::cuda::CUDAStream stream,
            std::shared_ptr<CommunicationHandler> &commHandler);

 private:
  torch::Tensor buf;
  long nelem{0};
  std::vector<torch::Tensor> param_group;
};