#include "GradSync.h"

#include <torch/torch.h>

#include "communication.h"
#include "logger.h"
#include "runtime.h"

void GradientSyncManager::FlushKey(size_t key) {
  auto &vec = grads_by_key_[key];

  assert(vec.size());

  // TODO possible bug here (or in NCCL): with graphs and grouped all reduces,
  // model performance was a little worse
  // TODO - using separate stream causing NCCL to hang with graphs, try again at some pt.
  commHandler_->comm_start({}, key);
  for (auto &grad : vec)
    commHandler_->all_reduce(grad, c10d::ReduceOp::SUM);
  commHandler_->comm_end();

  has_unjoined_work_ = true;
  vec.clear();
  pending_bytes_by_key_[key] = 0;
}

void GradientSyncManager::Flush() {
  for (auto &kp : pending_bytes_by_key_)
    if (kp.second) FlushKey(kp.first);
}

void GradientSyncManager::AddGradient(torch::Tensor grad,
                                      size_t comm_group_key) {
  assert(comm_group_key > 0);

  auto &vec = grads_by_key_[comm_group_key];
  vec.push_back(grad);
  pending_bytes_by_key_[comm_group_key] += grad.nbytes();

  if (flush_threshold_bytes_ &&
      pending_bytes_by_key_[comm_group_key] >= flush_threshold_bytes_)
      FlushKey(comm_group_key);
}

void GradientSyncManager::Join() {
  if (has_unjoined_work_) {
    commHandler_->sync();
    has_unjoined_work_ = false;
  }
}