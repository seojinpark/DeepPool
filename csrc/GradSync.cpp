#include "GradSync.h"

#include <torch/torch.h>

#include "communication.h"
#include "runtime.h"

using namespace torch::indexing;

void GradientSyncGroup::Coalesce() {
  if (rtctx->disable_grad_sync || !rtctx->sync_coalesce || !nelem) return;

  /* pad contigious buffer if needed */
  if (nelem % rtctx->sync_tensor_pad != 0)
    nelem += rtctx->sync_tensor_pad - (nelem % rtctx->sync_tensor_pad);

  /* create new contiguous buffer */
  buf = torch::empty({nelem}, param_group[0].mutable_grad().options());

  size_t nelem_offset = 0;
  for (auto &p : param_group) {
    auto &gr = p.mutable_grad();
    auto dst =
        buf.index({Slice(nelem_offset, nelem_offset + gr.numel())}).view_as(gr);

    /* copy gradients into contiguous buffer */
    dst.copy_(gr, true);
    nelem_offset += gr.numel();

    /* reassign gradient as view into buffer */
    p.mutable_grad() = dst;
  }
}

void GradientSyncGroup::Sync(
    size_t key, c10::cuda::CUDAStream stream,
    std::shared_ptr<CommunicationHandler> &commHandler) {
  if (rtctx->disable_grad_sync || !nelem) return;

  commHandler->comm_start(stream, key);
  if (rtctx->sync_coalesce) {
    commHandler->all_reduce(buf, c10d::ReduceOp::SUM);
  } else {
    for (auto &p : param_group)
      commHandler->all_reduce(p.mutable_grad(), c10d::ReduceOp::SUM);
  }
  commHandler->comm_end();
}