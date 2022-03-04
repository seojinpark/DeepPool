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

#include "runnableModule.h"

#include <ATen/autocast_mode.h>
#include <torch/script.h>
#include <torch/torch.h>

#include "JobContext.h"
#include "communication.h"
#include "json.hpp"
#include "logger.h"
#include "runtime.h"
#include "tracer.h"
#include "utils.h"

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class GraphTimer {
 public:
  static constexpr int kTrials = 200;
  GraphTimer(){};
  void StartCapture() {
    c10::cuda::device_synchronize();
    c10::cuda::CUDACachingAllocator::emptyCache();
    c10::cuda::device_synchronize();
    graph.capture_begin();
  }
  int64_t EndCaptureAndTime() {
    graph.capture_end();
    c10::cuda::device_synchronize();
    CpuTimer timer("timer");
    timer.start();
    for (int i = 0; i < kTrials; ++i) graph.replay();
    c10::cuda::device_synchronize();
    timer.stop();
    return timer.avgMicros() / kTrials;
  }

 private:
  DeepPool::CUDAGraph graph;
};

/**
 * Constructs RunnableModule
 */
RunnableModule::RunnableModule(
    json spec, std::shared_ptr<CommunicationHandler> commHandler,
    LossFunctions lf)
    : commHandler(commHandler),
      sync_manager_(commHandler, rtctx->sync_bucket_size),
      lossfn_(lf) {
  auto& layersInJson = spec["layers"];
  long initialBatchSize = layersInJson[0]["config"][0];

  globalBatchSize = spec["globalBatchSize"].get<long>();
  for (long sample : spec["sampleIndices"]) sampleIndices.push_back(sample);
  for (long batch : spec["initialBatchSizes"])
    initialBatchSizes.push_back(batch);

  assert(spec["rank"].get<int>() == rtctx->rank);
  DP_LOG(DEBUG, "Constructing runnable module.. rank:%d", rtctx->rank);
  DP_LOG(DEBUG, "             initialBatchSize:%ld", initialBatchSize);
  DP_LOG(DEBUG, "             layersInJson's size:%lu (from spec)",
         spec["layers"].size());
  DP_LOG(DEBUG, "             layersInJson's size:%lu", layersInJson.size());

  // It's important to reserve the same, so that layers won't get copied over
  // to another address.. (layer's are pointing each other with raw pointer.)
  layers.reserve(layersInJson.size());

  for (auto& ldsc : layersInJson) {
    int id = ldsc["id"].get<int>();
    std::string name = ldsc["name"].get<std::string>();
    std::string moduleLoc = ldsc["moduleSavedLocation"].get<std::string>();
    DP_LOG(DEBUG, " %d-th layer's name: %s, moduleLoc: %s", id, name.c_str(),
           moduleLoc.c_str());

    SpecialModuleTypes specialModule = SpecialModuleTypes::NOTSPECIAL;
    if (name == "concat") specialModule = SpecialModuleTypes::CONCAT;

    DP_LOG(DEBUG, " layer's module is loaded.");
    DP_LOG(DEBUG, " layer is%s concat.", name == "concat" ? "" : " not");

    // module.to(device);
    DP_LOG(DEBUG, " layer's module is moved to device and set for train mode.");

    long layerLocalBatch = ldsc["config"][0].get<long>();
    bool layerIsActive = layerLocalBatch > 0;

    torch::jit::Module module;

    if (layerIsActive) {
      module = torch::jit::load(moduleLoc);
      module.to(rtctx->c10dev);
      module.train();
    }

    bool doLocalGradSync = layerIsActive && ldsc["gpuAssignment"].size() > 1;
    doLocalGradSync &=
        rtctx->min_layer_sync <= static_cast<size_t>(rtctx->worldSize);

    auto layer = std::make_shared<Layer>(module, specialModule, id,
                                         layerIsActive, doLocalGradSync);
    if (layer->active){
      for (const auto& params : layer->module.parameters()) {
        parameters.push_back(params);
      }
    }

    layers.push_back(layer);

    layer->commGroupKey = RankVecToKey(ldsc["gpuAssignment"]);
    if (doLocalGradSync) {
      if (rtctx->nccl_groups.count(layer->commGroupKey) == 0) {
        DP_LOG(ERROR,
               "Runtime does not have a nccl communicator group for %lu\n",
               layer->commGroupKey);
      }
    }

    DP_LOG(DEBUG, " %d-th layer's name: %s, doLocalGradSync: %d, key: %lu", id,
           name.c_str(), doLocalGradSync, layer->commGroupKey);

    int last_lid = -1;
    for (auto& plidjson : ldsc["prevLayers"]) {
      int plid = plidjson.get<int>();
      // ensure that prevLayers is sorted
      assert(plid > last_lid);
      last_lid = plid;
      layer->prevLayers.push_back(layers.at(plid));
      layers.at(plid)->nextLayers.push_back(layer);
      layer->nr_current_depedencies++;
    }

    layer->layerLocalBatch = layerLocalBatch;

    layer->emptyOutSizes.push_back(0);
    for (int size : ldsc["outputDim"]) layer->emptyOutSizes.push_back(size);

    if (ldsc.contains("xfers")) {
      for (auto& item : ldsc["xfers"]) {
        size_t src_lid = item["prop"]["prevLayerId"];

        Xfer n;
        n.src = std::make_pair(item["src"], item["prop"]["txSampleOffset"]);
        n.dst = std::make_pair(item["dest"], item["prop"]["rxSampleOffset"]);
        n.nr_samples = item["prop"]["xferSamples"];
        n.src_lid = src_lid;
        n.tag = commHandler->getTag(item["name"].get<std::string>());

        if (n.src.first == n.dst.first)
          layer->xfers_local.push_back(n);
        else
          layer->xfers.push_back(n);

        if (n.dst.first == static_cast<size_t>(rtctx->rank))
          layer->rx_lids.insert(src_lid);

        if (n.src.first == static_cast<size_t>(rtctx->rank))
          layer->tx_lids.insert(src_lid);
      }
    }
    // layer->moduleName = name + ldsc["params"].dump() + "[" +
    //                     std::to_string(layerLocalBatch) + "]" +
    //                     ldsc["inputDim"].dump();
    // std::cout << "layer->moduleName.c_str()  " << layer->moduleName.c_str() << std::endl;

    if (rtctx->profile_layer_times_graph) {
      // TODO: if it's not that accurate, maybe add layer id?
      layer->moduleName = name + ldsc["params"].dump() + "[" +
                          std::to_string(layerLocalBatch) + "]" +
                          ldsc["inputDim"].dump();
      DP_LOG(DEBUG, "moduleName: %s", layer->moduleName.c_str());
    } else {
      layer->fwUsec = ldsc["gpuTime"][0].get<int>();
      layer->bwUsec = ldsc["gpuTime"][1].get<int>();
    }
  }

  for (auto& layer : layers) {
    DP_LOG(DEBUG, "lid: %d, xferOuts: %lu, xferIns: %lu", layer->id,
           layer->tx_lids.size(), layer->rx_lids.size());
  }

  for (auto& layer : layers) {
    DP_LOG(DEBUG, "lid: %d, fwUsec: %" PRId64 ", bwUsec: %" PRId64 "",
           layer->id, layer->fwUsec, layer->bwUsec);
  }

  SetupOptimizer();

  loss_tracker_ = torch::zeros({1}).to(rtctx->c10dev);

  /* Debug check */
  for (auto& l : layers) {
    for (auto& p : l->prevLayers)
      assert(std::find(p->nextLayers.begin(), p->nextLayers.end(), l) !=
             p->nextLayers.end());
    for (auto& p : l->nextLayers)
      assert(std::find(p->prevLayers.begin(), p->prevLayers.end(), l) !=
             p->prevLayers.end());
  }
}

void RunnableModule::SetMode(bool train) {
  if (train == isTrain_) return;
  isTrain_ = train;
  ResetGraphs();
  for (auto& l : layers) {
    if (!l->active) continue;
    if (train)
      l->module.train();
    else
      l->module.eval();
  }
  loss_tracker_ = torch::zeros({1}).to(rtctx->c10dev);
  nr_iters_ = 0;
}

/**
 * Dumps the entire model parameters into the given vector.
 */
void RunnableModule::SetupOptimizer() {
  std::vector<torch::Tensor> parameters;
  for (auto& layer : layers) {
    if (!layer->active) continue;
    for (const auto& params : layer->module.parameters()) {
      parameters.push_back(params);
    }
  }

#ifdef ENABLE_STREAMING_DATASET
  auto options = torch::optim::RMSpropOptions().lr(0.0001).momentum(0.9).alpha(0.9).eps(0.001).weight_decay(0.00001);
  optimizer = std::make_shared<torch::optim::RMSprop>(parameters, options);
  lrsched = std::make_unique<torch::optim::StepLR>(*optimizer, 10, 0.9);
#else
  optimizer = std::make_unique<torch::optim::SGD>(parameters, /*lr=*/0.01);
#endif
}

void RunnableModule::SetInputsTargets(torch::Tensor input,
                                      torch::Tensor target,
                                      torch::Tensor weight) {
  assert(state == JobState::INIT);

  size_t input_nb = input.defined() ? input.nbytes() : 0;

  if (input_nb) {
    /* should already be on device */
    assert(input.is_cuda());

    if (!input_buf.defined() || input_nb != input_buf.nbytes()) {
      /* reallocate input buffer */
      assert(!has_graph);
      input_buf = torch::empty(input.sizes(), input.options());
      assert(input_buf.is_cuda());
    }

    input_buf.copy_(input, /*non_blocking=*/true);
    input.record_stream(rtctx->torch_stream);
  }

  size_t target_nb = target.defined() ? target.nbytes() : 0;

  if (target_nb) {
    /* should already be on device */
    assert(target.is_cuda());

    if (!target_buf.defined() || target_nb != target_buf.nbytes()) {
      /* reallocate target buffer */
      assert(!has_graph);
      target_buf = torch::empty(target.sizes(), target.options());
      assert(target_buf.is_cuda());
    }

    target_buf.copy_(target, /*non_blocking=*/true);
    target.record_stream(rtctx->torch_stream);
  }

  size_t weight_nb = weight.defined() ? weight.nbytes() : 0;

  if (weight_nb) {
    /* should already be on device */
    assert(weight.is_cuda());

    if (!weight_buf.defined() || weight_nb != weight_buf.nbytes()) {
      /* reallocate target buffer */
      assert(!has_graph);
      weight_buf = torch::empty(weight.sizes(), weight.options());
      assert(weight_buf.is_cuda());
    }

    weight_buf.copy_(weight, /*non_blocking=*/true);
    weight.record_stream(rtctx->torch_stream);
  }

  /* enqueue input to first layer */
  layers[0]->tensors_in[0] = input_buf;
  fpTargets = target_buf;
  fpWeights = weight_buf;
}

torch::Tensor Layer::DoForward(bool captureLayer) {
  if (!active) {
    DP_LOG(DEBUG, "Layer %d is not active.", id);
    return {};
  }

  DP_LOG(DEBUG, "Layer %d is active.", id);
  assert(tensors_in.size());

  std::vector<torch::Tensor> vec;
  for (auto& p : tensors_in) vec.push_back(p.second);

  std::vector<c10::IValue> iVec;
  if (specialModule == SpecialModuleTypes::CONCAT &&
      module.get_method("forward").function().getSchema().arguments().size() <=
          2) {
    /* old list-argument concat */
    iVec.emplace_back(vec);
  } else {
    for (auto& v : vec) iVec.emplace_back(v);
  }

  GraphTimer fwdtimer;
  if (captureLayer) fwdtimer.StartCapture();

  output = module.forward(iVec).toTensor();
  /* verify output shape is as expected per job description */
  for (size_t i = 1; i < emptyOutSizes.size(); i++)
    assert(emptyOutSizes[i] == output.sizes().vec()[i]);
  assert(emptyOutSizes.size() == output.sizes().vec().size());

  if (captureLayer) fwUsec = fwdtimer.EndCaptureAndTime();

  for (auto& nl : nextLayers)
    nl->tensors_in[id] =
        output.detach().requires_grad_(output.is_floating_point());

  return output;
}

void Layer::DoBackward(bool captureLayer, torch::Tensor& fpOutput) {
  if (!active) {
    DP_LOG(DEBUG, "Layer %d is not active.", id);
    return;
  }

  if (!output.requires_grad()) {
    DP_LOG(DEBUG, "Layer %d does not require grad", id);
    return;
  }

  DP_LOG(DEBUG, "Layer %d is active.", id);

  GraphTimer bwdtimer;
  if (captureLayer) bwdtimer.StartCapture();

  /* last layer */
  if (nextLayers.size() == 0 && fpOutput.defined()) {
    DP_LOG(DEBUG, "Backward on fpLoss:%s", tsrSizeToStr(fpOutput).c_str());
    fpOutput.backward();
    fpOutput.reset();
  }

  for (size_t nli = 0; nli < nextLayers.size(); nli++) {
    auto& nl = nextLayers[nli];
    auto& grad = nl->tensors_in[id];
    DP_LOG(DEBUG, "Backward on output:%s grad(%d):%s",
           tsrSizeToStr(output).c_str(), nl->id, tsrSizeToStr(grad).c_str());
    output.backward(grad, nli < nextLayers.size() - 1);
    nl->tensors_in[id].reset();
  }

  if (captureLayer) bwUsec = bwdtimer.EndCaptureAndTime();

  for (auto& pLid : prevLayers) {
    tensors_in[pLid->id] = tensors_in[pLid->id].grad();
    DP_LOG(DEBUG, "Sending backward grad from %d to %d (size %s)", id, pLid->id,
           tsrSizeToStr(tensors_in[pLid->id]).c_str());
  }

  output.reset();
}

// TODO maybe some cleaner way to do in code, but this shouldn't impact
// performance
static torch::Tensor getSampleSlice(torch::Tensor& in, ssize_t offset,
                                    ssize_t nr_samples) {
  ssize_t cur_batch = in.sizes()[0];
  assert(offset + nr_samples <= cur_batch);
  std::vector<long> splits = {offset, nr_samples,
                              cur_batch - offset - nr_samples};
  std::vector<torch::Tensor> subs = in.split_with_sizes(splits);
  return subs[1];
}

/* Prepapre samples needed to process this layer */
void RunnableModule::ExecuteXfers(Layer* layer, bool backward) {
  if (!layer->xfers.size() && !layer->xfers_local.size()) return;

  DP_LOG(DEBUG, "Executing xfers for layer %d", layer->id);

  std::map<size_t, torch::Tensor> inbound_tensors;
  std::map<size_t, torch::Tensor> outbound_tensors;

  torch::TensorOptions topts(rtctx->c10dev);
  topts = topts.requires_grad(!backward);
  if (at::autocast::is_enabled())
    topts = topts.dtype(at::autocast::get_autocast_gpu_dtype());

  auto& outbound_lids = backward ? layer->rx_lids : layer->tx_lids;
  auto& inbound_lids = backward ? layer->tx_lids : layer->rx_lids;

  /* prepare pointers to outbound samples */
  for (size_t lid : outbound_lids)
    outbound_tensors[lid] = layer->tensors_in[lid];

  /* prepare pointers for inbound samples */
  for (size_t lid : inbound_lids) {
    std::vector<long> inDim = layers.at(lid)->emptyOutSizes;
    inDim[0] =
        backward ? layers.at(lid)->layerLocalBatch : layer->layerLocalBatch;

    torch::Tensor in = torch::empty(inDim, topts);
    layer->tensors_in[lid] = in;
    inbound_tensors[lid] = in;
  }

  if (layer->xfers.size()) commHandler->comm_start();

  bool did_recv = false;
  for (const auto& ixfer : layer->xfers) {
    const size_t lid = ixfer.src_lid;
    const std::pair<size_t, size_t>& src = backward ? ixfer.dst : ixfer.src;
    const std::pair<size_t, size_t>& dst = backward ? ixfer.src : ixfer.dst;

    DP_LOG(
        DEBUG,
        "Transferring %lu samples from rank %lu layer %lu pos %lu to rank %lu "
        "layer %d pos %lu",
        ixfer.nr_samples, src.first, lid, src.second, dst.first, layer->id,
        dst.second);

    if (src.first == static_cast<size_t>(rtctx->rank)) {
      auto tsr = getSampleSlice(outbound_tensors.at(lid), src.second,
                                ixfer.nr_samples);
      commHandler->send(tsr, ixfer.tag, dst.first);
    } else {
      assert(dst.first == static_cast<size_t>(rtctx->rank));
      auto tsr =
          getSampleSlice(inbound_tensors.at(lid), dst.second, ixfer.nr_samples);
      commHandler->recv(tsr, ixfer.tag, src.first);
      did_recv = true;
    }
  }

  if (layer->xfers.size()) commHandler->comm_end();

  for (const auto& ixfer : layer->xfers_local) {
    const size_t lid = ixfer.src_lid;
    const std::pair<size_t, size_t>& src = backward ? ixfer.dst : ixfer.src;
    const std::pair<size_t, size_t>& dst = backward ? ixfer.src : ixfer.dst;

    DP_LOG(DEBUG,
           "Copying %lu samples from layer %lu pos %lu to "
           "layer %d pos %lu",
           ixfer.nr_samples, lid, src.second, layer->id, dst.second);

    auto srcTsr =
        getSampleSlice(outbound_tensors.at(lid), src.second, ixfer.nr_samples);
    auto dstTsr =
        getSampleSlice(inbound_tensors.at(lid), dst.second, ixfer.nr_samples);

    CUDACHECK(cudaMemcpyAsync(dstTsr.data_ptr(), srcTsr.data_ptr(),
                              srcTsr.nbytes(), cudaMemcpyDeviceToDevice,
                              rtctx->torch_stream));
  }

  if (did_recv) commHandler->sync();
}

/**
 * Execute a forward pass of this model.
 *
 * \return Returns true if forward pass is completed.
 */
JobStatus RunnableModule::forwardAStep(bool captureLayer) {
  assert(layerQ.size() > 0);
  Layer* layer = layerQ.front();
  layerQ.pop_front();

  assert(layer->nr_current_depedencies == 0);
  assert(layer->status == LayerStatus::PENDING_FP);

  ExecuteXfers(layer);
  torch::Tensor output = layer->DoForward(captureLayer);
  layer->status = LayerStatus::PENDING_BP;
  layer->nr_current_depedencies = layer->nextLayers.size();

  DP_LOG(DEBUG, " ** Layer %d is processed.", layer->id);

  for (auto& nl : layer->nextLayers) {
    assert(nl->nr_current_depedencies > 0);
    if (--nl->nr_current_depedencies == 0) layerQ.push_back(nl.get());
  }

  if (!isTrain_) {
    layer->tensors_in.clear();
    layer->output.reset();
  }

  TimerRecordLayer(layer->timerkey, false);

  // Forward pass is completed.
  if (layerQ.empty()) {
    DP_LOG(DEBUG, "no more layers to process.");
    fpOutput = output;
    return COMPLETED;
  }

  return IN_PROGRESS;
}

/**
 * Compute the loss from forward pass.
 */
void RunnableModule::loss() {
  if (!fpOutput.defined()) return;

  if (lossfn_ == LossFunctions::CrossEntropyLoss) {
    auto loss_fct = torch::nn::CrossEntropyLoss();
    fpOutput = loss_fct(fpOutput, fpTargets.view({-1}));
#ifdef ENABLE_STREAMING_DATASET
  } else if (lossfn_ == LossFunctions::AnvilLoss) {
    auto loss_fct = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(torch::kNone));
    fpOutput = loss_fct(fpOutput, fpTargets);
    fpOutput = torch::mul(fpOutput,fpWeights);
    fpOutput = fpOutput.sum();
#endif
  } else {
    assert(lossfn_ == LossFunctions::NLLLoss);
    fpOutput = torch::nll_loss(fpOutput.log_softmax(1), fpTargets);
  }
  loss_tracker_ += fpOutput;
}

/**
 * Execute a backward pass of this model.
 *
 * \return Returns true if backward pass is completed.
 */
JobStatus RunnableModule::backwardAStep(bool captureLayer) {
  assert(layerQ.size() > 0);
  Layer* layer = layerQ.front();
  layerQ.pop_front();

  assert(layer->nr_current_depedencies == 0);
  assert(layer->status == LayerStatus::PENDING_BP);

  DP_LOG(DEBUG, "lid:%d.", layer->id);

  layer->DoBackward(captureLayer, fpOutput);
  ExecuteXfers(layer, true);
  layer->status = LayerStatus::PENDING_FP;
  layer->nr_current_depedencies = layer->prevLayers.size();

  if (layer->doLocalGradSync) {
    for (const auto& param : layer->module.parameters()) {
      if (param.mutable_grad().defined())
        sync_manager_.AddGradient(param.mutable_grad(), layer->commGroupKey);
    }
  }

  for (auto& pl : layer->prevLayers) {
    assert(pl->nr_current_depedencies > 0);
    if (--pl->nr_current_depedencies == 0) layerQ.push_back(pl.get());
  }

  assert(!layer->timerkey.empty());
  TimerRecordLayer(layer->timerkey, true);

  // Backward pass is completed.
  if (layerQ.empty()) {
    DP_LOG(DEBUG, "no more layers to process.");
    return COMPLETED;
  }
  return IN_PROGRESS;
}

/**
 * A helper to run a job.
 *
 * \param job   a context for the job to train.
 * \param[out] jobCompleted
 *    will be filled with true if the job is completely finished.
 *
 * \return    returns non-zero if iteration is finished.
 */
int RunnableModule::AdvanceTraining(bool doGraphCapture, bool layerProfile) {
  static CUDAPipeline p(2);  // Todo fix....

  doGraphCapture &= !has_graph;

  if (state == JobState::INIT) {
    DP_LOG(DEBUG, "JobState::INIT.");

    p.Lap();
    TimerRecordStage("start");

    layerQ.clear();
    layerQ.push_back(layers[0].get());

    if(!doGraphCapture)
      nr_iters_++;

    if (layers[0]->active && !layers[0]->tensors_in[0].defined())
      assert("MISSING INPUT TO FIRST LAYER!" && false);

    TimerRecordStage("load");

    /* start graph capture */
    if (doGraphCapture) {
      DP_LOG(NOTICE, "Starting capture.");
      graph_recording = true;
      c10::cuda::device_synchronize();
      graph_mempool = DeepPool::graph_pool_handle();
      fw_graph.capture_begin(graph_mempool);
      commHandler->precapture();
    } else if (has_graph) {
      /* skip to forward phase */
      state = JobState::FORWARD;
      TimerRecordStage("zero");
      return 0;
    }

    fpOutput.reset();
    for (auto& group : optimizer->param_groups())
      for (auto& param : group.params()) param.mutable_grad() = torch::Tensor();
    TimerRecordStage("zero");
    TimerRecordLayer("start", false);
    state = JobState::FORWARD;
    DP_LOG(DEBUG, "Foward pass is starting soon.");
  } else if (state == JobState::FORWARD) {
    DP_LOG(DEBUG, "JobState::FORWARD.");

    if (has_graph) {
      DP_LOG(DEBUG, "Replay iter.");
      main_fw_graph->Launch(rtctx->torch_stream);
      TimerRecordStage("forward");
      TimerRecordStage("loss");
      if (!isTrain_){
        TimerRecordStage("backward");
        TimerRecordStage("step");
        state = JobState::FINISH;
      }
      else
        state = JobState::BACKWARD;
      return 0;
    }

    JobStatus status = forwardAStep(layerProfile);

    if (status == COMPLETED) {
      TimerRecordStage("forward");
      if (doGraphCapture) {
        sync_manager_.Join();
        commHandler->postcapture();
        fw_graph.capture_end();
        if(isTrain_){
          bw_graph.capture_begin(graph_mempool);
          commHandler->precapture();
        }
      }
      if (!isTrain_) {
        for (auto& layer : layers) {
          layer->status = LayerStatus::PENDING_FP;
          layer->nr_current_depedencies = layer->prevLayers.size();
        }
        DP_LOG(DEBUG, "Foward pass is completed.");

        TimerRecordStage("loss");
        TimerRecordStage("backward");
        TimerRecordStage("step");
        state = JobState::FINISH;
      } else {
        DP_LOG(DEBUG, "Foward pass is completed. Calculating loss soon.");
        state = JobState::LOSS;
      }
    }
  } else if (state == JobState::LOSS) {
    DP_LOG(DEBUG, "JobState::LOSS.");
    loss();
    TimerRecordStage("loss");
    TimerRecordLayer("start", true);
    assert(layerQ.empty());
    layerQ.push_back(layers.back().get());
    DP_LOG(DEBUG, "Moving to backward pass.");
    state = JobState::BACKWARD;
  } else if (state == JobState::BACKWARD) {
    DP_LOG(DEBUG, "JobState::BACKWARD.");

    if (has_graph) {
      DP_LOG(DEBUG, "Replay iter.");
      main_bw_graph->Launch(rtctx->torch_stream);
      state = JobState::STEP;
      TimerRecordStage("backward");
      return 0;
    }

    JobStatus status = backwardAStep(layerProfile);
    if (status == COMPLETED) {
      TimerRecordStage("backward");
      state = JobState::SYNC;
      DP_LOG(DEBUG,
             "Backward pass is completed. Moving to gradient all-reduce.");
    }
  } else if (state == JobState::SYNC) {
    DP_LOG(DEBUG, "JobState::SYNC.");

    if (doGraphCapture) {
      sync_manager_.Join();
      commHandler->postcapture();
      bw_graph.capture_end();
      syncgraph.capture_begin(graph_mempool);
    }

    sync_manager_.Flush();
    sync_manager_.Join();

    if (doGraphCapture) {
      syncgraph.capture_end();
    }
    
    state = JobState::STEP;
  } else if (state == JobState::STEP) {
    DP_LOG(DEBUG, "JobState::STEP");
    if (has_graph) {
      step_graph->Launch(rtctx->torch_stream);
      state = JobState::FINISH;
      TimerRecordStage("step");
      return 0;
    }
     if (doGraphCapture) {
      stepgraph.capture_begin(graph_mempool);
      commHandler->precapture();
    }
    
    optimizer->step();
    TimerRecordStage("step");
    state = JobState::FINISH;
  } else if (state == JobState::FINISH) {
    DP_LOG(DEBUG, "JobState::FINISH");

    if (doGraphCapture) {
      float maingraphsplit = -1.0;  // 10.0;
      float stepgraphsplit = -1.0;  // 2.0;

      if (!isTrain_) {
        commHandler->postcapture();
        auto mainfwgraph_e =
            GraphPieces::GraphToExecs(fw_graph.getGRAPH(), maingraphsplit);
        main_fw_graph = GraphPieces::MergePieces({mainfwgraph_e});
      } else {
        commHandler->postcapture();
        stepgraph.capture_end();

        auto mainfwgraph_e =
            GraphPieces::GraphToExecs(fw_graph.getGRAPH(), maingraphsplit);
        main_fw_graph = GraphPieces::MergePieces({mainfwgraph_e});

        auto mainbwgraph_e =
            GraphPieces::GraphToExecs(bw_graph.getGRAPH(), maingraphsplit);
        auto syncgraph_e =
            GraphPieces::GraphToExecs(syncgraph.getGRAPH(), -1.0);
        main_bw_graph =
            GraphPieces::MergePieces({mainbwgraph_e, syncgraph_e});
        auto stepgraph_e =
            GraphPieces::GraphToExecs(stepgraph.getGRAPH(), stepgraphsplit);
        step_graph =
            GraphPieces::MergePieces({stepgraph_e});
      }
      has_graph = true;
      graph_recording = false;
      DP_LOG(NOTICE, "Ending capture.");

      //record zero for timers
      TimerRecordStage("zero");
      TimerRecordStage("forward");
      TimerRecordStage("loss");
      TimerRecordStage("backward");
      TimerRecordStage("step");
    }

    state = JobState::INIT;
    TimerRecordStage("stop");
    resetTimers();
    return 1;
  }
  return 0;
}

/**
 * Reset timers for profiling each layer. Happens every iteration.
 */
void RunnableModule::resetTimers() {
  if (graph_recording) return;

  if (rtctx->profile_stage_time) timers.SaveAndReset();

  if (!has_graph && rtctx->profile_layer_times_timers) {
    layerts_fwd.SaveAndReset();
    layerts_bwd.SaveAndReset();
  }
}

void RunnableModule::printLayerInGraphTimes() {
  if (rtctx->profile_layer_times_graph) {
    double sum = 0;
    for (auto& layer : layers) {
      double layerms =
          static_cast<double>(layer->fwUsec + layer->bwUsec) / 1000.0;
      printf(" %110s  %6.3f  %8" PRId64 "  %8" PRId64 "\n",
             layer->moduleName.c_str(), layerms, layer->fwUsec, layer->bwUsec);
      sum += layerms;
    }
    printf("%100s  %.3f\n", "SUM(avg)", sum);
  }

  if (rtctx->profile_layer_times_timers) {
    printf("layer fwd bwd\n");
    double total_fwd = 0, total_bwd = 0;
    for (auto& layer : layers) {
      auto fwdtm = layerts_fwd.GetP50(layer->timerkey, 200) * 1000.0;
      auto bwdtm = layerts_bwd.GetP50(layer->timerkey, 200) * 1000.0;
      total_fwd += fwdtm;
      total_bwd += bwdtm;
      printf("%s %.2f %.2f\n", layer->timerkey.c_str(), fwdtm, bwdtm);
    }
    printf("Total %.2f %2.f\n", total_fwd, total_bwd);
  }
}
