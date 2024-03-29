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

#include <torch/script.h>
#include <torch/torch.h>
#include "json.hpp"
#include "runtime.h"
#include "runnableModule.h"
#include "taskManager.h"
#include "logger.h"
#include "utils.h"
#include "communication.h"
#include "tracer.h"

using torch::autograd::Variable;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

static uint64_t bytes_inflight = 0;
static std::vector<torch::Tensor> pending_grads;

////////////////////////////////////////////////
// TsrXferFunc
////////////////////////////////////////////////
Variable
TsrXferFunc::forward(AutogradContext* ctx, Variable x, TsrXfer* xfer)
{
  ctx->saved_data["xfer"] = reinterpret_cast<int64_t>(xfer);
  DP_LOG(DEBUG, "TsrXferFunc::forward entered.. type: %d", xfer->type);

  if (xfer->type == TsrXfer::Send) {
    std::vector<torch::Tensor> splittedTsrs =
        x.split_with_sizes(xfer->splitSizes, xfer->splitCatDim);
    assert(splittedTsrs.size() == xfer->xferTagAndRank.size() + 1);
    size_t i;
    xfer->commHandler->comm_start();
    for (i = 0; i < xfer->xferTagAndRank.size(); ++i) {
      Tag tag = xfer->xferTagAndRank[i].first;
      Rank dest = xfer->xferTagAndRank[i].second;
      torch::Tensor tsr = splittedTsrs[i];
      DP_LOG(DEBUG, "Sending tag:%d to R:%d with %s", tag, dest,
          tsrSizeToStr(tsr).c_str());

      xfer->commHandler->send(tsr, tag, dest, /*async*/ true);
    }
    xfer->commHandler->comm_end();
    return splittedTsrs[i];
  }
  else if (xfer->type == TsrXfer::Recv) {
    std::vector<int64_t> inputSizes = x.sizes().vec();
    std::vector<torch::Tensor> tsrList;
    size_t i;

    // TODO: allocate single tensor buffer and direct recvs into correct portions
    // assert(xfer->splitCatDim == 0);

    for (i = 0; i < xfer->xferTagAndRank.size(); ++i) {
      inputSizes[xfer->splitCatDim] = xfer->splitSizes[i];
      torch::TensorOptions topts(rtctx->c10dev);
      torch::Tensor tsr = torch::empty(inputSizes, topts);
      tsrList.push_back(tsr);
    }

    xfer->commHandler->comm_start();
    for (i = 0; i < xfer->xferTagAndRank.size(); ++i) {
      Tag tag = xfer->xferTagAndRank[i].first;
      Rank src = xfer->xferTagAndRank[i].second;
      auto &tsr = tsrList.at(i);
      // DP_LOG(DEBUG, "Receiving tag:%d from R:%d with tensor: %s, %s, %s", tag, src,
      //     tsr.toString().c_str(), tsrSizeToStr(tsr).c_str(), tsrToStr(tsr).c_str());
      DP_LOG(DEBUG, "Receiving tag:%d from R:%d with tensor: %s", tag, src,
          tsrSizeToStr(tsr).c_str());
      xfer->commHandler->recv(tsr, tag, src, /*async*/ true);
    }
    xfer->commHandler->comm_end();
    xfer->commHandler->sync();
    tsrList.push_back(x);
    DP_LOG(DEBUG, "Concating %lu tensors", tsrList.size());
    auto concated = torch::cat(tsrList, xfer->splitCatDim);
    DP_LOG(DEBUG, "Concated tensor: %s", tsrSizeToStr(concated).c_str());
    return concated;
  } else {
    DP_LOG(ERROR, "xfer type is %d, which is not supported.", xfer->type);
    return x;
  }
}

variable_list
TsrXferFunc::backward(AutogradContext* ctx, variable_list grad_output)
{
  TsrXfer* xfer = reinterpret_cast<TsrXfer*>(ctx->saved_data["xfer"].toInt());
  Variable x = grad_output[0];
  DP_LOG(DEBUG, "grad_output size: %d", (int)grad_output.size());

  if (xfer->type == TsrXfer::Recv) {
    std::vector<torch::Tensor> splittedTsrs =
        x.split_with_sizes(xfer->splitSizes, xfer->splitCatDim);
    assert(splittedTsrs.size() == xfer->xferTagAndRank.size() + 1);
    size_t i;
    xfer->commHandler->comm_start();
    for (i = 0; i < xfer->xferTagAndRankBack.size(); ++i) {
      Tag tag = xfer->xferTagAndRankBack[i].first;
      Rank dest = xfer->xferTagAndRankBack[i].second;
      torch::Tensor tsr = splittedTsrs[i];
      DP_LOG(DEBUG, "Sending tag:%d to R:%d with %s", tag, dest,
          tsrSizeToStr(tsr).c_str());
      xfer->commHandler->send(tsr, tag, dest, /*async*/ true);
    }
    xfer->commHandler->comm_end();

    variable_list grad_inputs(2);
    grad_inputs[0] = splittedTsrs[i];

    DP_LOG(DEBUG, "Remainder tensor after sending grads out. %s, %s",
        splittedTsrs[i].toString().c_str(), tsrSizeToStr(splittedTsrs[i]).c_str());
    
    return grad_inputs;
    // return { splittedTsrs[i] };
  }
  else if (xfer->type == TsrXfer::Send) {
    std::vector<int64_t> inputSizes = x.sizes().vec();
    std::vector<torch::Tensor> tsrList;
    size_t i;
    for (i = 0; i < xfer->xferTagAndRankBack.size(); ++i) {
      inputSizes[xfer->splitCatDim] = xfer->splitSizes[i];
      torch::TensorOptions topts(rtctx->c10dev);
      torch::Tensor tsr = torch::empty(inputSizes, topts);
      tsrList.push_back(tsr);
    }

    xfer->commHandler->comm_start();
    for (i = 0; i < xfer->xferTagAndRankBack.size(); ++i) {
      Tag tag = xfer->xferTagAndRankBack[i].first;
      Rank src = xfer->xferTagAndRankBack[i].second;
      auto &tsr = tsrList[i];
      DP_LOG(DEBUG, "Receiving tag:%d from R:%d with tensor: %s", tag, src,
          tsr.toString().c_str());
      xfer->commHandler->recv(tsr, tag, src, /*async*/ true);
    }
    xfer->commHandler->comm_end();
    xfer->commHandler->sync();

    tsrList.push_back(x);
    // return { torch::cat(tsrList, xfer->splitCatDim) };

    variable_list grad_inputs(2);
    grad_inputs[0] = torch::cat(tsrList, xfer->splitCatDim);
    return grad_inputs;
  }
  else {
    DP_LOG(ERROR, "xfer type is %d, which is not supported.", xfer->type);
    return grad_output;
  }
}


/**
 * Constructs RunnableModule
 */
RunnableModule::RunnableModule(RuntimeContext* rtctx,
                               json spec,
                               CommunicationHandler* commHandler,
                               c10::Device device)
  : rtctx(rtctx)
  , rank(spec["rank"].get<int>())
  , globalBatchSize(spec["globalBatchSize"].get<int>())
  , moduleList()
  , layersInJson(spec["layers"])
  , initialBatchSize(layersInJson[0]["config"][0])
  , commHandler(commHandler)
  , device(device)
  // , leavesForBackward()
  // , fpCtx(layersInJson)
  , layers()
  , layerQ()
  , fpInput()
  , fpTargets()
  , fpOutput()
  , fpLoss()
  , detachTimer("detachTimer")
{
  DP_LOG(DEBUG, "Constructing runnable module.. rank:%d", rank);
  DP_LOG(DEBUG, "             initialBatchSize:%d", initialBatchSize);
  DP_LOG(DEBUG, "             layersInJson's size:%lu (from spec)", spec["layers"].size());
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
    if (name == "concat") {
      specialModule = SpecialModuleTypes::CONCAT;
    }
    torch::jit::Module module = torch::jit::load(std::string(rtctx->homedir) +
        "/DeepPoolRuntime/" + moduleLoc);
    DP_LOG(DEBUG, " layer's module is loaded.");
    if (name == "concat") {
      DP_LOG(DEBUG, " layer is concat.");
    } else {
      DP_LOG(DEBUG, " layer is not concat.");
    }

    module.to(device);
    module.train();
    DP_LOG(DEBUG, " layer's module is moved to device and set for train mode.");

    int layerLocalBatch = ldsc["config"][0].get<int>();
    bool layerIsActive = layerLocalBatch > 0;
    if (!layerIsActive) {
      hasInactiveLayer = true;
    }
    bool detachInput = true;
    if (name == "ReLU2d" || name == "ReLU1d") {
      detachInput = false;
      DP_LOG(DEBUG, "ReLU. detachInput: %d", detachInput);
    }
    
    std::vector<Layer*> prevLayers;
    for (auto& plidjson : ldsc["prevLayers"]) {
      int plid = plidjson.get<int>();
      prevLayers.push_back(&layers[plid]);
    }
    bool detachOutput = ldsc["nextLayers"].size() > 1;
    bool syncTwice = ldsc["gpuAssignment"].size() >= rtctx->min_layer_sync;
    if (!syncTwice) {
      DP_LOG(NOTICE, " %d-th layer's name: %s, syncTwice: %d", id, name.c_str(),
          syncTwice);
    }
    layers.emplace_back(module, specialModule, id, layerIsActive, detachInput,
                        detachOutput, prevLayers, syncTwice);

    // EmptyTensorSizes.
    layers.back().emptyInSizes.push_back(0);
    for (int size : ldsc["inputDim"]) {
      layers.back().emptyInSizes.push_back(size);
    }
    layers.back().emptyOutSizes.push_back(0);
    for (int size : ldsc["outputDim"]) {
      layers.back().emptyOutSizes.push_back(size);
    }
    
    // Communications.
    if (layerIsActive && ldsc.contains("tensorTx")) {
      std::map<int, std::vector<json> > sendListDict;
      for (auto& item : ldsc["tensorTx"]) {
        int nextLayerId = item["prop"]["nextLayerId"].get<int>();
        if (sendListDict.find(nextLayerId) == sendListDict.end()) {
          sendListDict[nextLayerId] = std::vector<json>();
        }
        sendListDict[nextLayerId].push_back(item);
      }
      for (const auto& kv : sendListDict) {
        const int nextLayerId = kv.first;
        const std::vector<json>& sendList = kv.second;

        TsrXfer xfer(commHandler);
        xfer.type = TsrXfer::Send;
        xfer.splitCatDim = 0; // Sample dimension.
        xfer.prevLayerId = id;
        xfer.nextLayerId = nextLayerId;
        xfer.recevingLayerForSend = &layers.back();
        int xferSampleSum = 0;
        for (const json& item : sendList) {
          int xferSamples = item["prop"]["xferSamples"].get<int>();
          xfer.splitSizes.push_back(xferSamples);
          xferSampleSum += xferSamples;

          auto xferName = item["name"].get<std::string>();
          Tag tag = commHandler->getTag(xferName);
          Tag tagB = commHandler->getTag(xferName + "_back");
          Rank dest = item["dest"].get<Rank>();
          xfer.xferTagAndRank.push_back(std::make_pair(tag, dest));
          xfer.xferTagAndRankBack.push_back(std::make_pair(tagB, dest));
        }

        int remainder;
        if (xfer.splitCatDim == 0) {
          DP_LOG(DEBUG, "total samples for layer: %d", ldsc["config"][0].get<int>());
          remainder = ldsc["config"][0].get<int>() - xferSampleSum;
        } else { // Other than sample dimension, use outputDim as its dimension is ordered correctly.
          remainder = ldsc["outputDim"][xfer.splitCatDim - 1].get<int>() - xferSampleSum;
        }
        
        DP_LOG(DEBUG, "remainder: %d, sum: %d", remainder, xferSampleSum);
        xfer.splitSizes.push_back(remainder);
        layers.back().xferOuts.push_back(std::move(xfer));
        DP_LOG(DEBUG, "xferOut registered. len(layer->xferOuts): %lu",
            layers.back().xferOuts.size());
      }
    }

    if (ldsc.contains("tensorRxJit")) {
      std::map<int, std::vector<json> > recvListDict;
      for (auto& item : ldsc["tensorRxJit"]) {
        int nextLayerId = item["prop"]["nextLayerId"].get<int>();
        if (recvListDict.find(nextLayerId) == recvListDict.end()) {
          recvListDict[nextLayerId] = std::vector<json>();
        }
        recvListDict[nextLayerId].push_back(item);
      }

      for (const auto& kv : recvListDict) {
        const int nextLayerId = kv.first;
        const std::vector<json>& recvList = kv.second;

        TsrXfer xfer(commHandler);
        xfer.type = TsrXfer::Recv;
        xfer.splitCatDim = 0;
        xfer.prevLayerId = id;
        xfer.nextLayerId = nextLayerId;
        xfer.recevingLayerForSend = &layers.back();
        int xferSampleSum = 0;
        for (const json& item : recvList) {
          int xferSamples = item["prop"]["xferSamples"].get<int>();
          xfer.splitSizes.push_back(xferSamples);
          xferSampleSum += xferSamples;

          auto xferName = item["name"].get<std::string>();
          Tag tag = commHandler->getTag(xferName);
          Tag tagB = commHandler->getTag(xferName + "_back");
          Rank src = item["src"].get<Rank>();
          xfer.xferTagAndRank.push_back(std::make_pair(tag, src));
          xfer.xferTagAndRankBack.push_back(std::make_pair(tagB, src));
        }

        int remainder;
        if (xfer.splitCatDim == 0) {
          remainder = layersInJson[nextLayerId]["config"][0].get<int>() - xferSampleSum;
          // remainder = ldsc["config"][0].get<int>() - xferSampleSum;
        } else { // Other than sample dimension, use inputDim as its dimension is ordered correctly.
          remainder = ldsc["inputDim"][xfer.splitCatDim - 1].get<int>() - xferSampleSum;
        }

        DP_LOG(DEBUG, "remainder: %d, sum: %d", remainder, xferSampleSum);
        xfer.splitSizes.push_back(remainder);
        layers.back().xferIns.push_back(std::move(xfer));
        DP_LOG(DEBUG, "xferIn registered. len(layer->xferIns): %lu", layers.back().xferIns.size());
      }
    }

    if (rtctx->profile) {
      // TODO: if it's not that accurate, maybe add layer id?
      layers.back().moduleName = name + ldsc["params"].dump() +
          "[" + std::to_string(layerLocalBatch) + "]" + ldsc["inputDim"].dump();
      DP_LOG(DEBUG, "moduleName: %s", layers.back().moduleName.c_str());
    }
    // if (ldsc.contains("gpuTime")) {
    layers.back().fwUsec = ldsc["gpuTime"][0].get<int>();
    layers.back().bwUsec = ldsc["gpuTime"][1].get<int>();
    // }    
    // DP_LOG(DEBUG, " id: %d  fwUsec: %d, bwUsec: %d", id, layers.back().fwUsec, layers.back().bwUsec);

    moduleList.push_back(module);
    DP_LOG(DEBUG, " layer's module is pushed back.");
    DP_LOG(DEBUG, " id: %d and moduleListsize: %d", id, (int)moduleList.size());
    assert(id + 1 == (int)moduleList.size());
  }

  for (auto& layer : layers) {
    DP_LOG(DEBUG, "lid: %d, xferOuts: %lu, xferIns: %lu", layer.id,
        layer.xferOuts.size(), layer.xferIns.size());
  }

  for (auto& layer : layers) {
    DP_LOG(DEBUG, "lid: %d, fwUsec: %" PRId64 ", bwUsec: %" PRId64 "",
        layer.id, layer.fwUsec, layer.bwUsec);
  }

  /* set up fake data pipelines for input + target */
  std::vector<int64_t> inputSizes;
  inputSizes.push_back(initialBatchSize);
  for (int size : layersInJson[0]["inputDim"]) inputSizes.push_back(size);
  auto inputFn = [=] { return torch::randn(inputSizes); };
  input_pipeline = TensorGeneratorPipeline(inputFn);
  int targetCount = layersInJson.back()["config"][0];

  if (layersInJson.back()["outputDim"].size() <= 2) {
    auto targetOpts = torch::TensorOptions().dtype(torch::kInt64);
    auto targetFn = [=] { return torch::randint(/*low=*/0, /*high=*/1000, {targetCount}, targetOpts); };
    target_pipeline = TensorGeneratorPipeline(targetFn);
  } else {
    std::vector<int64_t> outputSizes;
    outputSizes.push_back(targetCount);
    for (int size : layersInJson.back()["outputDim"]) outputSizes.push_back(size);
    outputSizes.erase(outputSizes.begin() + 1);
    auto targetFn = [=] { return torch::randn(outputSizes); };
    target_pipeline = TensorGeneratorPipeline(targetFn);
  }
}

/**
 * Dumps the entire model parameters into the given vector.
 */
void
RunnableModule::getParameters(std::vector<torch::Tensor>* parameters)
{
  for (const auto& module : moduleList) {
    for (const auto& params : module.parameters()) {
      parameters->push_back(params);
    }
  }
}

/**
 * Dumps the entire model parameters into the given vector.
 */
void
RunnableModule::getActiveParameters(std::vector<torch::Tensor>* parameters)
{
  for (auto& layer : layers) {
    if (layer.active) {
      for (const auto& params : layer.module.parameters()) {
        parameters->push_back(params);
      }
    }
  }
}

/**
 * Initiate an iteration.
 */
void
RunnableModule::iterInit()
{
  layerQ.clear();
  layerQ.push_back(&layers[0]);
  fpInput = input_pipeline.GetNext();
  fpTargets = target_pipeline.GetNext();
  fpOutput.reset();
  fpLoss.reset();
  // reduceBuckets.clear();
  // reduceBuckets.emplace_back();
}

void
RunnableModule::resetForNewIter()
{
  for (auto& layer : layers) {
    layer.status = LayerStatus::PENDING_FP;
  }
}

/**
 * Execute a forward pass of this model.
 * 
 * \return Returns true if forward pass is completed.
 */
JobStatus
RunnableModule::forwardAStep(bool captureLayer)
{
  DP_LOG(DEBUG, "layerQ size: %d", (int)layerQ.size());
  Layer* layer = layerQ.front();
  layerQ.pop_front();

  // TODO: potentially we can make track if the cuda kernel is finished
  // or probably finished.
  // bool skipSinceNotReady = false;
  if (layer->status == LayerStatus::PENDING_BP) {
    DP_LOG(DEBUG, "%d-th layer is processed again.", layer->id);
    return IN_PROGRESS;
  }
  DP_LOG(DEBUG, "lid:%d.", layer->id);
  for (auto& prevLayer : layer->prevLayers) {
    if (prevLayer->status == LayerStatus::PENDING_FP) {
      DP_LOG(DEBUG, "Layer %d is skipped for now, must do %d first.",
          layer->id, prevLayer->id);
      return IN_PROGRESS;
    }
  }
  
  if (layer->active) {
    DP_LOG(DEBUG, "Layer %d is active.", layer->id);

    std::vector<torch::Tensor> inputVec;
    if (layer->prevLayers.size() == 0) {
      DP_LOG(DEBUG, "Adding to inputVec: %s.", tsrSizeToStr(fpInput).c_str());
      inputVec.push_back(fpInput);
    } else if (layer->prevLayers.size() >= 1) {
      std::map<int, torch::Tensor> inputsByPid;
      for (auto& prevLayer : layer->prevLayers) {
        torch::Tensor prevOut;
        if (prevLayer->outputsAfterXfer.count(layer->id) > 0) {
          prevOut = prevLayer->outputsAfterXfer[layer->id];
        } else {
          prevOut = prevLayer->output;
        }
        if (!prevOut.defined()) {
          DIE("prevOut is not defined.");
        }
        if (layer->detachInput) {
          // detachTimer.start();
          layer->detachedInputs[prevLayer->id] = prevOut.detach();
          layer->detachedInputs[prevLayer->id].requires_grad_();
          prevOut = layer->detachedInputs[prevLayer->id];
          // detachTimer.stop();
          DP_LOG(DEBUG, "Detached input tensor");
        }
        inputsByPid[prevLayer->id] = prevOut;
      }
      
      for (auto& plidInputPair : inputsByPid) {
        DP_LOG(DEBUG, "Adding to inputVec: %s.", tsrSizeToStr(plidInputPair.second).c_str());
        inputVec.push_back(plidInputPair.second);
      }
    } else {
      DIE("%d-th layer negative number of previous layers??", layer->id);
    }

    torch::Tensor output;
    if (layer->specialModule == SpecialModuleTypes::CONCAT) {
      // temporary hack to solve the problem of passing list of tensors as input.
      output = torch::cat(inputVec, 1);
    } else {
      std::vector<torch::jit::IValue> ivalVec;
      ivalVec.push_back(inputVec[0]);

      if (captureLayer) { // Used layer time profiling.
        c10::cuda::device_synchronize();
        layer->moduleFwGraph.capture_begin();
      }

      output = layer->module.forward(ivalVec).toTensor();

      if (captureLayer) { // Used layer time profiling.
        layer->moduleFwGraph.capture_end();
        c10::cuda::device_synchronize();
        CpuTimer timer("fwTimer");
        timer.start();
        int repeat = 200;
        for (int i = 0; i < repeat; ++i) {
          layer->moduleFwGraph.replay();
        }
        c10::cuda::device_synchronize();
        timer.stop();
        layer->avgLayerTime = static_cast<double>(timer.avgMicros())
                              / 1000.0 / repeat;
        layer->fwUsec = timer.avgMicros() / repeat;
      }
      DP_LOG(DEBUG, "module.forward called.");
    }

    if (rtctx->profile) {
      layer->fpTimer->record();
    }

    if (layer->detachOutput) {
      layer->outputBeforeDetach = output;
      output = output.detach();
      output.requires_grad_();
    }

    // Send samples after running this layer.
    DP_LOG(DEBUG, "len(layer->xferOuts): %lu", layer->xferOuts.size());
    layer->outputsAfterXfer.clear();
    for (TsrXfer& xfer : layer->xferOuts) {
      torch::Tensor remainingOutput = TsrXferFunc::apply(output, &xfer);
      layer->outputsAfterXfer[xfer.nextLayerId] = remainingOutput;
      DP_LOG(DEBUG, "Split & sent samples.");
    }
    layer->output = output;

    // auto h = layer->output.register_hook([layer](torch::Tensor grad){
    //   DP_LOG(DEBUG, "lid:%d grad: %s", layer->id, tsrToStr(grad).c_str());
    // });
    idleCtxPtr->processLayerTime(layer->fwUsec, true);
  } else { // This rank doesn't participate for this layer.
    DP_LOG(DEBUG, "Layer %d is not active.", layer->id);
    idleCtxPtr->processLayerTime(layer->fwUsec, false);
  }

  // Recv parts of output processed by another GPU.
  for (TsrXfer& xfer : layer->xferIns) {
    //TODO: assert that next layer is active.
    torch::Tensor localOut;
    if (layer->active) { // Don't use outputsAfterXfer if not active.
      if (layer->outputsAfterXfer.count(xfer.nextLayerId) > 0) {
        localOut = layer->outputsAfterXfer[xfer.nextLayerId];
      } else {
        localOut = layer->output;
      }
    }

    if (!localOut.defined()) {
      assert(!layer->active);
      DP_LOG(DEBUG, "localOut is not defined. Must be inactive? Using an empty tensor.");
      torch::TensorOptions topts(rtctx->c10dev);
      topts = topts.requires_grad(true);
      localOut = torch::empty(layer->emptyOutSizes, topts);
      DP_LOG(DEBUG, "Empty localOut tensor: %s", localOut.toString().c_str());
    }
    torch::Tensor remainingOutput = TsrXferFunc::apply(localOut, &xfer);
    layer->outputsAfterXfer[xfer.nextLayerId] = remainingOutput;
    DP_LOG(DEBUG, "Received (nextLayer: %d) & concatenated samples. %s",
        xfer.nextLayerId, tsrSizeToStr(remainingOutput).c_str());
  }
  
  layer->status = LayerStatus::PENDING_BP;
  DP_LOG(DEBUG, " ** Layer %d is processed.", layer->id);
  
  for (auto& nextLayerPtr : layer->nextLayers) {
    if (nextLayerPtr->status == LayerStatus::PENDING_FP) {
      layerQ.push_back(nextLayerPtr);
      DP_LOG(DEBUG, "nextLayer %d is queued for processing.", nextLayerPtr->id);
    } else {
      DP_LOG(DEBUG, "nextLayer %d is already processed.", nextLayerPtr->id);
    }
  }

  // Forward pass is completed.
  if (layerQ.empty()) {
    DP_LOG(DEBUG, "no more layers to process.");
    if (layer->output.defined()) {
      fpOutput = layer->output;
    } else {
      fpOutput.reset();
    }
    return COMPLETED;
  }
  return IN_PROGRESS;
}

/**
 * Compute the loss from forward pass.
 */
void
RunnableModule::loss()
{
  if (fpOutput.defined()) {    
    fpLoss = torch::nll_loss(fpOutput, fpTargets);
    // DP_LOG(DEBUG, "fpLoss: %s", tsrToStr(fpLoss).c_str());
    fpLoss.backward();
    DP_LOG(DEBUG, "fpLoss.backward() done. ");
    // idleCtxPtr->processLayerTime(1000, true);
  } else {
    if (idleCtxPtr->jobType == IdleTimeCtx::FG) { // Don't deduct time for BG.
      idleCtxPtr->processLayerTime(3000, false);  // For WRN.
      // idleCtxPtr->processLayerTime(2000, false);  // For VGG16.
    }
  }
}

/**
 * Execute a backward pass of this model.
 * 
 * \return Returns true if backward pass is completed.
 */
JobStatus
RunnableModule::backwardAStep(bool captureLayer)
{
  Layer* layer = layerQ.front();
  layerQ.pop_front();

  // TODO: potentially we can make track if the cuda kernel is finished
  // or probably finished.
  if (layer->status == LayerStatus::PENDING_FP) {
    DP_LOG(DEBUG, "%d-th layer is processed again.", layer->id);
    return IN_PROGRESS;
  }
  DP_LOG(DEBUG, "lid:%d.", layer->id);
  for (auto& nextLayer : layer->nextLayers) {
    if (nextLayer->status == LayerStatus::PENDING_BP) {
      DP_LOG(DEBUG, "Layer %d is skipped for now, must do %d first.",
          layer->id, nextLayer->id);
      return IN_PROGRESS;
    }
  }

  bool mustRunBackward = layer->active;
  if (!layer->active && layer->outputsAfterXfer.size()) {
    mustRunBackward = true;
    DP_LOG(DEBUG, "Layer %d is inactive, but backward is called for sending "
        "out gradients.", layer->id);
#if VERBOSE
    bool someNextLayerIsActive = false;
    for (const auto nextLayer : layer->nextLayers) {
      if (nextLayer->active)
        someNextLayerIsActive = true;
    }
    assert(someNextLayerIsActive);
#endif
  }

  if (mustRunBackward) {
    if (layer->nextLayers.size() == 0) {
      DP_LOG(DEBUG, "No nextLayers.");
    } else if (layer->nextLayers.size() >= 1) {
      // for (auto nextLayerPtr : layer->nextLayers) {
      for (size_t nli = 0; nli < layer->nextLayers.size(); nli++) {
        auto nextLayerPtr = layer->nextLayers[nli];
        if (nextLayerPtr->detachInput) {
          bool retainGraph = nli < layer->nextLayers.size() - 1;
          torch::Tensor grad;
          if (nextLayerPtr->detachedInputs[layer->id].defined()) {
            grad = nextLayerPtr->detachedInputs[layer->id].grad();
          } else {
            DP_LOG(DEBUG, "nextLayerPtr->detachInput is not defined. Using empty tensor.");
            torch::TensorOptions topts(rtctx->c10dev);
            grad = torch::empty(layer->emptyOutSizes, topts);
          }
          DP_LOG(DEBUG, "nextLayerPtr(%d)->detachedInputs[%d]: %s, grad: %s",
              nextLayerPtr->id, layer->id,
              nextLayerPtr->detachedInputs[layer->id].toString().c_str(),
              tsrSizeToStr(grad).c_str());
          
          if (captureLayer) { // Used layer time profiling.
            c10::cuda::device_synchronize();
            layer->moduleBwGraph.capture_begin();
          }
          
          if (layer->outputsAfterXfer.count(nextLayerPtr->id)) {
            DP_LOG(DEBUG, "Backward on outputsAfterXfer:%s gradIn:%s", 
                tsrSizeToStr(layer->outputsAfterXfer[nextLayerPtr->id]).c_str(),
                tsrSizeToStr(grad).c_str());
            layer->outputsAfterXfer[nextLayerPtr->id].backward(grad, retainGraph);
          } else if (layer->active) {
            DP_LOG(DEBUG, "Backward on output:%s gradIn:%s", 
                tsrSizeToStr(layer->output).c_str(), tsrSizeToStr(grad).c_str());
            layer->output.backward(grad, retainGraph);
          } else {
            DP_LOG(DEBUG, "Backward is not called since inactive layer & "
                "no xferIn for layer %d", nextLayerPtr->id);
          }

          if (captureLayer) { // Used layer time profiling.
            layer->moduleBwGraph.capture_end();
            c10::cuda::device_synchronize();

            CpuTimer timer("bwTimer");
            timer.start();
            int repeat = 200;
            for (int i = 0; i < repeat; ++i) {
              layer->moduleBwGraph.replay();
            }
            c10::cuda::device_synchronize();
            timer.stop();
            layer->avgLayerTime += static_cast<double>(timer.avgMicros())
                                    / 1000.0 / repeat;
            layer->bwUsec = timer.avgMicros() / repeat;
          }

        } else {
          DP_LOG(DEBUG, "  nextLayerPtr(%d)->detachInput is false!", nextLayerPtr->id);
        }
      }

      if (layer->active && layer->detachOutput) {
        DP_LOG(DEBUG, "  output was detached previously. Invoking backward on outputBeforeDetach.");
        layer->outputBeforeDetach.backward(layer->output.grad());
      }
    }
    DP_LOG(DEBUG, "Layer %d is active.", layer->id);
    idleCtxPtr->processLayerTime(layer->bwUsec, true);
  } else { // This rank doesn't participate for this layer.
    DP_LOG(DEBUG, "Layer %d is not active.", layer->id);
    idleCtxPtr->processLayerTime(layer->bwUsec, false);
  }
  
  if (rtctx->profile) {
    layer->bpTimer->record();
  }
  layer->status = LayerStatus::PENDING_FP;

  if (layer->syncTwice && layer->active) {
    for (const auto& param : layer->module.parameters()) {
      auto grad = param.mutable_grad();
      bytes_inflight += grad.nbytes();
      pending_grads.push_back(grad);
      backwards_did_sync = true;
    }

    if (rtctx->sync_bucket_size > 0 && bytes_inflight >= rtctx->sync_bucket_size) {
      commHandler->comm_start(rtctx->grad_sync_stream);
      for (auto &p : pending_grads)
        commHandler->all_reduce(p, c10d::ReduceOp::SUM, true);
      commHandler->comm_end();
      bytes_inflight = 0;
      pending_grads.clear();

      // if (param.mutable_grad().numel() > ReduceBucket::elemLimit) {
      //   commHandler->all_reduce(param.mutable_grad(), c10d::ReduceOp::SUM, true);
      // } else {
      //   if (reduceBuckets.back().holdGrad(param.mutable_grad())) {
      //     commHandler->all_reduce(reduceBuckets.back().buffer, c10d::ReduceOp::SUM, true);
      //     reduceBuckets.emplace_back();
      //   }
      // }
    }

  }
  
  for (auto& prevLayerPtr : layer->prevLayers) {
    if (prevLayerPtr->status == LayerStatus::PENDING_BP) {
      layerQ.push_back(prevLayerPtr);
    } else {
      DP_LOG(DEBUG, "prevLayer %d is already processed.", prevLayerPtr->id);
    }
  }

  // Forward pass is completed.
  if (layerQ.empty()) {
    DP_LOG(DEBUG, "no more layers to process.");
    return COMPLETED;
  }
  return IN_PROGRESS;
}

void
RunnableModule::gradientSync() {
  if (bytes_inflight) {
      commHandler->comm_start(rtctx->grad_sync_stream);
      for (auto &p : pending_grads)
        commHandler->all_reduce(p, c10d::ReduceOp::SUM, true);
      commHandler->comm_end();
      bytes_inflight = 0;
      pending_grads.clear();
  }
  if (backwards_did_sync) {
    commHandler->sync(rtctx->grad_sync_stream);
    backwards_did_sync = false;
  }

  // if (reduceBuckets.back().grads.size() > 0) {
  //   reduceBuckets.back().wrapUp();
  //   commHandler->all_reduce(reduceBuckets.back().buffer, c10d::ReduceOp::SUM, true);
  // }
  // commHandler->sync();
  // for (ReduceBucket& bucket : reduceBuckets) {
  //   bucket.splitAndUpdateGrads();
  // }

  // DP_LOG(DEBUG, "reduceBuckets.size(): %d", reduceBuckets.size());

  // // First sync within a host & with fast networking.
  // for (auto& layer : layers) {
  //   if (layer.syncTwice) {
  //     for (const auto& param : layer.module.parameters()) {
  //       commHandler->all_reduce(param.mutable_grad(), c10d::ReduceOp::SUM, false);
  //     }
  //   }
  // }
  
  // // Second sync to outside of box. Let's mimic that overhead by performing
  // // intra-host sync again. (200Gbps * 8 GPUs = 1.6Tbps ~ NVSwitch bandwidth.)
  // // Assuming each GPU has ConnectX-6, and a box has 8 GPUs.
  // int64_t totalGradSize = 0;
  // for (auto& layer : layers) {
  //   for (const auto& param : layer.module.parameters()) {
  //     totalGradSize += param.numel(); 
  //   }
  // }
  // torch::TensorOptions topts(rtctx->c10dev);
  // torch::Tensor grad = torch::empty({1, totalGradSize}, topts);
  // // DP_LOG(NOTICE, "2nd tensor xfer: %s", tsrSizeToStr(grad).c_str());
  // commHandler->all_reduce(grad, c10d::ReduceOp::SUM, false);
  // // Need another internal sync? Included in the 1st sync?
}

/**
 * Initialize timers for profiling each layer.
 */
void
RunnableModule::initProfileTimers(CudaTimer* ct_load, CudaTimer* ct_loss) {
  if (rtctx->profile) {
    DP_LOG(NOTICE, "initProfileTimers invoked");
    for (auto& layer : layers) {
      layer.fpTimer = std::make_unique<CudaTimer>(ct_load);
      layer.bpTimer = std::make_unique<CudaTimer>(ct_loss);
    }
  }
}

/**
 * Reset timers for profiling each layer. Happens every iteration.
 */
void
RunnableModule::resetProfileTimers() {
  if (rtctx->profile) {
    for (auto& layer : layers) {
      layer.fpTimer->saveAndReset();
      layer.bpTimer->saveAndReset();
    }
  }
}

void
RunnableModule::printLayerInGraphTimes() {
  if (!rtctx->profile) {
    return;
  }

  double sum = 0;
  for (auto& layer : layers) {
    printf(" %110s  %6.3f  %8" PRId64 "  %8" PRId64 "\n", layer.moduleName.c_str(),
        layer.avgLayerTime, layer.fwUsec, layer.bwUsec);
    sum += layer.avgLayerTime;
  }
  printf("%100s  %.3f\n", "SUM(avg)", sum);
}

/**
 * Reset timers for profiling each layer. Happens every iteration.
 */
void
RunnableModule::printProfileTimers(int warmupIters) {
  if (!rtctx->profile) {
    return;
  }

  auto getName2LayerTime = [&] (float percentile) {
    std::vector<std::pair<float, const char*> > fpTimes;
    std::vector<std::pair<float, const char*> > bpTimes;
    for (auto& layer : layers) {
      if (!layer.detachInput) {
        continue;
      }
      fpTimes.push_back(std::make_pair<float, const char*>(
          layer.fpTimer->getPercentile(percentile, warmupIters),
          layer.moduleName.c_str()));
      bpTimes.push_back(std::make_pair<float, const char*>(
          layer.bpTimer->getPercentile(percentile, warmupIters),
          layer.moduleName.c_str()));
    }
    std::sort(fpTimes.begin(), fpTimes.end());
    std::sort(bpTimes.begin(), bpTimes.end());

    float lastTime = 0;
    std::unordered_map<const char*, float> nameToTime;
    for (auto& timeName : fpTimes) {
      float layerTime = timeName.first - lastTime;
      nameToTime[timeName.second] = layerTime;
      lastTime = timeName.first;
    }
    lastTime = 0;
    for (auto& timeName : bpTimes) {
      float layerTime = timeName.first - lastTime;
      nameToTime[timeName.second] += layerTime;
      lastTime = timeName.first;
    }
    return nameToTime;
  };

  // Get average.
  std::vector<std::pair<float, const char*> > fpTimes;
  std::vector<std::pair<float, const char*> > bpTimes;
  for (auto& layer : layers) {
    if (!layer.detachInput) {
      continue;
    }
    fpTimes.push_back(std::make_pair<float, const char*>(
        layer.fpTimer->getAvg(warmupIters), layer.moduleName.c_str()));
    bpTimes.push_back(std::make_pair<float, const char*>(
        layer.bpTimer->getAvg(warmupIters), layer.moduleName.c_str()));
  }
  std::sort(fpTimes.begin(), fpTimes.end());
  std::sort(bpTimes.begin(), bpTimes.end());
  // printf("## Forward time\n");
  float lastTime = 0;
  std::unordered_map<const char*, float> nameToTime;
  for (auto& timeName : fpTimes) {
    float layerTime = timeName.first - lastTime;
    // printf("%110s  %.3f\n", timeName.second, layerTime);
    nameToTime[timeName.second] = layerTime;
    lastTime = timeName.first;
  }
  // printf("## Backward time\n");
  lastTime = 0;
  for (auto& timeName : bpTimes) {
    float layerTime = timeName.first - lastTime;
    // printf("%110s  %.3f\n", timeName.second, layerTime);
    nameToTime[timeName.second] += layerTime;
    lastTime = timeName.first;
  }

  
  std::unordered_map<const char*, float> p50Times = getName2LayerTime(50);
  std::unordered_map<const char*, float> p90Times = getName2LayerTime(90);
  std::unordered_map<const char*, float> p99Times = getName2LayerTime(99);
  
  // printf("## Sum\n");
  printf("%110s  avg(ms)    p50     p90     p99\n", "#config");
  float sum = 0;
  for (auto& timeName : fpTimes) {
    const char* name = timeName.second;
    float avgT = nameToTime[name];
    sum += avgT;
    printf("%110s  %6.3f  %6.3f  %6.3f  %6.3f\n", name, avgT, p50Times[name],
           p90Times[name], p99Times[name]);
  }
  printf("%100s  %.3f\n", "SUM(avg)", sum);
}