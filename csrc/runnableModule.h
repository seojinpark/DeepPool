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

#ifndef RUNNABLE_MODULE_H
#define RUNNABLE_MODULE_H

#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <deque>
#include <vector>

#include "CUDAGraph.h"
#include "GradSync.h"
#include "GraphPieces.h"
#include "Manager.h"
#include "communication.h"
#include "json.hpp"
#include "logger.h"
#include "runtime.h"
#include "tracer.h"
#include "utils.h"

using json = nlohmann::json;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

/**
 * Forward declarations. Do not include headers unless necessary.
 */
class CommunicationHandler;
class RunnableModule;

typedef int Tag;

struct Xfer {
  std::pair<size_t, size_t> src; /* rank and offset */
  std::pair<size_t, size_t> dst; /* rank and offset */
  size_t nr_samples;
  size_t src_lid;
  bool skip_backward;
  Tag tag;
};

/**
 * Flipping status flag. This variable tracks the execution of the layer.
 * Specifically, it is used to (1) prevent duplicate execution and (2) ensure
 * the previous layer is executed so that its output tensor is valid.
 */
enum class LayerStatus {
  PENDING_FP = 0,  // pending forward pass (last done job was backward).
  PENDING_BP       // pending backward pass (last done job was forward).
};

enum class SpecialModuleTypes { NOTSPECIAL = 0, CONCAT, DISTRIBUTE };

/**
 * Description / context of a layer for training.
 */
struct Layer {
  Layer(torch::jit::Module module, SpecialModuleTypes specialModule, int id,
        bool active, bool doLocalGradSync, std::string name)
      : module(module),
        specialModule(specialModule),
        id(id),
        active(active),
        doLocalGradSync(doLocalGradSync) {
    std::stringstream ss;
    ss << "LAYER_" << id << "_" << name;
    layername = ss.str();
  }

  torch::Tensor DoForward(RunnableModule *model, bool captureLayer);
  void DoBackward(RunnableModule *model, bool captureLayer);
  void saveModule(std::string savePath){module.save(savePath);};
  void loadModule(std::string loadPath){module = torch::jit::load(loadPath);};

  /* stores inputs on forward pass, gradients on backward pass */
  std::map<size_t, torch::Tensor> tensors_in;

  /* stores result of forward pass for backward */
  torch::Tensor output;
  /* stores outputs for distribute nodes */
  std::map<size_t, torch::Tensor> dist_outputs;

  std::vector<Xfer> xfers;
  size_t nr_nccl_recv{0}, nr_nccl_send{0};
  std::vector<Xfer> xfers_local;

  std::set<size_t> tx_lids;
  std::set<size_t> rx_lids;

  /* defined if gradients should be synced after this layer's backprop */
  c10::optional<GradientSyncGroup> grad_sync_group;

  std::string layername;

  torch::jit::Module module;
  int64_t fwUsec{0};
  int64_t bwUsec{0};
  const SpecialModuleTypes
      specialModule;  // 0: not special, use module. 1: concat.
  const int id;
  const bool active;  // Inactive means no samples assigned for this runtime.
  const bool doLocalGradSync;  // Perform gradient all-reduce within a host.
                               // (all layers do all-reduce over NIC)
  size_t commGroupKey;
  std::vector<std::shared_ptr<Layer>> prevLayers;  // sorted by id
  std::vector<std::shared_ptr<Layer>> nextLayers;
  LayerStatus status{LayerStatus::PENDING_FP};
  size_t nr_current_depedencies{0};
  long layerLocalBatch;
  std::vector<std::vector<int64_t>> emptyOutSizes;
  std::vector<std::vector<int64_t>> emptyInSizes;
  std::vector<c10::ScalarType> inOpts;
  std::vector<c10::ScalarType> outOpts;
  std::string moduleName;  // Used to output profiled runtimes.

  size_t localRank;

  Layer(const Layer &) = delete;
  Layer &operator=(const Layer &) = delete;
  Layer(Layer &&) = delete;
  Layer &operator=(Layer &&) = delete;
};

class ScopedGraphRecorder;

enum JobStatus { IN_PROGRESS = 0, COMPLETED, YIELD };

enum class LossFunctions {
  NLLLoss = 0,
  CrossEntropyLoss,
};

enum class JobState {
  INIT = 0,
  FORWARD,
  LOSS,
  BACKWARD,
  SYNC,
  STEP,
  FINISH,
  NUM_JOB_STATES  // must be the last element in the enum
};

/**
 * A module that holds parameters (or submodules) and
 * provides functionalities to run training iteration.
 */
class RunnableModule {
 public:
  RunnableModule(json specInJson,
                 std::shared_ptr<CommunicationHandler> commHandler,
                 LossFunctions lf);
  ~RunnableModule() {
    if (cur_task && has_graph) {
      GpuManager::getInstance()->RemoveTask(cur_task);
    }
  }

  int AdvanceTraining(bool doGraphCapture, bool layerProfile);

  void printLayerInGraphTimes();

  void ExecuteXfers(std::shared_ptr<Layer> &layer, bool backward = false);

  void SetTrain() {
    assert(state == JobState::INIT);
    SetMode(true);
  }

  void SetEval() {
    assert(state == JobState::INIT);
    SetMode(false);
  }

  torch::Tensor getOutput() { return fpOutput; }

  void SetInputsTargets(std::vector<torch::Tensor> inputs,
                        torch::Tensor target = {});

  const auto &GetTimers() { return timers; }

  long GetGlobalBatchSize() const { return globalBatchSize; }

  double GetAvgLoss() {
    return loss_tracker_.item().toDouble() / static_cast<double>(nr_iters_);
  }

  bool isTrain() const { return isTrain_; }

  RunnableModule(const RunnableModule &) = delete;
  RunnableModule &operator=(const RunnableModule &) = delete;
  RunnableModule(RunnableModule &&) = delete;
  RunnableModule &operator=(RunnableModule &&) = delete;

 private:
  friend struct Layer;
  friend class JobContext;
  friend class ScopedGraphRecorder;

  std::shared_ptr<GpuTask> cur_task;

  CudaTimerChain timers;
  CudaTimerChain layerts_fwd, layerts_bwd;

  bool isTrain_{true};

  void SetMode(bool train);

  long globalBatchSize;
  std::vector<long> sampleIndices;

  inline void TimerRecordLayer(std::string name, bool backwards) {
    if (!rtctx->profile_layer_times_timers || has_graph || graph_recording)
      return;

    if (backwards)
      layerts_bwd.Record(name);
    else
      layerts_fwd.Record(name);
  }

  inline void TimerRecordStage(std::string name) {
    if (rtctx->profile_stage_time && !has_graph && !graph_recording)
      timers.Record(name);
  }

  JobStatus forwardAStep(bool captureLayer);
  JobStatus backwardAStep(bool captureLayer);
  void loss();
  void resetTimers();
  void SetupOptimizer();
  void saveOptimizer(std::string savePath){torch::save(*optimizer.get(), savePath);};
  void SetupOptimizer(std::string loadPath){torch::load(*optimizer.get(), loadPath);};

  ////////////////////////////////////////////
  // Internal data structure.
  ////////////////////////////////////////////
  std::shared_ptr<CommunicationHandler> commHandler;
  std::map<size_t, GradientSyncGroup> grad_sync_groups;
  // Topologically sorted list of layers.
  std::vector<std::shared_ptr<Layer>> layers;
  std::vector<std::shared_ptr<Layer>> input_layers;
  std::shared_ptr<Layer> lossLayer;
  std::unique_ptr<torch::optim::SGD> optimizer;
  ////////////////////////////////////////////
  // Context for tracking partial progress.
  ////////////////////////////////////////////
  std::deque<std::shared_ptr<Layer>> layerQ;
  torch::Tensor fpTargets;
  torch::Tensor fpOutput;
  torch::Tensor fpLoss;
  LossFunctions lossfn_;

  torch::Tensor loss_tracker_;
  size_t nr_iters_{0};

  JobState state{JobState::INIT};

  bool first_pass{true};
  bool backwards_did_sync{false};
  bool has_graph{false};
  bool graph_recording{false};
  torch::Tensor target_buf;
  std::vector<torch::Tensor> input_bufs;
  at::cuda::CUDAEvent ev;

  at::cuda::MempoolId_t graph_mempool;

  void ResetGraphs() {
    rtctx->torch_stream.synchronize();
    if (!has_graph) return;
    GpuManager::getInstance()->RemoveTask(cur_task);
    isTrain_ ? cur_task->ResetToTrain() : cur_task->ResetToEval();
    GpuManager::getInstance()->AddTask(cur_task);
    rtctx->torch_stream.synchronize();
  }
};

class ScopedGraphRecorder {
 public:
  ScopedGraphRecorder(RunnableModule *model, unsigned int flags,
                      std::string debug_name = "")
      : model_(model), flags_(flags), debug_name_(debug_name) {
    if (!model_->graph_recording) return;
    graph.reset(new DeepPool::CUDAGraph());
    c10::cuda::device_synchronize();
    graph->capture_begin(model_->graph_mempool);
  }
  ~ScopedGraphRecorder() {
    if (!graph) return;
    graph->capture_end();
    c10::cuda::device_synchronize();
    size_t nr_nodes = 0;
    CUDA_API_CALL(cudaGraphGetNodes(graph->getGRAPH(), nullptr, &nr_nodes));
    if (nr_nodes > 0) model_->cur_task->AddTask({graph, flags_, debug_name_});
  }

 private:
  RunnableModule *model_;
  unsigned int flags_;
  std::string debug_name_;
  std::shared_ptr<DeepPool::CUDAGraph> graph;
};

#endif
