
#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>

#include "json.hpp"
#include "runnableModule.h"
#include "runtime.h"
#include "utils.h"
#include <unordered_map>
#include <iostream> 
#include <sstream> 

using namespace torch::indexing;

struct Example {
  Example(torch::Tensor d, torch::Tensor target) : target(target) {
    data.push_back(d);
  }
  Example(std::vector<torch::Tensor> data, torch::Tensor target)
      : data(data), target(target) {}
  std::vector<torch::Tensor> data;
  torch::Tensor target;
};

class Dataset {
 public:
  virtual Example getNext() = 0;
  virtual size_t GetItersPerEpoch() = 0;
  virtual bool IsDone() = 0;
  virtual void Reset() = 0;
  static Dataset *fromName(std::string name, json jobParams, size_t worldSize, size_t rank,
                           long globalBatchSize,
                           std::vector<std::shared_ptr<Layer>> input_layers,
                           std::vector<long> sampleIndices,
                           size_t fake_train_iters_per_epoch);

  Example globalToPerRankExample(Example ex) {
    std::vector<torch::Tensor> data;
    torch::Tensor target;

    assert(ex.data.size() == input_layers.size());
    for (size_t i = 0; i < input_layers.size(); i++) {
      auto &iLayer = input_layers[i];
      if (!iLayer->active) {
        data.emplace_back();
        continue;
      }
      size_t sample_start = iLayer->layerLocalBatch * iLayer->localRank;
      size_t sample_end = sample_start + iLayer->layerLocalBatch;
      torch::Tensor dataSlice = ex.data[i].index({Slice(sample_start, sample_end)});
      data.push_back(dataSlice.pin_memory());
    }

    std::vector<torch::Tensor> samplesOrdered;
    for (auto &slice : slices_)
      samplesOrdered.push_back(ex.target.index({slice}));
  if (samplesOrdered.size()) {
    target = torch::cat(samplesOrdered);
    target = target.pin_memory();
  }
  
    return {data, target};
  }

 protected:
  size_t worldSize_;
  size_t rank_;
  long globalBatchSize_;
  long localBatchSize;
  Dataset(size_t worldSize, size_t rank, long globalBatchSize,
          std::vector<std::shared_ptr<Layer>> input_layers,
          std::vector<long> sampleIndices)
      : localBatchSize(globalBatchSize/worldSize),
      globalBatchSize_(globalBatchSize),
      worldSize_(worldSize),
        rank_(rank),
        input_layers(input_layers) {
    if (sampleIndices.size() == 0) return;
    long start_sample = sampleIndices[0];
    long last_sample = start_sample;

    /* find and record ranges of consecutive samples indices */
    for (size_t i = 1; i < sampleIndices.size(); i++) {
      auto sidx = sampleIndices[i];
      if (sidx != last_sample + 1) {
        slices_.emplace_back(start_sample, last_sample + 1);
        start_sample = sidx;
      }
      last_sample = sidx;
    }
    slices_.emplace_back(start_sample, last_sample + 1);
  }

 private:
  std::vector<Slice> slices_;
  std::vector<std::shared_ptr<Layer>> input_layers;
};

class TensorPipeline {
 public:
  TensorPipeline(std::vector<torch::Tensor> next) { SupplyNext(next); }

  void SupplyNext(std::vector<torch::Tensor> next) {
    /* run next HtoD transfer */
    auto origstream = c10::cuda::getCurrentCUDAStream();
    c10::cuda::setCurrentCUDAStream(rtctx->xfer_stream);
    next_up_.clear();
                // std::chrono::_V2::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    for (auto &n : next) {
      if (n.defined() && n.nbytes())
        next_up_.push_back(
            n.to(rtctx->c10dev, /*non_blocking*/ true, /*copy*/ false));
      else
        next_up_.push_back(n);
    }
  //           std::chrono::_V2::steady_clock::time_point t1 = std::chrono::steady_clock::now();

  // using msec = std::chrono::duration<double, std::micro>;
  // double load0 = std::chrono::duration_cast<msec>(t1 - t0).count();
  // DP_LOG(
  //     NOTICE,
  //     "moving from cpu to gpu async: %.2f", load0);

    c10::cuda::setCurrentCUDAStream(origstream);
  }

  std::vector<torch::Tensor> GetNext(std::vector<torch::Tensor> next) {
    /* current stream must wait for xfer before using returned tsr */
        // std::chrono::_V2::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    xfer_ev_.record(rtctx->xfer_stream);
    xfer_ev_.block(c10::cuda::getCurrentCUDAStream());

        // std::chrono::_V2::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    auto out = next_up_;
    next_up_.clear();
    if (next.size()) SupplyNext(next);

        // std::chrono::_V2::steady_clock::time_point t2 = std::chrono::steady_clock::now();

  // using msec = std::chrono::duration<double, std::micro>;
  // double load0 = std::chrono::duration_cast<msec>(t1 - t0).count();
  // double load1 = std::chrono::duration_cast<msec>(t2 - t1).count();
  // DP_LOG(
  //     NOTICE,
  //     "waiting for stream: %.2f\tsupplyNext(): %.2f", load0, load1);

    return out;
  }

 private:
  std::vector<torch::Tensor> next_up_;
  at::cuda::CUDAEvent xfer_ev_;
};

class DatasetPipelineWrapper {
 public:
  DatasetPipelineWrapper(std::shared_ptr<Dataset> dataset) : dataset_(dataset) {
    auto next_sample = dataset_->getNext();
    data_pipeline_.reset(new TensorPipeline(next_sample.data));
    target_pipeline_.reset(new TensorPipeline({next_sample.target}));
  }

  bool IsDone() { return is_done_; }

  size_t GetItersPerEpoch() { return dataset_->GetItersPerEpoch(); }

  void Reset() {
    dataset_->Reset();
    is_done_ = false;
    auto next_sample = dataset_->getNext();
    data_pipeline_->SupplyNext(next_sample.data);
    target_pipeline_->SupplyNext({next_sample.target});
  }

  Example getNext() {
    assert(!is_done_);
    if (dataset_->IsDone()) {
      auto data = data_pipeline_->GetNext({});
      auto target = target_pipeline_->GetNext({});
      is_done_ = true;
      return {data, target.at(0)};
    }

    // std::chrono::_V2::steady_clock::time_point load0 = std::chrono::steady_clock::now();
    auto next_sample = dataset_->getNext();
    // std::chrono::_V2::steady_clock::time_point load1 = std::chrono::steady_clock::now();
    auto data = data_pipeline_->GetNext(next_sample.data);
    // std::chrono::_V2::steady_clock::time_point load2 = std::chrono::steady_clock::now();
    auto target = target_pipeline_->GetNext({next_sample.target});
    // std::chrono::_V2::steady_clock::time_point load3 = std::chrono::steady_clock::now();

  // using msec = std::chrono::duration<double, std::micro>;
  // double load01 = std::chrono::duration_cast<msec>(load1 - load0).count();
  // double load12 = std::chrono::duration_cast<msec>(load2 - load1).count();
  // double load23 = std::chrono::duration_cast<msec>(load3 - load2).count();
  // DP_LOG(
  //     NOTICE,
  //     "dataset.getNext(): %.2f\tdatapipeline.getNext(): %.2f\t targetpipeline.getNext(): %.2f", load01, load12, load23);

    return {data, target.at(0)};
  }

 private:
  std::shared_ptr<Dataset> dataset_;
  std::unique_ptr<TensorPipeline> data_pipeline_;
  std::unique_ptr<TensorPipeline> target_pipeline_;
  bool is_done_{false};
};
