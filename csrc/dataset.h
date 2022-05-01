
#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>

#include "runnableModule.h"
#include "runtime.h"
#include "utils.h"

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
  static Dataset *fromName(std::string name, size_t rank, long globalBatchSize,
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
      data.push_back(ex.data[i].index({Slice(sample_start, sample_end)}));
    }

    std::vector<torch::Tensor> samplesOrdered;
    for (auto &slice : slices_)
      samplesOrdered.push_back(ex.target.index({slice}));
    if (samplesOrdered.size()) target = torch::cat(samplesOrdered);

    return {data, target};
  }

 protected:
  long globalBatchSize_;
  Dataset(size_t rank, long globalBatchSize,
          std::vector<std::shared_ptr<Layer>> input_layers,
          std::vector<long> sampleIndices)
      : globalBatchSize_(globalBatchSize),
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
  size_t rank_;
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
    for (auto &n : next) {
      if (n.defined() && n.nbytes())
        next_up_.push_back(
            n.to(rtctx->c10dev, /*non_blocking*/ true, /*copy*/ false));
      else
        next_up_.push_back(n);
    }
    c10::cuda::setCurrentCUDAStream(origstream);
  }

  std::vector<torch::Tensor> GetNext(std::vector<torch::Tensor> next) {
    /* current stream must wait for xfer before using returned tsr */
    xfer_ev_.record(rtctx->xfer_stream);
    xfer_ev_.block(c10::cuda::getCurrentCUDAStream());

    auto out = next_up_;
    next_up_.clear();
    if (next.size()) SupplyNext(next);
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
    auto next_sample = dataset_->getNext();
    auto data = data_pipeline_->GetNext(next_sample.data);
    auto target = target_pipeline_->GetNext({next_sample.target});

    return {data, target.at(0)};
  }

 private:
  std::shared_ptr<Dataset> dataset_;
  std::unique_ptr<TensorPipeline> data_pipeline_;
  std::unique_ptr<TensorPipeline> target_pipeline_;
  bool is_done_{false};
};
