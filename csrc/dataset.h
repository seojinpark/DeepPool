#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <torch/torch.h>

#include "runtime.h"
#include "utils.h"

#include "cifar10.h"

class Dataset {
 public:
 
  Dataset(size_t rank, long globalBatchSize,
          std::vector<long> initialBatchSizes, 
          std::vector<long> sampleIndices)
      : rank_(rank),
        globalBatchSize_(globalBatchSize),
        initialBatchSizes_(initialBatchSizes),
        sampleIndices_(sampleIndices){}
  virtual ~Dataset() {}
  virtual std::map<std::string, torch::Tensor> getNext() = 0;
  virtual size_t GetItersPerEpoch() = 0;
  virtual bool IsDone() = 0;
  virtual void Reset() = 0;
  virtual std::map<std::string, torch::Tensor> getNextThisRank() = 0;
  
  static std::shared_ptr<Dataset> fromName(std::string name, size_t rank, long globalBatchSize,
                           std::vector<long> initialBatchSizes,
                           std::vector<long> sampleIndices,
                           size_t fake_train_iters_per_epoch,
                           size_t worldSize = 1);


  // Dataset(Dataset const &) = delete;
  // Dataset &operator=(Dataset const &) = delete;

  // Dataset(Dataset &&moveable){
  //   std::cout << "moving dataset\n";
  //     // moved_from = false;
  //     // moveable.moved_from = true;
  //     // And now we spell out the explicit default move constructor
  // }

  // Dataset &operator=(Dataset &&moveable) {
  //   std::cout << "moving oper - dataset\n";
  //     // moved_from = false;
  //     // moveable.moved_from = true;
  //     // And now we spell out the explicit default move assignment operator
  //     return *this;
  // }

 protected:
  size_t rank_;
  long globalBatchSize_;
  std::vector<long> initialBatchSizes_;
  std::vector<long> sampleIndices_;
};

class TensorPipeline {
 public:
  TensorPipeline(torch::Tensor next) {
    tensorbytes = next.defined() ? next.nbytes() : 0;
    SupplyNext(next);
  }

  void SupplyNext(torch::Tensor next) {
    if (!tensorbytes) {
      next_up_ = next;
      return;
    }

    /* run next HtoD transfer */
    auto origstream = c10::cuda::getCurrentCUDAStream();
    c10::cuda::setCurrentCUDAStream(rtctx->xfer_stream);
    next_up_ = next.to(rtctx->c10dev, /*non_blocking*/ true, /*copy*/ false);
    xfer_ev_.record();
    c10::cuda::setCurrentCUDAStream(origstream);
  }

  torch::Tensor GetNext(c10::optional<torch::Tensor> next) {
    assert(next_up_);
    auto &tsr = next_up_.value();

    if (!tensorbytes) return torch::Tensor();

    /* current stream must wait for xfer before using returned tsr */
    xfer_ev_.block(c10::cuda::getCurrentCUDAStream());

    next_up_ = {};
    if (next) SupplyNext(next.value());
    return tsr;
  }

 private:
  size_t tensorbytes;
  c10::optional<torch::Tensor> next_up_;
  at::cuda::CUDAEvent xfer_ev_;
};

class DatasetPipelineWrapper {
 public:
  DatasetPipelineWrapper(std::shared_ptr<Dataset> dataset) : dataset_(dataset) {
    auto next_sample = dataset_->getNextThisRank();
    for ( const auto &[key, value]: next_sample)
        pipelines_[key].reset(new TensorPipeline(value));
  }

  bool IsDone() { return is_done_; }

  size_t GetItersPerEpoch() { return dataset_->GetItersPerEpoch(); }

  void Reset() {
    dataset_->Reset();
    is_done_ = false;
    auto next_sample = dataset_->getNextThisRank();
    for ( const auto &[key, pipe]: pipelines_)
        pipe->SupplyNext(next_sample[key]);
  }

  std::map<std::string, torch::Tensor> getNextThisRank() {
    assert(!is_done_);
    std::map<std::string, torch::Tensor> rtn_vals;
    if (dataset_->IsDone()) {
      for ( const auto &[key, pipe]: pipelines_)
        rtn_vals[key] = pipe->GetNext({});
      is_done_ = true;
      return rtn_vals;
    }
    else{
      auto next_sample = dataset_->getNextThisRank();
      if(dataset_->IsDone())
        is_done_ = true;
      if (next_sample.size() > 0)
        for ( const auto &[key, pipe]: pipelines_)
            rtn_vals[key] = pipe->GetNext(next_sample[key]);
    }
    return rtn_vals;
  }

 private:
  std::shared_ptr<Dataset> dataset_;
  std::map<std::string, std::unique_ptr<TensorPipeline>> pipelines_;
  bool is_done_{false};
};



class FakeDataset : public Dataset {
 public:
  FakeDataset(size_t rank, long globalBatchSize,
              std::vector<long> initialBatchSizes,
              std::vector<long> sampleIndices,
              std::function<torch::data::Example<>()> gen,
              size_t images_per_epoch);
  std::map<std::string, torch::Tensor> getNextThisRank() override; 
  std::map<std::string, torch::Tensor> getNext() override;
  bool IsDone() override;
  void Reset() override;
  size_t GetItersPerEpoch() override;
  ~FakeDataset() {cached_.clear();};

 private:
  size_t batches_per_epoch_;
  size_t ctr_{0};
  std::vector<torch::data::Example<>> cached_;
};



class CifarDataset : public Dataset {
 public:
  CifarDataset(size_t rank, long globalBatchSize,
               std::vector<long> initialBatchSizes,
               std::vector<long> sampleIndices, bool is_eval);
  std::map<std::string, torch::Tensor> getNext();
  bool IsDone();
  void Reset();
  size_t GetItersPerEpoch();
  std::map<std::string, torch::Tensor> getNextThisRank(); 
  ~CifarDataset() {};

 private:
  c10::optional<torch::data::Iterator<torch::data::Example<>>> cur_iter;
  size_t batches_per_epoch_;

  std::unique_ptr<torch::data::StatelessDataLoader<
      torch::data::datasets::MapDataset<
          torch::data::datasets::MapDataset<
              CIFAR10, torch::data::transforms::Normalize<>>,
          torch::data::transforms::Stack<torch::data::Example<>>>,
      torch::data::samplers::SequentialSampler>>
      loader;
};