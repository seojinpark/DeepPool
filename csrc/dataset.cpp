#include "dataset.h"

#include <absl/flags/flag.h>
#include <torch/torch.h>

#include "logger.h"
#ifdef ENABLE_STREAMING_DATASET
#include "streamingDataset.h"
#endif

ABSL_FLAG(std::string, cifar_dataset,
          "/home/friedj/mlsf/multimodel/data/cifar-10-batches-bin/", "");

FakeDataset::FakeDataset(size_t rank, long globalBatchSize,
                         std::vector<long> initialBatchSizes,
                         std::vector<long> sampleIndices,
                         std::function<torch::data::Example<>()> gen,
                         size_t images_per_epoch)
    : Dataset(rank, globalBatchSize, initialBatchSizes, sampleIndices) {
  for (size_t i = 0; i < 64; i++) cached_.emplace_back(gen());
  batches_per_epoch_ = images_per_epoch / globalBatchSize;
}

size_t FakeDataset::GetItersPerEpoch() { return batches_per_epoch_; };

bool FakeDataset::IsDone() { return ctr_ >= batches_per_epoch_; }

std::map<std::string, torch::Tensor> FakeDataset::getNext() {
  assert(!IsDone());
  torch::data::Example<> vals = cached_[ctr_++ % cached_.size()];
  return {{std::string("data"), vals.data}, {std::string("target"), vals.target}};
}

void FakeDataset::Reset() { ctr_ = 0; }

std::map<std::string, torch::Tensor> FakeDataset::getNextThisRank(){
  std::map<std::string, torch::Tensor> rtn;
  auto ex = getNext();
  if (initialBatchSizes_.at(rank_))
    rtn["data"] = ex["data"].split_with_sizes(initialBatchSizes_)[rank_];

  if (sampleIndices_.size()){
    std::vector<long> spl(globalBatchSize_, 1);
    auto splitsamples =
        ex["target"].split_with_sizes(spl); // TODO make this clean....
    std::vector<torch::Tensor> samplesOrdered;
    for (auto &s : sampleIndices_)
      samplesOrdered.push_back(splitsamples.at(s));
    rtn["target"] = torch::cat(samplesOrdered);
  }
  return rtn;
}





CifarDataset::CifarDataset(size_t rank, long globalBatchSize,
                           std::vector<long> initialBatchSizes,
                           std::vector<long> sampleIndices, bool is_eval)
    : Dataset(rank, globalBatchSize, initialBatchSizes, sampleIndices) {
  DP_LOG(DEBUG, "Using CIFAR dataset");
  auto c = CIFAR10(absl::GetFlag(FLAGS_cifar_dataset),
                   is_eval ? CIFAR10::Mode::kTest : CIFAR10::Mode::kTrain)
               .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406},
                                                         {0.229, 0.224, 0.225}))
               .map(torch::data::transforms::Stack<>());
  batches_per_epoch_ = c.size().value() / globalBatchSize;
  loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(c), globalBatchSize);
  cur_iter = loader->begin();
}

bool CifarDataset::IsDone() {
  if (cur_iter == loader->end())
    return true;
  else if (cur_iter.value()->data.sizes().vec()[0] < globalBatchSize_)
    return true;

  return false;
}

std::map<std::string, torch::Tensor> CifarDataset::getNextThisRank(){
  std::map<std::string, torch::Tensor> rtn;
  auto ex = getNext();

  // torch::Tensor data, target;
  if (initialBatchSizes_.at(rank_))
    rtn["data"] = ex["data"].split_with_sizes(initialBatchSizes_)[rank_];

  if (sampleIndices_.size()){
    std::vector<long> spl(globalBatchSize_, 1);
    auto splitsamples =
        ex["target"].split_with_sizes(spl); // TODO make this clean....
    std::vector<torch::Tensor> samplesOrdered;
    for (auto &s : sampleIndices_)
      samplesOrdered.push_back(splitsamples.at(s));
    rtn["target"] = torch::cat(samplesOrdered);
  }
  return rtn;
}

std::map<std::string, torch::Tensor> CifarDataset::getNext()
{
  assert(!IsDone());
  auto cur_example = *cur_iter.value();
  cur_iter = ++cur_iter.value();
  return {{std::string("data"), cur_example.data}, {std::string("target"), cur_example.target}};
}

size_t CifarDataset::GetItersPerEpoch() { return batches_per_epoch_; };

void CifarDataset::Reset() { cur_iter = loader->begin(); }


std::shared_ptr<Dataset> Dataset::fromName(std::string name, size_t rank, long globalBatchSize,
                           std::vector<long> initialBatchSizes,
                           std::vector<long> sampleIndices,
                           size_t fake_train_iters_per_epoch,
                           size_t worldSize) {
  bool eval = name.find("eval") != std::string::npos;
  bool validation = name.find("validation") != std::string::npos;
  if (name.find("cifar") != std::string::npos)
    return std::make_shared<CifarDataset>(rank, globalBatchSize, initialBatchSizes,
                            sampleIndices, eval);
#ifdef ENABLE_STREAMING_DATASET
  else if (name.find("anvil") != std::string::npos)
    return std::make_shared<StreamingDataset>(rank, globalBatchSize, initialBatchSizes,
                            sampleIndices, eval, worldSize, validation);
#endif

  long fake_images = globalBatchSize * fake_train_iters_per_epoch;

  if (name.find("gpt2") != std::string::npos) {
    DP_LOG(DEBUG, "Using GPT2 fake dataset");
    auto dopts = torch::TensorOptions().dtype(torch::kInt32);
    auto topts =
        torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
    auto gen = [=] {
      auto data = torch::randint(/*low=*/0, /*high=*/1024,
                                 {globalBatchSize, 1024}, dopts);
      auto target = torch::randint(/*low=*/0, /*high=*/1024,
                                   {globalBatchSize, 1024}, topts);
      return torch::data::Example<>(data, target);
    };
    return std::make_shared<FakeDataset>(rank, globalBatchSize, initialBatchSizes,
                           sampleIndices, gen, eval ? 1000 : fake_images);
  }

  DP_LOG(DEBUG, "Using fake dataset");
  long px = name.find("inception") != std::string::npos ? 299 : 224;
  auto targetOpts =
      torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
  auto gen = [=] {
    auto data = torch::randn({globalBatchSize, 3, px, px});
    auto target =
        torch::randint(/*low=*/0, /*high=*/1000, {globalBatchSize}, targetOpts);
    return torch::data::Example<>(data, target);
  };
  return std::make_shared<FakeDataset>(rank, globalBatchSize, initialBatchSizes,
                         sampleIndices, gen, eval ? 1000 : fake_images);
}