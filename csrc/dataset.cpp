#include "dataset.h"

#include <absl/flags/flag.h>
#include <torch/torch.h>

#include "cifar10.h"

ABSL_FLAG(std::string, cifar_dataset,
          "/home/friedj/mlsf/multimodel/data/cifar-10-batches-bin/", "");

class FakeDataset : public Dataset {
 public:
  FakeDataset(size_t rank, long globalBatchSize,
              std::vector<long> initialBatchSizes,
              std::vector<long> sampleIndices, std::vector<long> inputShape,
              size_t target_classes, size_t images_per_epoch);
  torch::data::Example<> getNext() override;
  bool IsDone() override;
  void Reset() override;

 private:
  size_t batches_per_epoch_;
  size_t ctr_;
  std::vector<torch::data::Example<>> cached_;
};

class CifarDataset : public Dataset {
 public:
  CifarDataset(size_t rank, long globalBatchSize,
               std::vector<long> initialBatchSizes,
               std::vector<long> sampleIndices, bool is_eval);
  torch::data::Example<> getNext() override;
  bool IsDone() override;
  void Reset() override;

 private:
  c10::optional<torch::data::Iterator<torch::data::Example<>>> cur_iter;

  std::unique_ptr<torch::data::StatelessDataLoader<
      torch::data::datasets::MapDataset<
          torch::data::datasets::MapDataset<
              CIFAR10, torch::data::transforms::Normalize<>>,
          torch::data::transforms::Stack<torch::data::Example<>>>,
      torch::data::samplers::SequentialSampler>>
      loader;
};

FakeDataset::FakeDataset(size_t rank, long globalBatchSize,
                         std::vector<long> initialBatchSizes,
                         std::vector<long> sampleIndices,
                         std::vector<long> inputShape, size_t target_classes,
                         size_t images_per_epoch)
    : Dataset(rank, globalBatchSize, initialBatchSizes, sampleIndices) {
  std::vector<long> fullShape;
  fullShape.push_back(globalBatchSize);
  fullShape.insert(fullShape.end(), inputShape.begin(), inputShape.end());
  auto targetOpts = torch::TensorOptions().dtype(torch::kInt64);
  for (size_t i = 0; i < 64; i++) {
    auto data = torch::randn(inputShape);
    auto target = torch::randint(/*low=*/0, /*high=*/target_classes,
                                 {globalBatchSize}, targetOpts);
    cached_.emplace_back(data, target);
  }
  batches_per_epoch_ = images_per_epoch / globalBatchSize;
}

bool FakeDataset::IsDone() { return ctr_ >= batches_per_epoch_; }

torch::data::Example<> FakeDataset::getNext() {
  assert(!IsDone());
  return cached_[ctr_++ % cached_.size()];
}

void FakeDataset::Reset() { ctr_ = 0; }

CifarDataset::CifarDataset(size_t rank, long globalBatchSize,
                           std::vector<long> initialBatchSizes,
                           std::vector<long> sampleIndices, bool is_eval)
    : Dataset(rank, globalBatchSize, initialBatchSizes, sampleIndices) {
  auto c = CIFAR10(absl::GetFlag(FLAGS_cifar_dataset),
                   is_eval ? CIFAR10::Mode::kTest : CIFAR10::Mode::kTrain)
               .map(torch::data::transforms::Normalize<>({0.485, 0.456, 0.406},
                                                         {0.229, 0.224, 0.225}))
               .map(torch::data::transforms::Stack<>());
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

torch::data::Example<> CifarDataset::getNext() {
  assert(!IsDone());
  auto cur_example = *cur_iter.value();
  cur_iter = ++cur_iter.value();
  return cur_example;
}

void CifarDataset::Reset() { cur_iter = loader->begin(); }

Dataset *Dataset::fromName(std::string name, size_t rank, long globalBatchSize,
                           std::vector<long> initialBatchSizes,
                           std::vector<long> sampleIndices) {
  if (name.find("cifar") != std::string::npos) {
    bool eval = name.find("eval") != std::string::npos;
    return new CifarDataset(rank, globalBatchSize, initialBatchSizes,
                            sampleIndices, eval);
  }
  long px = name.find("inception") != std::string::npos ? 299 : 224;
  return new FakeDataset(rank, globalBatchSize, initialBatchSizes,
                         sampleIndices, {3, px, px}, 1000, 100000);
}