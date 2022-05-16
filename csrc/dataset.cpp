#include "dataset.h"

#include <absl/flags/flag.h>
#include <torch/torch.h>

#include "cifar10.h"
#include "logger.h"

ABSL_FLAG(std::string, cifar_dataset,
          "/home/friedj/mlsf/multimodel/data/cifar-10-batches-bin/", "");

class FakeDataset : public Dataset {
 public:
  FakeDataset(size_t rank, long globalBatchSize,
              std::vector<std::shared_ptr<Layer>> input_layers,
              std::vector<long> sampleIndices, std::function<Example()> gen,
              size_t samples_per_epoch);
  Example getNext() override;
  bool IsDone() override;
  void Reset() override;
  size_t GetItersPerEpoch() override;

 private:
  size_t batches_per_epoch_;
  size_t ctr_{0};
  std::vector<Example> cached_;
};

class CifarDataset : public Dataset {
 public:
  CifarDataset(size_t rank, long globalBatchSize,
               std::vector<std::shared_ptr<Layer>> input_layers,
               std::vector<long> sampleIndices, bool is_eval);
  Example getNext() override;
  bool IsDone() override;
  void Reset() override;
  size_t GetItersPerEpoch() override;

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

FakeDataset::FakeDataset(size_t rank, long globalBatchSize,
                         std::vector<std::shared_ptr<Layer>> input_layers,
                         std::vector<long> sampleIndices,
                         std::function<Example()> gen, size_t samples_per_epoch)
    : Dataset(rank, globalBatchSize, input_layers, sampleIndices) {
  for (size_t i = 0; i < 64; i++)
    cached_.emplace_back(globalToPerRankExample(gen()));
  batches_per_epoch_ = samples_per_epoch / globalBatchSize;
}

size_t FakeDataset::GetItersPerEpoch() { return batches_per_epoch_; };

bool FakeDataset::IsDone() { return ctr_ >= batches_per_epoch_; }

Example FakeDataset::getNext() {
  assert(!IsDone());
  return cached_[ctr_++ % cached_.size()];
}

void FakeDataset::Reset() { ctr_ = 0; }

CifarDataset::CifarDataset(size_t rank, long globalBatchSize,
                           std::vector<std::shared_ptr<Layer>> input_layers,
                           std::vector<long> sampleIndices, bool is_eval)
    : Dataset(rank, globalBatchSize, input_layers, sampleIndices) {
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

Example CifarDataset::getNext() {
  assert(!IsDone());
  auto cur_example = *cur_iter.value();
  cur_iter = ++cur_iter.value();
  return globalToPerRankExample({{cur_example.data}, cur_example.target});
}

size_t CifarDataset::GetItersPerEpoch() { return batches_per_epoch_; };

void CifarDataset::Reset() { cur_iter = loader->begin(); }

Dataset *Dataset::fromName(std::string name, json jobParams, size_t rank,
                           long globalBatchSize,
                           std::vector<std::shared_ptr<Layer>> input_layers,
                           std::vector<long> sampleIndices,
                           size_t fake_train_iters_per_epoch) {
  bool eval = name.find("eval") != std::string::npos;
  if (name.find("cifar") != std::string::npos)
    return new CifarDataset(rank, globalBatchSize, input_layers, sampleIndices,
                            eval);

  long fake_samples = globalBatchSize * fake_train_iters_per_epoch;

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
      return Example(data, target);
    };
    return new FakeDataset(rank, globalBatchSize, input_layers, sampleIndices,
                           gen, eval ? 1000 : fake_samples);
  }

  if (name.find("dlrm") != std::string::npos) {
    DP_LOG(DEBUG, "Using dlrm fake dataset");
    auto targetOpts = torch::TensorOptions().requires_grad(false);
    auto gen = [=] {
      assert(jobParams.contains("dlrm_m_den"));
      assert(jobParams.contains("dlrm_emb_size"));
      assert(jobParams.contains("dlrm_nr_emb"));
      assert(jobParams.contains("dlrm_num_indices_per_lookup"));

      int64_t m_den = jobParams["dlrm_m_den"].get<int64_t>();
      int64_t emb_size = jobParams["dlrm_emb_size"].get<int64_t>();
      int64_t nr_emb = jobParams["dlrm_nr_emb"].get<int64_t>();
      int64_t indices_per_lookup =
          jobParams["dlrm_num_indices_per_lookup"].get<int64_t>();

      DP_LOG(DEBUG,
             "DLRM params: m_den %ld, emb_size %ld, nr_emb %ld, "
             "indices_per_lookup %ld",
             m_den, emb_size, nr_emb, indices_per_lookup);

      auto dense = torch::randn({globalBatchSize, m_den});
      std::vector<torch::Tensor> inputs;
      inputs.push_back(dense);
      for (int64_t i = 0; i < nr_emb; i++) {
        inputs.push_back(
            torch::randint(0, emb_size, {globalBatchSize, indices_per_lookup})
                .to(torch::kInt64));
      }

      auto target = torch::zeros({globalBatchSize, 1}, targetOpts);
      return Example(inputs, target);
    };
    return new FakeDataset(rank, globalBatchSize, input_layers, sampleIndices,
                           gen, eval ? 1000 : fake_samples);
  }

  DP_LOG(DEBUG, "Using fake dataset");
  auto targetOpts =
      torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);
  auto gen = [=] {
    std::vector<torch::Tensor> ts;
    for (auto &iLayer : input_layers) {
      assert(iLayer->emptyInSizes.size() == 1);
      auto inDim = iLayer->emptyInSizes[0];
      inDim[0] = globalBatchSize;
      auto topts = torch::TensorOptions().dtype(iLayer->inOpts.at(0));
      ts.push_back(torch::randn(inDim, topts));
    }
    auto target =
        torch::randint(/*low=*/0, /*high=*/1000, {globalBatchSize}, targetOpts);
    return Example(ts, target);
  };
  return new FakeDataset(rank, globalBatchSize, input_layers, sampleIndices,
                         gen, eval ? 1000 : fake_samples);
}
