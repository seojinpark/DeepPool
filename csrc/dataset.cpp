#include "dataset.h"

#include <absl/flags/flag.h>
#include <torch/torch.h>

#include "cifar10.h"
#include "catsDogs.h"
#include "logger.h"

ABSL_FLAG(std::string, cifar_dataset,
          "/home/friedj/data/cifar-10-batches-bin/", "");

class CatsDogsDataset : public Dataset
{
public:
  CatsDogsDataset(size_t worldSize, size_t rank, long globalBatchSize,
                  std::vector<std::shared_ptr<Layer>> input_layers,
                  std::vector<long> sampleIndices, bool is_eval, std::string filepath, int num_workers);
  Example getNext() override;
  bool IsDone() override;
  void Reset() override;
  size_t GetItersPerEpoch() override;

private:
  c10::optional<torch::data::Iterator<torch::data::Example<>>> cur_iter;
  size_t batches_per_epoch_;

  std::unique_ptr<torch::data::StatelessDataLoader<
      torch::data::datasets::MapDataset<
          CatsDogs, torch::data::transforms::Stack<torch::data::Example<>>>,
      torch::data::samplers::DistributedRandomSampler>>
      loader;

  int iteration_count;
  int epoch_count;
};

CatsDogsDataset::CatsDogsDataset(size_t worldSize, size_t rank, long globalBatchSize,
                                 std::vector<std::shared_ptr<Layer>> input_layers,
                                 std::vector<long> sampleIndices, bool is_eval, std::string filepath, int num_workers)
    : Dataset(worldSize, rank, globalBatchSize, input_layers, sampleIndices)
{
  DP_LOG(NOTICE, "Using CatsDogs dataset");

  auto dataset = CatsDogs(filepath, worldSize, rank)
               .map(torch::data::transforms::Stack<>());
  batches_per_epoch_ = dataset.size().value() / (globalBatchSize/worldSize);

  epoch_count = 0;
  torch::data::samplers::DistributedRandomSampler sampler = torch::data::samplers::DistributedRandomSampler(dataset.size().value(), /*num_replicas=*/worldSize, /*rank=*/rank);
  sampler.set_epoch(epoch_count);

  loader = torch::data::make_data_loader(
                  std::move(dataset),
                  sampler,
                  torch::data::DataLoaderOptions().batch_size(globalBatchSize/worldSize).workers(num_workers).drop_last(true));

  // loader =
  //     torch::data::make_data_loader<torch::data::samplers::DistributedRandomSampler>(
  //         std::move(dataset), torch::data::DataLoaderOptions().batch_size(globalBatchSize/worldSize).workers(num_workers).drop_last(true));

  // torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
  // std::move(c), torch::data::DataLoaderOptions().batch_size(globalBatchSize).drop_last(true)); // good rule of thumb is number of workers equal to CPU cores
  cur_iter = loader->begin();
}

bool CatsDogsDataset::IsDone()
{
  if (cur_iter == loader->end())
  {
    // std::cout << "Total iterations: " << iteration_count << std::endl;
    return true;
  }
  else if (cur_iter.value()->data.sizes().vec()[0] < localBatchSize)
  {
    // std::cout << "Total iterations: " << iteration_count << std::endl;
    return true;
  }

  return false;
}

Example CatsDogsDataset::getNext()
{
  assert(!IsDone());
  auto cur_example = *cur_iter.value();
  cur_iter = ++cur_iter.value();

  iteration_count++;
  // if (iteration_count < 10 || iteration_count > batches_per_epoch_ - 10)
  // {
  //   // save hash to make sure is valid
  //   std::stringstream buffer;
  //   buffer << cur_example.data;
  //   std::hash<std::string> hasher;
  //   size_t hash = hasher(buffer.str());

  //   std::ofstream outfile;
  //   outfile.open("imageHashes.txt", std::ios_base::app); // append instead of overwrite
  //   outfile << iteration_count << " " << hash << std::endl;
  // }

  // std::cout << cur_example.target << std::endl;
  // std::cout << cur_example.data << std::endl;

  // converts pytorch Example to our own class Example
  // return globalToPerRankExample({cur_example.data, cur_example.target});


  //           std::chrono::_V2::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  // torch::Tensor pinnedData = cur_example.data.pin_memory();
  //           std::chrono::_V2::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  //   torch::Tensor pinnedTarget = cur_example.target.pin_memory();
  //           std::chrono::_V2::steady_clock::time_point t2 = std::chrono::steady_clock::now();

  // using msec = std::chrono::duration<double, std::micro>;
  // double load0 = std::chrono::duration_cast<msec>(t1 - t0).count();
  //   double load1 = std::chrono::duration_cast<msec>(t2 - t1).count();

  // DP_LOG(
  //     NOTICE,
  //     "pinning data: %.2f\tpinning target: %.2f", load0, load1);

    return {cur_example.data, cur_example.target};

}

size_t CatsDogsDataset::GetItersPerEpoch() { return batches_per_epoch_; };

void CatsDogsDataset::Reset()
{
  // must reset before creating the iterator
  epoch_count++;
  loader->sampler_.set_epoch(epoch_count);
  std::cout << "Setting epoch to " << epoch_count << std::endl;

  cur_iter = loader->begin();
  iteration_count = 0;
}

Dataset *Dataset::fromName(std::string name, json jobParams, size_t worldSize, size_t rank,
                           long globalBatchSize,
                           std::vector<std::shared_ptr<Layer>> input_layers,
                           std::vector<long> sampleIndices,
                           size_t fake_train_iters_per_epoch)
{

  bool eval = name.find("eval") != std::string::npos;

  if (name.find("catsDogs") != std::string::npos)
  {
    int num_workers = 1;
    if (jobParams.contains("num_workers"))
    {
      num_workers = jobParams["num_workers"].get<int>();
    }

    // evaluation dataset is different from training dataset
    if (jobParams.contains("evaluation_data") && eval)
    {
      std::string data_path = jobParams["evaluation_data"].get<std::string>();
      return new CatsDogsDataset(worldSize, rank, globalBatchSize, input_layers, sampleIndices,
                                 eval, data_path, num_workers);
    }

    if (jobParams.contains("training_data"))
    {
      std::string data_path = jobParams["training_data"].get<std::string>();
      return new CatsDogsDataset(worldSize, rank, globalBatchSize, input_layers, sampleIndices,
                                 eval, data_path, num_workers);
    }
    else
    {
      DP_LOG(DEBUG, "CatsDogs dataset cannot be used unless a filepath to the labeled csv is given. Add \"training_data\":\"my/path/some_data.csv\" to the job parameters.");
    }
  }
}
