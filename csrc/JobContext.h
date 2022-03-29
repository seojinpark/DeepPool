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

#ifndef TASK_MANAGER_H
#define TASK_MANAGER_H

#include <torch/torch.h>

#include <memory>
#include <string>

#include "json.hpp"
using json = nlohmann::json;

/**
 * Forward declarations. Do not include headers unless necessary.
 */
class RunnableModule;
class CommunicationHandler;
class Dataset;
class DatasetPipelineWrapper;
class AutoLRSClient;

/**
 * Context holding data for each training task.
 */
class JobContext {
 public:
  JobContext(std::unique_ptr<RunnableModule> model, std::string name,
             std::shared_ptr<CommunicationHandler> commHandler,
             json job_params);
  ~JobContext();

  bool RunWithBe() const { return run_with_be_; }
  size_t GetEpochsToTrain() const { return epochsToTrain; }
  bool ShouldRunTest() const { return runTestRoutine_; }

  /* Run a sample through the NN */
  std::tuple<torch::Tensor, torch::Tensor> Infer(torch::Tensor input, 
                                                  torch::Tensor target = {}, 
                                                  torch::Tensor weights = {});

  /* Run one training iteration with this input/target */
  void Train(torch::Tensor input,
             torch::Tensor target, 
             torch::Tensor weights = {});

  /* Test the model on the test dataset */
  void Test(int64_t curEpoch = -1, bool do_stats = false);

  /* Test the model on the test dataset */
  double Validation(int64_t curEpoch = -1, bool trackLoss=false);

  /* Advance one step through the the model */
  void StepOne(bool *iter_done);

  /* Advance to the end of an iteration*/
  void FinishIteration();

  /* Train one full epoch */
  void TrainOneEpoch(int64_t curEpoch = -1, bool trackLoss=false);

  void printJobStatistics();
  void graphLossResults();
  void graphLearningRateResults();

  double updateScalar(double val, uint32_t root);

  std::shared_ptr<RunnableModule> model;
  std::string name;
  std::shared_ptr<CommunicationHandler> commHandler;
  std::chrono::time_point<std::chrono::steady_clock> start, end;
  uint64_t be_img_start, be_img_end;

  size_t getTrainItersPerEpoch();
  bool shouldEarlyStop(double valLoss);
  bool isLowerLoss(double valLoss);
  void saveModel(std::string inpath="");
  void restoreModel(std::string inpath="");
  size_t getWarmupIters(){return warmupIters;};
  std::string getCheckpointDir(){return checkpointDir;}
  size_t getNGpus(){return nr_gpus_;}
  double getLastValLoss(){return val_loss_tracker_.back();}

 private:
  // Params
  bool run_with_be_{false};
  bool runTestRoutine_{false};
  size_t nr_gpus_;
  size_t epochsToTrain{1};
  size_t itersToTrain{5000};
  size_t warmupIters{200};
  size_t profile_iter_start{3};
  size_t niter_to_profile{5};
  bool autocast_{false};
  bool earlyStopping{true};
  double earlyMinImprovement{0.03};
  size_t earlyRetriesSet{32};
  size_t earlyRetries{earlyRetriesSet};
  size_t earlyWindowSize{16};
  std::deque<double> earlyLossLog;
  std::string checkpointDir;
  double lowestLossSeen{std::numeric_limits<double>::infinity()};

  bool job_done_{false};

  std::shared_ptr<Dataset> train_dataset_;
  std::shared_ptr<Dataset> eval_dataset_;
  std::shared_ptr<Dataset> validation_dataset_;
  std::shared_ptr<DatasetPipelineWrapper> dataset_pipeline_;

  std::shared_ptr<AutoLRSClient> lrsched_;
  // std::shared_ptr<torch::optim::StepLR> lrsched_;
  std::vector<double> train_loss_tracker_;
  std::vector<double> val_loss_tracker_;
  std::vector<double> lr_tracker_;

  bool iter_in_progress{false};
  size_t totiters{0};                     // total iters executed
  size_t iters_before_graph_capture{10};  // set high to disable graph capture
};

#endif  // TASK_MANAGER_H