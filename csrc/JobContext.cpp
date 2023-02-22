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

#include "JobContext.h"

#include <ATen/autocast_mode.h>
#include <cuda_profiler_api.h>
#include <torch/torch.h>

#include <memory>
#include <string>

#include "BeTask.h"
#include "communication.h"
#include "dataset.h"
#include "logger.h"
#include "runnableModule.h"
#include "runtime.h"
#include "utils.h"

/**
 * Contructs context for a training job.
 */
JobContext::JobContext(std::unique_ptr<RunnableModule> modelIn,
                       std::string name,
                       std::shared_ptr<CommunicationHandler> commHandler,
                       json job_params)
    : model(std::move(modelIn)),
      name(name),
      commHandler(std::move(commHandler)) {
  run_with_be_ = job_params["run_with_be"].get<bool>();
  nr_gpus_ = job_params["nr_gpus"].get<size_t>();

if (job_params.contains("checkpoint_path")) checkpoint_path = job_params["checkpoint_path"].get<std::string>();
if (job_params.contains("loading_path")) loading_path = job_params["loading_path"].get<std::string>();

  std::string dset = "random";
  if (job_params.contains("dset")) dset = job_params["dset"].get<std::string>();

  if (job_params.contains("catsdogs") &&
      job_params["catsdogs"].get<bool>()) {
    dset = "catsDogs";
    /* cifar default includes 10 epochs with test routine */
    runTestRoutine_ = true;
    epochsToTrain = 1;
  } else if (job_params.contains("cifar_training") &&
      job_params["cifar_training"].get<bool>()) {
    dset = "cifar";
    /* cifar default includes 10 epochs with test routine */
    runTestRoutine_ = true;
    epochsToTrain = 20;
  } else if (name.find("gpt2") != std::string::npos) {
    dset = "gpt2";
    runTestRoutine_ = false;
  } else if (name.find("Inception") != std::string::npos) {
    dset = "inception";
  }

  if (job_params.contains("autocast") && job_params["autocast"].get<bool>()) {
    DP_LOG(DEBUG, "Using autocast");
    autocast_ = true;
  }

  if (job_params.contains("run_test_routine"))
    runTestRoutine_ = job_params["run_test_routine"].get<bool>();

  if (job_params.contains("epochs_to_train"))
    epochsToTrain = job_params["epochs_to_train"].get<size_t>();

  train_dataset_.reset(
      Dataset::fromName(dset, job_params, rtctx->rank, model->globalBatchSize,
                        model->input_layers, model->sampleIndices, 2000));
  dataset_pipeline_.reset(new DatasetPipelineWrapper(train_dataset_));

  if (runTestRoutine_) {
    eval_dataset_.reset(Dataset::fromName(
        dset + "_eval", job_params, rtctx->rank, model->globalBatchSize,
        model->input_layers, model->sampleIndices, 10));
  }
  if (!rtctx->use_fg_graph)
    iters_before_graph_capture = itersToTrain * epochsToTrain;
}

/**
 * Destructs context for a training job.
 * Defined here to avoid incomplete type destruction in 'JobContext.h' of
 * some unique_ptr members.
 */
JobContext::~JobContext() {}

void JobContext::printJobStatistics() {
  model->printLayerInGraphTimes();
  size_t iters = totiters - warmupIters;
  using msec = std::chrono::duration<double, std::milli>;
  double elapsed_ms = std::chrono::duration_cast<msec>(end - start).count();
  double total_iter_ms = elapsed_ms / (double)iters;
  double total_iter_ps = 1e3 / total_iter_ms;
  double be_img_ps = be_img_end - be_img_start;
  be_img_ps = 1e3 * be_img_ps / elapsed_ms;
  const auto &timers = model->GetTimers();
  DP_LOG(
      NOTICE,
      "A training job %s is completed (%lu iters, %.2f ms/iter, %.2f iter/s, "
      "%.2f be img/s, %lu globalBatchSize)."
      " AverageTiming (ms) => zero: %.1f, load:%.1f, fp:%.1f, loss:%.1f, "
      "bp:%.1f, opt: %.1f, iter:%.1f"
      " P50 (ms) => fp:%.1f, loss:%.1f, bp:%.1f, iter:%.1f",
      name.c_str(), iters, total_iter_ms, total_iter_ps, be_img_ps,
      model->globalBatchSize, timers.GetAvg("zero", warmupIters),
      timers.GetAvg("load", warmupIters), timers.GetAvg("forward", warmupIters),
      timers.GetAvg("loss", warmupIters),
      timers.GetAvg("backward", warmupIters),
      timers.GetAvg("step", warmupIters), timers.GetAvg("stop", warmupIters),
      timers.GetP50("forward", warmupIters), timers.GetP50("loss", warmupIters),
      timers.GetP50("backward", warmupIters),
      timers.GetP50("stop", warmupIters));
}

/**
 * A helper to run a job.
 *
 * \param job   a context for the job to train.
 * \param[out] jobCompleted
 *    will be filled with true if the job is completely finished.
 *
 * \return    returns non-zero if iteration is finished.
 */
void JobContext::StepOne(bool *iter_done) {
  if (job_done_) return;
  bool graphCapture = totiters == iters_before_graph_capture;
  bool profile = rtctx->profile_layer_times_graph && totiters == 3;

  if (!iter_in_progress) {
    if (rtctx->cuda_profile && totiters == profile_iter_start)
      CUDA_API_CALL(cudaProfilerStart());
    if (graphCapture || profile) GpuManager::getInstance()->Pause();
    if (totiters == warmupIters) {
      rtctx->torch_stream.synchronize();
      start = std::chrono::steady_clock::now();
      be_img_start = GetBeCounter();
    }
  }

  at::autocast::set_enabled(autocast_);

  iter_in_progress = !model->AdvanceTraining(graphCapture, profile);

  at::autocast::set_enabled(false);

  if (iter_done) *iter_done = !iter_in_progress;

  if (!iter_in_progress) {
    if (autocast_) at::autocast::clear_cache();
    if (graphCapture || profile) GpuManager::getInstance()->Resume();
    if (rtctx->cuda_profile &&
        totiters == profile_iter_start + niter_to_profile - 1) {
      CUDA_API_CALL(cudaProfilerStop());
      exit(0);
    }
    rtctx->fgcounter++;
    if (profile) {
      job_done_ = true;
      return;
    }
    ++totiters;
  }
}

void JobContext::Test() {
  double total = 0.0;
  torch::Tensor correct = torch::zeros({1}).to(at::kLong).to(rtctx->c10dev);


  if (iters_before_graph_capture < totiters && rtctx->use_fg_graph)
    iters_before_graph_capture = totiters + 5;

  size_t i = 0;
  while (!eval_dataset_->IsDone()) {
    total += model->GetGlobalBatchSize();

    auto batch = eval_dataset_->getNext();
    for (auto &input : batch.data)
      if (input.defined())
        input = input.to(rtctx->c10dev);
    auto output = Infer(batch.data);
    if (output.defined() && output.nbytes() > 0) {
      auto pred = output.argmax(1);
      correct += pred.eq(batch.target.to(rtctx->c10dev)).sum();
    }
    DP_LOG(DEBUG, "Evaluate iteration %lu/%lu\n", ++i,
           eval_dataset_->GetItersPerEpoch());
  }

  iters_before_graph_capture = 0;

  if (nr_gpus_ > 1) {
    rtctx->torch_stream.synchronize();  // sync before calling into NCCL
    commHandler->comm_start();
    commHandler->all_reduce(correct, c10d::ReduceOp::SUM);
    commHandler->comm_end();
    commHandler->sync();
  }

  double corr = correct.item().toDouble();

  DP_LOG(NOTICE, "Evaluate: Total: %.1f Correct: %.1f | Accuracy: %.3f", total,
         corr, static_cast<double>(corr) / total);
  
    eval_dataset_->Reset();
}

torch::Tensor JobContext::Infer(std::vector<torch::Tensor> inputs) {
  torch::NoGradGuard guard;
  model->SetEval();
  model->SetInputsTargets(inputs, {});
  FinishIteration();
  return model->getOutput();
}

void JobContext::Train(std::vector<torch::Tensor> inputs,
                       torch::Tensor target) {
  model->SetTrain();
  model->SetInputsTargets(inputs, target);
  FinishIteration();
}

void JobContext::TrainOneEpoch() {
      std::chrono::_V2::steady_clock::time_point start = std::chrono::steady_clock::now();
  
  double averageDataTime = 0;
  double averageTrainTime = 0;
  size_t i = 0;
  if (!model->isTrain() && iters_before_graph_capture < totiters &&
      rtctx->use_fg_graph)
    iters_before_graph_capture = totiters + 5;
  while (!dataset_pipeline_->IsDone() && !job_done_) {
    std::chrono::_V2::steady_clock::time_point startDataLoading = std::chrono::steady_clock::now();
    auto batch = dataset_pipeline_->getNext();
    std::chrono::_V2::steady_clock::time_point endDataLoading = std::chrono::steady_clock::now();

    std::chrono::_V2::steady_clock::time_point startTraining = std::chrono::steady_clock::now();
    Train(batch.data, batch.target);
    std::chrono::_V2::steady_clock::time_point endTrianing = std::chrono::steady_clock::now();

  using msec = std::chrono::duration<double, std::micro>;

  double dataTime = std::chrono::duration_cast<msec>(endDataLoading - startDataLoading).count();
  double trainTime = std::chrono::duration_cast<msec>(endTrianing - startTraining).count();

  DP_LOG(
      NOTICE,
      "dataloading: %.2f\ttraining: %.2f", dataTime, trainTime);
  i++;
  averageDataTime += dataTime;
  averageTrainTime += trainTime;

    // DP_LOG(DEBUG, "Training iteration %lu/%lu\n", ++i,
    //        dataset_pipeline_->GetItersPerEpoch());
  }

  end = std::chrono::steady_clock::now();

    std::chrono::_V2::steady_clock::time_point startSync0 = std::chrono::steady_clock::now();
  double loss = model->GetAvgLoss();
      std::chrono::_V2::steady_clock::time_point startSync1 = std::chrono::steady_clock::now();
  DP_LOG(NOTICE, "Epoch done. Loss %.2f", loss);
  iters_before_graph_capture = 0;
  rtctx->torch_stream.synchronize();
      std::chrono::_V2::steady_clock::time_point startSync2 = std::chrono::steady_clock::now();
  be_img_end = GetBeCounter();
      std::chrono::_V2::steady_clock::time_point startSync3 = std::chrono::steady_clock::now();
  dataset_pipeline_->Reset();

  std::chrono::_V2::steady_clock::time_point endSync = std::chrono::steady_clock::now();
  using msec = std::chrono::duration<double, std::micro>;
    double syncTime01 = std::chrono::duration_cast<msec>(startSync1 - startSync0).count();
  double syncTime12 = std::chrono::duration_cast<msec>(startSync2 - startSync1).count();
  double syncTime23 = std::chrono::duration_cast<msec>(startSync3 - startSync2).count();
  double syncTime34 = std::chrono::duration_cast<msec>(endSync - startSync3).count();

    DP_LOG(
      NOTICE,
      "sync times %.2f, %.2f, %.2f, %.2f", syncTime01, syncTime12, syncTime23, syncTime34);
      

  double syncTime = std::chrono::duration_cast<msec>(endSync - startSync0).count();

  double elapsed_ms = std::chrono::duration_cast<msec>(end - start).count();
  double total_iter_ms = elapsed_ms / (double)i;
  double total_iter_ps = 1e6 / total_iter_ms;
  DP_LOG(
      NOTICE,
      "epoch time: %.2f ms, teardown time: %.2f ms, iter/s : %.2f, average data loading: %.2f ,s, average training time: %.2f ms", elapsed_ms / 1e3, syncTime / 1e3, total_iter_ps, averageDataTime/i/1e3, averageTrainTime/i/1e3);
      
}

/**
 * A helper to run a job.
 *
 * \param job   a context for the job to train.
 * \param[out] jobCompleted
 *    will be filled with true if the job is completely finished.
 *
 * \return    returns non-zero if iteration is finished.
 */
void JobContext::FinishIteration() {
  bool iter_done = false;

  do {
    StepOne(&iter_done);
  } while (!iter_done && !job_done_);


    
}


void JobContext::save_variables(){

  if (checkpoint_path.empty()) {
        std::cout << "No checkpoint path set, failed to save model." << std::endl;
        return;
  }
        model->saveOptimizer(checkpoint_path + "/optimizer.pt");
    // torch::save(*optimizer_.get(), std::filesystem::path(checkpoint_path_).append("optimizer.pt").string());
    for (auto layer : model->layers){
        if(layer->active){
            // layer->module.train();
            layer->saveModule(checkpoint_path + "/" + layer->layername + ".pt");
        }
    }

    DP_LOG(
      NOTICE,
      "Saved model to %s", checkpoint_path.c_str());
}

void JobContext::restore_variables(){
    if (loading_path.empty()) {
        std::cout << "No path set, failed to load model." << std::endl;
        return;
  }

    for (auto layer : model->layers){
        if(layer->active){
            layer->loadModule(loading_path + "/" + layer->layername + ".pt");
            layer->module.to(rtctx->c10dev);
            // layer->module.train();
            // for (const auto& params : layer->module.parameters()){
            //     // std::cout << params.requires_grad() << std::endl;
            //     model->parameters.push_back(params);
            // }
            // layer->module.to(rtctx->c10dev);
        }
    }
    
    model->SetupOptimizer();
    model->SetupOptimizer(loading_path + "/optimizer.pt");
        DP_LOG(
      NOTICE,
      "Loaded model from %s", loading_path.c_str());
}
