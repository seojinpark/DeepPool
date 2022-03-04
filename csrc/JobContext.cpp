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

#ifdef ENABLE_STREAMING_DATASET
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "stats.h"
#endif

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

  std::string dset = "random";
  if (job_params.contains("cifar_training") &&
      job_params["cifar_training"].get<bool>()) {
    dset = "cifar";
    /* cifar default includes 10 epochs with test routine */
    runTestRoutine_ = true;
    epochsToTrain = 10;
  } else if (name.find("gpt2") != std::string::npos) {
    dset = "gpt2";
    runTestRoutine_ = false;
  } else if (name.find("Inception") != std::string::npos) {
    dset = "inception";
#ifdef ENABLE_STREAMING_DATASET
  } else if (name.find("anvil") != std::string::npos) {
    dset = "anvil";
    epochsToTrain = 300;
    if (job_params.contains("epochsToTrain"))
      epochsToTrain = job_params["epochsToTrain"].get<uint64_t>();
    runTestRoutine_ = true;
#endif
  }

  if (job_params.contains("autocast") && job_params["autocast"].get<bool>()) {
    DP_LOG(DEBUG, "Using autocast");
    autocast_ = true;
  }

  if (job_params.contains("run_test_routine"))
    runTestRoutine_ = job_params["run_test_routine"].get<bool>();

  if (job_params.contains("epochs_to_train"))
    epochsToTrain = job_params["epochs_to_train"].get<size_t>();

  train_dataset_ = Dataset::fromName(dset, rtctx->rank, model->globalBatchSize,
                        model->initialBatchSizes, model->sampleIndices, 2000, rtctx->worldSize);

  eval_dataset_ = Dataset::fromName(dset + "_eval", rtctx->rank, model->globalBatchSize,
                        model->initialBatchSizes, model->sampleIndices, 10, rtctx->worldSize);

  dataset_pipeline_ = std::make_shared<DatasetPipelineWrapper>(train_dataset_);

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
    if ((graphCapture || profile) && IsBeEnabled()) BePause();
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
    if ((graphCapture || profile) && IsBeEnabled() && run_with_be_) BeResume();
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
    if (!graphCapture)
      ++totiters;
  }
}

#ifdef ENABLE_STREAMING_DATASET
auto tensorToCvImage(at::Tensor tensor)
{
    int width = tensor.sizes()[0];
    int height = tensor.sizes()[1];
    int channels = tensor.sizes()[2];
    auto dtype = CV_8UC3;

    if (channels == 1)
      dtype = CV_8UC1;
    try
    {
        cv::Mat output_mat(height, width, dtype, tensor.contiguous().data_ptr());
        return output_mat.clone();
    }
    catch (const c10::Error& e)
    {
        std::cout << "an error has occured : " << e.msg() << std::endl;
    }
    auto rtnmat = cv::Mat(height, width, dtype);
    // cv::imwrite("test.png", rtnmat);
    return rtnmat;
}

void JobContext::Test(int64_t curEpoch) {
  UNUSED(curEpoch);
  double total = 0.0;
  torch::Tensor correct = torch::zeros({1}).to(at::kLong).to(rtctx->c10dev);

  eval_dataset_->Reset();

  if (iters_before_graph_capture < totiters && rtctx->use_fg_graph)
    iters_before_graph_capture = totiters + 5;

  int64_t frames = 1;
  std::vector<std::string> label_list = {"background", "big_plane", "small_plane"};
  auto stats = Stats(label_list, "background", frames, 0.01, 0.3, "chip", 0.8);
  auto totals = torch::zeros({3}, torch::TensorOptions().dtype(torch::kInt32)).to(rtctx->c10dev).contiguous();

  size_t iters = 0;
  while (!eval_dataset_->IsDone()) {
    auto batch = eval_dataset_->getNextThisRank();
    if(batch.find("idx") == batch.end())
      continue;

    torch::Tensor input = batch["data"];
    auto bi = batch["idx"].item<int64_t>();

    if (input.defined()) input = input.to(rtctx->c10dev);
    auto output = Infer(input).cpu().contiguous();

    if (output.defined() && output.nbytes() > 0) {
      auto pred = torch::softmax(output, 1).clone();
      auto one_hot = torch::zeros_like(pred, c10::TensorOptions().dtype(torch::kFloat32)).scatter(
                        1, 
                        batch["target"].index({torch::indexing::Slice(torch::indexing::None),
                                               torch::indexing::None,
                                               torch::indexing::Slice(torch::indexing::None), 
                                               torch::indexing::Slice(torch::indexing::None)}),
                        1.0);
      auto images = input.permute({0, 2, 3, 1}).cpu().detach().clone();
      auto labels = one_hot.permute({0, 2, 3, 1}).cpu().detach().clone();
      pred = pred.permute({0, 2, 3, 1}).cpu().detach();

      for (int64_t index = 0; index < images.sizes()[0]; index++){
          std::map<std::string, at::Tensor> result({
            // {"iid", iid},
            // {"view", view[index]}, 
            {"image", images.index({index, torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 1)})}, 
            {"pred", pred.index({index, torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice()})}, 
            {"labels", labels.index({index, torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice()})}
            // {"size", osize[index]}
          });

          auto res = stats.batch(result);

          totals.index_put_({0}, totals.index({0}).item<int32_t>()+std::get<0>(res));
          totals.index_put_({1}, totals.index({1}).item<int32_t>()+std::get<1>(res));
          totals.index_put_({2}, totals.index({2}).item<int32_t>()+std::get<2>(res));

          if ((std::get<0>(res) > 0 || std::get<1>(res) > 0 || std::get<2>(res) > 0)){ // && curEpoch > 20) {
              char buff[1024];
              snprintf(buff, sizeof(buff), "/DeepPool/samples/rank%d_iter%ld_index%ld_0.jpg", rtctx->rank, iters, index);
              std::string buffAsStdStr = buff;
              auto cvImg = tensorToCvImage(std::get<3>(res));
              cv::imwrite(buffAsStdStr, cvImg);

              memset(buff, 0, 1024);
              snprintf(buff, sizeof(buff), "/DeepPool/samples/rank%d_iter%ld_index%ld_probability.jpg", rtctx->rank, iters, index);
              buffAsStdStr = buff;
              auto probability = result["pred"];
              auto disp_sprob = torch::unsqueeze((255.999*probability/probability.max()), -1).to(torch::kUInt8).contiguous().clone();
              cvImg = tensorToCvImage(disp_sprob);
              cv::imwrite(buffAsStdStr, cvImg);

#ifdef STATS_DEBUG
              auto rawimage = result["image"];
              rawimage -= rawimage.min();
              rawimage /= rawimage.max();
              if (rawimage.sizes()[2] == 1)
                  rawimage = torch::cat({rawimage.clone(), rawimage.clone(), rawimage.clone()}, 2);
              auto disp_raw_image = (255*rawimage).to(torch::kUInt8).contiguous().clone();
              // std::cout << disp_image << std::endl;
              auto rawtmp = tensorToCvImage(disp_raw_image);
              snprintf(buff, sizeof(buff), "/DeepPool/samples/rank%d_iter%ld_index%ld_rawimage.jpg", rtctx->rank, iters, index);
              buffAsStdStr = buff;    
              cv::imwrite(buffAsStdStr, rawtmp);

              for (int label_index=0; label_index<result["labels"].sizes()[2]; label_index++){
                if (label_index != 0){
                  auto lbls = result["labels"].index({torch::indexing::Slice(), torch::indexing::Slice(), label_index});
                  snprintf(buff, sizeof(buff), "/DeepPool/samples/rank%d_iter%ld_index%ld_%s_label.jpg", rtctx->rank, iters, index, label_list[label_index].c_str());
                  buffAsStdStr = buff;
                  // std::cout << lbls << std::endl;
                  lbls = torch::unsqueeze(lbls, -1);
                  std::cout << lbls.sizes() << std::endl;
                  if (lbls.sizes()[2] <= 1)
                      lbls = torch::cat({lbls.clone(), lbls.clone(), lbls.clone()}, 2);
                  std::cout << lbls.sizes() << std::endl;
                  auto disp_image = (255.999*lbls/lbls.max()).to(torch::kUInt8).contiguous().clone();
                  cv::imwrite(buffAsStdStr, tensorToCvImage(disp_image));


                  auto predtmp = result["pred"].index({torch::indexing::Slice(), torch::indexing::Slice(), label_index});
                  snprintf(buff, sizeof(buff), "/DeepPool/samples/rank%d_iter%ld_index%ld_%s_pred.jpg", rtctx->rank, iters, index, label_list[label_index].c_str());
                  buffAsStdStr = buff;
                  predtmp = torch::unsqueeze(predtmp, -1);
                  std::cout << pred.sizes() << std::endl;
                  if (predtmp.sizes()[2] <= 1)
                      predtmp = torch::cat({predtmp.clone(), predtmp.clone(), predtmp.clone()}, 2);
                  std::cout << predtmp.sizes() << std::endl;
                  auto disp_predtmp = (255.999*predtmp/predtmp.max()).to(torch::kUInt8).contiguous().clone();
                  cv::imwrite(buffAsStdStr, tensorToCvImage(disp_predtmp));


                }
              }
#endif
              // snprintf(buff, sizeof(buff), "/DeepPool/samples/rank%ld_iter%ld_index%ld_labels.jpg", rtctx->rank, iters, index);
              // std::string buffAsStdStr = buff;
          }

      }
    }
            
    printf("bi=%ld, i=%ld -> %dp - %dm - %df\n", bi, iters*64, totals.index({0}).item<int32_t>(), 
            totals.index({1}).item<int32_t>(), totals.index({2}).item<int32_t>());
    DP_LOG(DEBUG, "Evaluate iteration %lu/%lu\n", ++iters,
           eval_dataset_->GetItersPerEpoch());
    iters += 1;
  }
  
  // stats.close_chip();
  iters_before_graph_capture = 0;

  if (nr_gpus_ > 1) {
    rtctx->torch_stream.synchronize();  // sync before calling into NCCL
    commHandler->comm_start();
    commHandler->all_reduce(totals, c10d::ReduceOp::SUM);
    commHandler->comm_end();
    commHandler->sync();

    torch::Tensor resultsAsTensor = torch::zeros({SIZE_16_MiB}, torch::TensorOptions().dtype(torch::kUInt8)).contiguous();
    if(rtctx->rank != 0){
      uint64_t resultsSize = sizeof(ResultsMsgHeader)+(stats.getResults()->size()*sizeof(Result));
      assert(resultsSize < SIZE_16_MiB);

      ResultsMsgHeader msgHeader{stats.getResults()->size()};
      uint64_t offset = sizeof(ResultsMsgHeader);
      memcpy(resultsAsTensor.data_ptr(), (void *)&msgHeader, sizeof(ResultsMsgHeader));
      for(auto res : *stats.getResults()){
        memcpy(((uint8_t*)resultsAsTensor.data_ptr())+offset, (void *)&res, sizeof(Result));
        offset += sizeof(Result);
      }
    }

    if(rtctx->rank != 0){
      resultsAsTensor = resultsAsTensor.to(rtctx->c10dev);
      rtctx->torch_stream.synchronize();  // sync before calling into NCCL
      commHandler->comm_start();
      commHandler->send(resultsAsTensor, -1, 0);
      commHandler->comm_end();
      commHandler->sync();

      std::cout<<"currentResults size " << stats.getResults()->size() << std::endl;
    }
    else{
      for(uint64_t gpu_rank=1; gpu_rank < nr_gpus_; gpu_rank++){
        resultsAsTensor = torch::mul(resultsAsTensor, 0);
        resultsAsTensor = resultsAsTensor.to(rtctx->c10dev);
        rtctx->torch_stream.synchronize();  // sync before calling into NCCL
        commHandler->comm_start();
        commHandler->recv(resultsAsTensor, -1, gpu_rank);
        commHandler->comm_end();
        commHandler->sync();

        resultsAsTensor = resultsAsTensor.cpu().contiguous();
        std::vector<Result>* currentResults = stats.getResults();
        std::cout<<"currentResults size " << currentResults->size() << std::endl;
        ResultsMsgHeader* receivedMsgHeader = (ResultsMsgHeader*)resultsAsTensor.data_ptr();
        std::cout<<"receivedMsgHeader size " << receivedMsgHeader->numberOfResults << std::endl;
        uint64_t offset = sizeof(ResultsMsgHeader);
        for(uint64_t idx = 0; idx < receivedMsgHeader->numberOfResults; idx++){
          Result* newResult = (Result*)(((uint8_t*)resultsAsTensor.data_ptr())+offset);
          currentResults->push_back(*newResult);
          offset += sizeof(Result);
        }
        std::cout<<"currentResults size " << currentResults->size() << std::endl;
      }
      printf("final -> %dp - %dm - %df\n", totals.index({0}).item<int32_t>(), 
            totals.index({1}).item<int32_t>(), totals.index({2}).item<int32_t>());
    }
  }
  if(rtctx->rank == 0)
    stats.close_chip();

  double corr = correct.item().toDouble();

  DP_LOG(NOTICE, "Evaluate: Total: %.1f Correct: %.1f | Accuracy: %.3f", total,
         corr, static_cast<double>(corr) / total);
}
#else
void JobContext::Test(int64_t curEpoch) {
  double total = 0.0;
  torch::Tensor correct = torch::zeros({1}).to(at::kLong).to(rtctx->c10dev);

  eval_dataset_->Reset();

  if (iters_before_graph_capture < totiters && rtctx->use_fg_graph)
    iters_before_graph_capture = totiters + 5;

  size_t i = 0;
  while (!eval_dataset_->IsDone()) {
    total += model->GetGlobalBatchSize();

    auto batch = eval_dataset_->getNextThisRank();
    torch::Tensor input = batch["data"];
    if (input.defined()) input = input.to(rtctx->c10dev);
    auto output = Infer(input);
    if (output.defined() && output.nbytes() > 0) {
      auto pred = output.argmax(1);
      correct += pred.eq(batch["target"].to(rtctx->c10dev)).sum();
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
}
#endif

torch::Tensor JobContext::Infer(torch::Tensor input) {
  bool will_do_graph_capture = false;
  if(totiters == iters_before_graph_capture && !model->has_graph)
    will_do_graph_capture = true;

  torch::NoGradGuard guard;
  model->SetEval();
  model->SetInputsTargets(input);
  FinishIteration();

  if(will_do_graph_capture){
    //model doesnt actuall run when the graph is captured
    model->SetInputsTargets(input);
    FinishIteration();
  }
  return model->getOutput();
}

void JobContext::Train(torch::Tensor input, torch::Tensor target, torch::Tensor weights) {
  bool will_do_graph_capture = false;
  if(totiters == iters_before_graph_capture && !model->has_graph)
    will_do_graph_capture = true;

  model->SetTrain();
  model->SetInputsTargets(input, target, weights);
  FinishIteration();

  if(will_do_graph_capture){
    //model doesnt actuall run when the graph is captured
    model->SetInputsTargets(input, target, weights);
    FinishIteration();
  }
}

void JobContext::TrainOneEpoch(int64_t curEpochh) {
  dataset_pipeline_->Reset();
  model->ResetAvgLoss();
  size_t i = 0;
  if (iters_before_graph_capture < totiters && rtctx->use_fg_graph)
    iters_before_graph_capture = totiters + 5;
  while (!dataset_pipeline_->IsDone() && !job_done_) {
    auto batch = dataset_pipeline_->getNextThisRank();
    if(batch.find("data") == batch.end())
      continue;

    Train(batch["data"], batch["target"], batch["weight"]);
    DP_LOG(DEBUG, "Training iteration %lu/%lu\n", ++i,
           dataset_pipeline_->GetItersPerEpoch());
           
    if (nr_gpus_ > 1) {
      rtctx->torch_stream.synchronize();  // sync before calling into NCCL
      commHandler->comm_start();
      auto tmp_sync = torch::zeros({1}).to(rtctx->c10dev);;
      commHandler->all_reduce(tmp_sync, c10d::ReduceOp::SUM);
      commHandler->comm_end();
      commHandler->sync();
    }
  }
#ifdef ENABLE_STREAMING_DATASET
  model->lrsched->step();
#endif
  double loss = model->GetAvgLoss();
  DP_LOG(NOTICE, "Epoch done. Loss %.2f", loss);
  printf("Epoch %ld/%ld done. Loss %.2f\n", curEpochh, epochsToTrain,loss);
  iters_before_graph_capture = 0;
  rtctx->torch_stream.synchronize();
  end = std::chrono::steady_clock::now();
  be_img_end = GetBeCounter();
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

size_t JobContext::getTrainItersPerEpoch(){
  return train_dataset_->GetItersPerEpoch();
};