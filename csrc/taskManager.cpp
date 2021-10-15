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

#include "taskManager.h"
#include <torch/torch.h>
#include <memory>
#include <string>
#include "Cycles.h"
#include "utils.h"
#include "runnableModule.h"
#include "runtime.h"
#include "communication.h"
#include "logger.h"

#include "CUDASleep.h"

#include <cuda_profiler_api.h>

#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGraph.h>
#include <ATen/cuda/CUDAEvent.h>

using Cycles = RAMCloud::Cycles;

class CUDAPipeline {
 public:
  CUDAPipeline(size_t depth) : depth_(depth) {}
  void Lap() {
    if (cur_idx_++ % depth_ != 0) return;
    while (!ev_.query()) usleep(100);
    ev_ = at::cuda::CUDAEvent();
    ev_.record();
  }

 private:
  size_t depth_;
  size_t cur_idx_{0};
  at::cuda::CUDAEvent ev_;
};

static cudaGraphExec_t GraphSubset(std::set<cudaGraphNode_t> lnodes,
                                   cudaGraph_t graph) {
  std::vector<cudaGraphNode_t> nodes;
  size_t nr;
  CUDA_API_CALL(cudaGraphGetNodes(graph, nullptr, &nr));
  nodes.resize(nr);
  CUDA_API_CALL(cudaGraphGetNodes(graph, nodes.data(), &nr));

  cudaGraph_t gclone;
  CUDA_API_CALL(cudaGraphClone(&gclone, graph));
  for (auto& n : nodes) {
    if (lnodes.count(n)) continue;
    cudaGraphNode_t clnode;
    CUDA_API_CALL(cudaGraphNodeFindInClone(&clnode, n, gclone));
    CUDA_API_CALL(cudaGraphDestroyNode(clnode));
  }

  cudaGraphExec_t exec;
  CUDA_API_CALL(cudaGraphInstantiate(&exec, gclone, nullptr, nullptr, 0));
  CUDA_API_CALL(cudaGraphDestroy(gclone));

  return exec;
}

static float TimeGraphNode(std::set<cudaGraphNode_t> lnodes,
                           cudaGraph_t graph) {
  cudaGraphExec_t exec = GraphSubset(lnodes, graph);
  CUDA_API_CALL(cudaGraphUpload(exec, 0));

  cudaEvent_t begin, end;
  CUDA_API_CALL(cudaEventCreateWithFlags(&begin, cudaEventDefault));
  CUDA_API_CALL(cudaEventCreateWithFlags(&end, cudaEventDefault));
  CUDA_API_CALL(cudaDeviceSynchronize());
  gpu_nsleep(5000000, 0);
  CUDA_API_CALL(cudaEventRecord(begin));
  CUDA_API_CALL(cudaGraphLaunch(exec, 0));
  CUDA_API_CALL(cudaEventRecord(end));
  CUDA_API_CALL(cudaDeviceSynchronize());

  float ms;
  CUDA_API_CALL(cudaEventElapsedTime(&ms, begin, end));
  CUDA_API_CALL(cudaEventDestroy(begin));
  CUDA_API_CALL(cudaEventDestroy(end));

  CUDA_API_CALL(cudaGraphExecDestroy(exec));

  return ms;
}

std::vector<cudaGraphExec_t> GraphPartitioner(cudaGraph_t graph,
                                              float ms_piece_split) {
  assert(graph != nullptr);

  if (ms_piece_split <= 0) {
    cudaGraph_t clone;
    CUDA_API_CALL(cudaGraphClone(&clone, graph));
    cudaGraphExec_t gr;
    CUDA_API_CALL(cudaGraphInstantiate(&gr, clone, nullptr, nullptr, 0));
    CUDA_API_CALL(cudaGraphDestroy(clone));
    return {gr};
  }

  size_t nr;

  std::vector<cudaGraphNode_t> nodes;
  std::queue<cudaGraphNode_t> stack;
  std::set<cudaGraphNode_t> seen;

  assert(graph != nullptr);
  CUDA_API_CALL(cudaGraphGetRootNodes(graph, nullptr, &nr));
  nodes.resize(nr);
  CUDA_API_CALL(cudaGraphGetRootNodes(graph, nodes.data(), &nr));
  assert(nr == 1);

  std::vector<std::set<cudaGraphNode_t>> layers;
  std::vector<float> layers_time;

  std::set<cudaGraphNode_t> cur_layer;

  auto visit = [&](std::vector<cudaGraphNode_t>& nodes) {
    for (auto& p : nodes) {
      if (seen.count(p) != 0) continue;
      seen.insert(p);
      stack.push(p);
      cur_layer.insert(p);
    }
  };

  visit(nodes);
  float ms_tot = 0.0;

  while (stack.size() > 0) {
    while (stack.size() > 1) {
      auto node = stack.front();
      stack.pop();

      CUDA_API_CALL(cudaGraphNodeGetDependentNodes(node, nullptr, &nr));
      if (!nr) continue;
      nodes.resize(nr);
      CUDA_API_CALL(cudaGraphNodeGetDependentNodes(node, nodes.data(), &nr));
      visit(nodes);
    }

    if (stack.size() == 0)  // Handle?
      continue;

    assert(stack.size() == 1);
    layers.push_back(cur_layer);

    float layer_ms = TimeGraphNode(cur_layer, graph);
    layers_time.push_back(layer_ms);
    ms_tot += layer_ms;
    cur_layer.clear();

    auto node = stack.front();
    stack.pop();

    CUDA_API_CALL(cudaGraphNodeGetDependentNodes(node, nullptr, &nr));
    if (!nr) continue;
    nodes.resize(nr);
    CUDA_API_CALL(cudaGraphNodeGetDependentNodes(node, nodes.data(), &nr));
    visit(nodes);
  }

  assert(stack.size() == 0);
  assert(cur_layer.size() == 0);

  float cur_layer_ms = 0;

  cur_layer.clear();
  std::vector<cudaGraphExec_t> merged_layers;

  for (size_t i = 0; i < layers.size(); i++) {
    auto& l = layers.at(i);
    cur_layer.insert(l.begin(), l.end());
    cur_layer_ms += layers_time.at(i);

    if (cur_layer_ms >= ms_piece_split) {
      merged_layers.push_back(GraphSubset(cur_layer, graph));
      cur_layer.clear();
      cur_layer_ms = 0;
    }
  }
  merged_layers.push_back(GraphSubset(cur_layer, graph));

  return merged_layers;
}

class BeRunner {
public:
  void Lap() {
    while (status.load() != 0) {
      int s = 1;
      if (status.load() != 2) status.compare_exchange_strong(s, 2);
      usleep(100);
    }
  }
  void Pause() {
    auto stat = status.load();
    if (stat == 2) return;
    assert(stat == 0);
    status.store(1);
    while (status.load() != 2) usleep(100);
  }
  void Resume() {
    status.store(0);
  }
private:
  std::atomic<int> status{2};
};


/**
 * Contructs context for a training job.
 */
JobContext::JobContext(std::unique_ptr<RunnableModule> modelIn, std::string name,
    std::unique_ptr<DataLoader> dataLoader,
    std::unique_ptr<CommunicationHandler> commHandler,
    std::unique_ptr<TargetShuffler> targetShuffler,
    c10::Device device,
    int epochsToTrain,
    std::unique_ptr<torch::optim::Optimizer> optimizer)
  : model(std::move(modelIn))
  , name(name)
  , dataLoader(std::move(dataLoader))
  , commHandler(std::move(commHandler))
  , targetShuffler(std::move(targetShuffler))
  , epochsToTrain(epochsToTrain)
  , optimizer(std::move(optimizer))
  , device(device)
  , epoch(0)
  , iter(0)
  , itersToTrain(500) // = len(dataLoader) if dataLoader != None else None #TODO: this is a temporary hack..
  , state(JobState::INIT)
  , timers()
{
  if (rtctx->use_fg_graph) {
    iters_before_graph_capture = 50;
  } else {
    iters_before_graph_capture = 5000;
  }

  timers.reserve(CT_NUM_OF_EVENTS);
  timers.emplace_back();
  CudaTimer* startTimer = &timers.back();
  CudaTimer* lastTimer = startTimer;
  for (int i = 1; i < CT_NUM_OF_EVENTS - 1; ++i) {
    timers.emplace_back(lastTimer);
    lastTimer = &timers.back();
  }
  timers.emplace_back(startTimer); // CT_STOP measures from CT_START to CT_STOP;

  // Initialize timers.
  model->initProfileTimers(&timers[CT_LOAD], &timers[CT_LOSS]);
}

/**
 * Destructs context for a training job.
 * Defined here to avoid incomplete type destruction in 'taskManager.h' of
 * some unique_ptr members.
 */
JobContext::~JobContext() {}

#include <condition_variable>
#include <mutex>

std::mutex mtx;
std::condition_variable cv;
bool beinited = false;

static std::atomic<uint64_t> fgcounter{0};
static std::atomic<uint64_t> becounter{0};
static BeRunner be_controller;
static long be_bsize = 0;

/* tremendous WIP */
void BeRunner(long bsize) {
  be_bsize = bsize;
  bool use_graph_partitioner = rtctx->be_graph_split_ms > 0.0;
  int samplePerKernel = rtctx->samplePerKernel;
  assert(bsize % samplePerKernel == 0);
  long splitways = bsize / samplePerKernel;
  assert(!use_graph_partitioner || splitways == 1);
  assert(!use_graph_partitioner || rtctx->use_be_graph);

  torch::jit::script::Module m = torch::jit::load(rtctx->be_jit_file);
  m.train();
  m.to(rtctx->c10dev);

  std::vector<torch::Tensor> params;
  for (const auto &p : m.parameters()) params.push_back(p);

  torch::optim::SGD optim(params, torch::optim::SGDOptions(0.1).momentum(0.9));

  long px = rtctx->be_jit_file.find("inception") == std::string::npos ? 224 : 299;
  auto tensor = torch::rand({bsize, 3, px, px}).to(rtctx->c10dev);

  std::vector<int64_t> splitSizes(splitways, bsize / splitways);
  std::cerr << "split: " << splitSizes << std::endl;
  auto tenss = tensor.split_with_sizes(splitSizes);
  std::vector<c10::cuda::CUDAStream> streams;
  for (size_t i = 0; i < tenss.size(); i++) streams.push_back(c10::cuda::getStreamFromPool(false));
  auto target =
        torch::empty(bsize).uniform_(0, 1000).to(at::kLong).to(rtctx->c10dev);
  auto targs = target.split_with_sizes(splitSizes);

  at::autocast::set_enabled(true);

  assert(static_cast<size_t>(splitways) == tenss.size());
  auto fn = [&] {
    auto orig_stream = c10::cuda::getCurrentCUDAStream();
    optim.zero_grad();
    at::cuda::CUDAEvent ev;
    ev.record(orig_stream);
    for (size_t i = 0; i < tenss.size(); i++) {
      auto &st = streams.at(i);
      if (splitways > 1) {
        c10::cuda::setCurrentCUDAStream(st);
        ev.block(st);
      }
      auto ret = m.operator()({tenss.at(i)});
      auto loss = torch::nll_loss(ret.toTensor().log_softmax(1), targs.at(i));
      loss.backward();
      if (splitways > 1) {
        at::cuda::CUDAEvent ev2;
        ev2.record(st);
        ev2.block(orig_stream);
      }
    }

    c10::cuda::setCurrentCUDAStream(orig_stream);
    optim.step();

    at::autocast::clear_cache();
  };

  auto cstream = c10::cuda::getStreamFromPool(false);
  c10::cuda::setCurrentCUDAStream(cstream);

  for (size_t i = 0; i < 50; i++) fn();
  at::cuda::CUDAGraph graph;
  c10::cuda::device_synchronize();
  graph.capture_begin();
  fn();
  graph.capture_end();
  c10::cuda::device_synchronize();
  {
    std::lock_guard<std::mutex> lk(mtx);
    beinited = true;
  }

  CUDAPipeline p(1);

  if (use_graph_partitioner) {
    auto parts = GraphPartitioner(graph.getGRAPH(), rtctx->be_graph_split_ms);
    be_controller.Resume();
    cv.notify_one();
    while (true) {
      for (auto &pp : parts) {
        be_controller.Lap();
        p.Lap();
        CUDACHECK(cudaGraphLaunch(pp, cstream));
      }
      becounter.store(becounter.load() + bsize);
    }
  }

  be_controller.Resume();
  cv.notify_one();

  while (true) {
    be_controller.Lap();
    p.Lap();
    if (rtctx->use_be_graph)
      graph.replay();
    else
      fn();
    becounter.store(becounter.load() + bsize);
  }
}


/**
 * Constructs a TaskManager.
 */
TaskManager::TaskManager(RuntimeContext* rtctx)
  : rtctx(rtctx)
  , _mutex()
  , jobList()
{
  rtctx->taskManager = this;
  if (rtctx->be_batch_size > 0) {
    long bsize = rtctx->be_batch_size;
    std::thread([=] { BeRunner(bsize); }).detach();
    std::unique_lock<std::mutex> lk(mtx);
    cv.wait(lk, []{return beinited;});
  }

  std::thread([&] {
    using namespace std::chrono;
    size_t lastc = becounter.load();
    size_t lastfg = fgcounter.load();
    auto lastt = steady_clock::now();
    while (true) {
      sleep(1);
      size_t newtr = becounter.load();
      size_t newfg = fgcounter.load();
      auto now = steady_clock::now();
      auto s = duration_cast<seconds>(now - lastt).count();
      std::cerr << "BE im/s: " << (newtr - lastc) / s << " FG iter/s: " << (newfg - lastfg) / s << std::endl;
      lastt = now;
      lastc = newtr;
      lastfg = newfg;
    }
  }).detach();

}

/**
 * Adds a new training job submitted by coordinator.
 * 
 * \return  The number of jobs currently scheduled.
 */
int
TaskManager::addTrainingJob(std::unique_ptr<JobContext> job)
{
  std::lock_guard<std::mutex> lock(_mutex);
  jobList.push_back(std::move(job));
  DP_LOG(LogLevel::NOTICE, "Added a new job. %s", jobList.back()->name.c_str());
  return jobList.size();
}

/**
 * A poller to make a progress on training tasks.
 *
 * \return The number of jobs that are executed (or scheduled to CUDA).
 */
int
TaskManager::poll()
{
  std::lock_guard<std::mutex> lock(_mutex);
  // Cycles::sleep(1000000);
  if (jobList.empty()) {
    return 0;
  }

  int jobsScheduled = 0;
  JobContext* mainJob = jobList[0].get();
  JobContext* subJob = nullptr;
  if (jobList.size() > 1) {
    subJob = jobList[1].get();
  }
  bool jobCompleted = false;
  trainSingleStep(mainJob, &jobCompleted);
  // int idleUs = 0, spentUs = 0;
  // trainSingleStep(mainJob, &jobCompleted, &idleUs, &spentUs);
  // if (mainJob.totiters > 41 && idleTime > 0) {
  //   if (subJob) {
  //     int idleUs2 = 0, spentUs2 = 0;
  //     trainSingleStep(subJob, &jobCompleted, &idleUs2, &spentUs2);
  //   }
  // }


  if (jobCompleted) {
    size_t warmupIters = 100;
    // mainJob->model->printProfileTimers(warmupIters);
    mainJob->model->printLayerInGraphTimes();
    size_t totiters = mainJob->totiters - warmupIters;
    using msec = std::chrono::duration<double, std::milli>;
    double elapsed_ms = std::chrono::duration_cast<msec>(mainJob->end - mainJob->start).count();
    double total_iter_ms = elapsed_ms / (double)totiters;
    double total_iter_ps = 1e3 / total_iter_ms;
    double be_img_ps = mainJob->be_img_end - mainJob->be_img_start;
    be_img_ps = 1e3 * be_img_ps / elapsed_ms;
    DP_LOG(NOTICE, "A training job %s is completed (%lu iters, %.2f ms/iter, %.2f iter/s, %.2f be img/s)."
        " AverageTiming (ms) => zero: %.1f, load:%.1f, fp:%.1f, loss:%.1f, bp:%.1f, opt: %.1f, iter:%.1f"
        " P50 (ms) => fp:%.1f, loss:%.1f, bp:%.1f, iter:%.1f",
        mainJob->name.c_str(), totiters, total_iter_ms, total_iter_ps, be_img_ps,
        mainJob->timers[CT_ZERO].getAvg(warmupIters),
        mainJob->timers[CT_LOAD].getAvg(warmupIters),
        mainJob->timers[CT_FP].getAvg(warmupIters),
        mainJob->timers[CT_LOSS].getAvg(warmupIters),
        mainJob->timers[CT_BP].getAvg(warmupIters),
        mainJob->timers[CT_OPT].getAvg(warmupIters),
        mainJob->timers[CT_STOP].getAvg(warmupIters),
        mainJob->timers[CT_FP].getP50(warmupIters),
        mainJob->timers[CT_LOSS].getP50(warmupIters),
        mainJob->timers[CT_BP].getP50(warmupIters),
        mainJob->timers[CT_STOP].getP50(warmupIters));

    // DP_LOG(NOTICE, " -- detachTime: %" PRIu64" us", mainJob->model->detachTimer.avgMicros());

    jobList.erase(jobList.begin());
    DP_LOG(NOTICE, "Removed the completed job. Remaining: %lu", jobList.size());
  }
  jobsScheduled++;
  return jobsScheduled;
}

/**
 * A helper to run a job.
 * 
 * \param job   a context for the job to train.
 * \param[out] jobCompleted 
 *    will be filled with true if the job is completely finished.
 * 
 * \return    returns non-zero if it actively worked.
 */
int
TaskManager::trainSingleStep(JobContext* job, bool* jobCompleted)
{
  if (job->state == JobState::INIT) {

    if (be_bsize > 0 && job->totiters == 0) {
      if (!job->run_with_be) {
        be_controller.Pause();
      } else {
        be_controller.Resume();
      }
    }

    if (job->totiters == job->profile_iter_start)
      CUDA_API_CALL(cudaProfilerStart());

    /* record starting point for BE training */
    if (job->totiters == 100) {
      rtctx->torch_stream.synchronize();
      job->be_img_start = becounter.load();
      job->start = std::chrono::steady_clock::now();
    }

    if (job->iter >= job->itersToTrain) {
      DP_LOG(DEBUG, "epoch is completed.");
      job->iter = 0;
      job->epoch++;
    }
    if (job->epoch >= job->epochsToTrain || 
        (rtctx->profile && job->totiters == job->iters_before_graph_capture)) {
      DP_LOG(DEBUG, "training is completed.");
      rtctx->torch_stream.synchronize();
      job->end = std::chrono::steady_clock::now();
      job->be_img_end = becounter.load();
      *jobCompleted = true;
      return 0;
    }

    job->model->resetProfileTimers();
    for (int tpIdx = CT_NUM_OF_EVENTS - 1; tpIdx >= CT_START; --tpIdx) {
      DP_LOG(DEBUG, "timer.saveAndReset() for %d. recorded:%d", tpIdx, job->timers[tpIdx].isRecorded());
      job->timers[tpIdx].saveAndReset();
    }
    job->timers[CT_START].record();
    DP_LOG(DEBUG, "JobState::INIT.");

    job->model->iterInit();

    /* start graph capture */
    if (job->totiters == job->iters_before_graph_capture) {
      if (job->run_with_be && be_bsize > 0) be_controller.Pause();
      c10::cuda::device_synchronize();
      DP_LOG(NOTICE, "Starting capture.");
      job->model->graph_mempool = at::cuda::graph_pool_handle();
      job->model->maingraph.capture_begin(job->model->graph_mempool);
      job->commHandler->precapture();
    } else if (job->totiters > job->iters_before_graph_capture) {
      /* skip to forward phase */
      job->state = JobState::FORWARD;
      return 1;
    }

    job->optimizer->zero_grad();
    job->timers[CT_ZERO].record();

    job->timers[CT_LOAD].record();
    job->state = JobState::FORWARD;
    DP_LOG(DEBUG, "Foward pass is starting soon.");
  } else if (job->state == JobState::FORWARD) {
    DP_LOG(DEBUG, "JobState::FORWARD.");
    if (job->totiters > job->iters_before_graph_capture) {
      DP_LOG(DEBUG, "Replay iter.");

      static CUDAPipeline p(8);
      p.Lap();
      if (job->model->maingraph_parts.size() > 0) {
        for (auto& p : job->model->maingraph_parts)
          CUDACHECK(cudaGraphLaunch(p, rtctx->torch_stream));
      } else {
        job->model->maingraph.replay();
      }

      job->model->syncgraph.replay();
      if (job->model->stepgraph_parts.size() > 0) {
        for (auto& p : job->model->stepgraph_parts)
          CUDACHECK(cudaGraphLaunch(p, rtctx->torch_stream));
      } else {
        job->model->stepgraph.replay();
      }

      job->state = JobState::FINISH;
      return 1;
    }

    bool capture = rtctx->profile && job->totiters == job->iters_before_graph_capture - 1;
    JobStatus status = job->model->forwardAStep(capture);

    if (status == COMPLETED) {
      job->timers[CT_FP].record();
      // TODO: add a loss calculation here? or as another state?
      DP_LOG(DEBUG, "Foward pass is completed. Calculating loss.");
      
      job->model->loss();
      job->timers[CT_LOSS].record();
      assert(job->model->layerQ.empty());
      job->model->layerQ.push_back(&job->model->layers.back());
      DP_LOG(DEBUG, "Moving to backward pass.");
      job->state = JobState::BACKWARD;
    }
  } else if (job->state == JobState::BACKWARD) {
    DP_LOG(DEBUG, "JobState::BACKWARD.");
    
    bool capture = rtctx->profile && job->totiters == job->iters_before_graph_capture - 1;
    JobStatus status = job->model->backwardAStep(capture);
    // TODO: get idle time for backward separately.
    
    if (status == COMPLETED) {
      job->timers[CT_BP].record();
      job->state = JobState::SYNC;
      DP_LOG(DEBUG, "Backward pass is completed. Moving to gradient all-reduce.");
    }
  } else if (job->state == JobState::SYNC) {
    DP_LOG(DEBUG, "JobState::SYNC.");

    if (job->totiters == job->iters_before_graph_capture) {
      job->commHandler->postcapture();
      job->model->maingraph.capture_end();
      job->model->syncgraph.capture_begin(job->model->graph_mempool);
      job->model->gradientSync();
      job->model->syncgraph.capture_end();
      job->model->stepgraph.capture_begin(job->model->graph_mempool);
    } else {
      job->model->gradientSync();
    }
    job->timers[CT_SYNC].record();
    job->state = JobState::STEP;
  } else if (job->state == JobState::STEP) {
    DP_LOG(DEBUG, "JobState::STEP");
    job->optimizer->step();
    job->timers[CT_OPT].record();
    job->state = JobState::FINISH;
  } else if (job->state == JobState::FINISH) {
    DP_LOG(DEBUG, "JobState::FINISH");

    if (job->totiters == job->profile_iter_start + job->niter_to_profile)
      CUDA_API_CALL(cudaProfilerStop());

    if (job->totiters == job->iters_before_graph_capture) {
      job->model->stepgraph.capture_end();

      // job->model->maingraph_parts = GraphPartitioner(job->model->maingraph.getGRAPH(), 10.0);
      // job->model->stepgraph_parts =  GraphPartitioner(job->model->stepgraph.getGRAPH(), 2.0);

      if (job->run_with_be && be_bsize > 0) be_controller.Resume();
      DP_LOG(NOTICE, "Ending capture.");
    }
    job->totiters++;
    job->iter++;
    fgcounter++;

    job->state = JobState::INIT;
    job->timers[CT_STOP].record();
  }
  return 1;
}
