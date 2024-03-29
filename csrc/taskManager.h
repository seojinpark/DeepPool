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
#include <mutex>
#include <vector>
#include <string>
#include "tracer.h"

/**
 * Forward declarations. Do not include headers unless necessary.
 */
class RuntimeContext;
class RunnableModule;
namespace torch {
  namespace optim {
    class Optimizer;
  }
}
// namespace c10 {
//   class Device;
// }
class CommunicationHandler;

class DataLoader {
};

class TargetShuffler {
};

enum class JobState {
  INIT = 0,
  FORWARD,
  BACKWARD,
  SYNC,
  STEP,
  FINISH,
  NUM_JOB_STATES // must be the last element in the enum
};

enum TracePoint {
  CT_START = 0,
  CT_ZERO,
  CT_LOAD,
  CT_FP,
  CT_LOSS,
  CT_BP,
  CT_SYNC,
  CT_OPT,
  CT_STOP,
  CT_NUM_OF_EVENTS // CT_NUM must be the last element.
};

struct IdleTimeCtx {
  enum Type {
    FG,
    BG
  };
  int64_t remainingIdleUsec {0};
  int64_t* idleUsecOfMainPtr {nullptr}; // Used only for Subjob.
  Type jobType {FG}; // 1: MainJob, 2: SubJob
  void processLayerTime(int usec, bool isActive) {
    if (jobType == FG) {
      if (isActive) {
        remainingIdleUsec = 0;
      } else {
        remainingIdleUsec += usec;
      }
    } else if (jobType == BG) {
      if (isActive) {
        *idleUsecOfMainPtr -= usec;
      }
    }
  }
};

/**
 * Context holding data for each training task.
 */
struct JobContext {
 public:
  JobContext(std::unique_ptr<RunnableModule> model, std::string name,
      std::unique_ptr<DataLoader> dataLoader,
      std::unique_ptr<CommunicationHandler> commHandler,
      std::unique_ptr<TargetShuffler> targetShuffler,
      c10::Device device,
      int epochsToTrain = 1,
      std::unique_ptr<torch::optim::Optimizer> optimizer = nullptr
      // std::unique_ptr<torch::optim::SGD> optimizer = nullptr
      );
  ~JobContext();

  std::unique_ptr<RunnableModule> model;
  std::string name;
  std::unique_ptr<DataLoader> dataLoader;
  std::unique_ptr<CommunicationHandler> commHandler;
  std::unique_ptr<TargetShuffler> targetShuffler;
  size_t epochsToTrain;
  std::unique_ptr<torch::optim::Optimizer> optimizer;
  c10::Device device;
  // self.dataLoaderIt = iter(self.dataLoader) if dataLoader != None else None
  // self.criterion = nn.CrossEntropyLoss().cuda(device) if criterion == None else criterion
  size_t epoch;
  size_t iter;
  size_t totiters{0}; // total iters executed
  size_t itersToTrain; // = len(dataLoader) if dataLoader != None else None #TODO: this is a temporary hack..
  // self.itersPerPoll = 50
  // self.training_initialized = False
  // self.itersToCapture = set(range(250, 260))
  size_t profile_iter_start{5000};
  size_t niter_to_profile{5};
  size_t iters_before_graph_capture{50}; // set high to disable graph capture
  bool run_with_be{false};
  JobState state;

  std::vector<CudaTimer> timers;

  std::chrono::time_point<std::chrono::steady_clock> start, end;
  uint64_t be_img_start, be_img_end;
  IdleTimeCtx idleCtx;
};

/**
 * Manages training jobs by scheduling tasks to CUDA devices.
 * 
 * Public methods are thread-safe.
 */
class TaskManager {
 public:
  TaskManager(RuntimeContext* rtctx);
  ~TaskManager() {} // TODO(seojin): implement
  int addTrainingJob(std::unique_ptr<JobContext> job);
  int addBgJob();
  int poll();
 private:
  bool trainAllTheWayWithBg(JobContext* mainJob);
  int trainSingleStep(JobContext* job, bool* jobCompleted);
  int64_t getNextStepTime(JobContext* job);
  void printJobStatistics(JobContext* job);

  RuntimeContext* rtctx;
  std::mutex _mutex;                // Monitor lock for TaskManager.
  std::vector< std::unique_ptr<JobContext> > jobList;  
                                    // Holds unfinished training jobs.
                                    // Assumes jobs are ordered by priority.
  std::unique_ptr<JobContext> bgJob {};
};

#endif // TASK_MANAGER_H