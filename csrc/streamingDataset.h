#pragma once

#include <torch/torch.h>
#include <torch/types.h>
#include <pthread.h>
#include "dataset.h"

struct sharedBuffers { 
    bool ready = false;
    int stopWorker = 0;
    int sock = -1;
    int64_t datasetSize;
    int64_t index;

    // uint64_t numH5Files;
    // uint64_t h5FileIdx;
    // char h5FileNamesBuffer[2048] = {0};

    // size_t iidVecStrLen;
    // size_t iidVecLen;
    // void* iidVec;

    uint64_t weightsNDims;
    uint64_t weightsDims[8] = {0};
    size_t weightsLen;
    void* weights;

    uint64_t imagesNDims;
    uint64_t imagesDims[8] = {0};
    size_t imagesLen;
    void* images;

    uint64_t labelsNDims;
    uint64_t labelsDims[8] = {0};
    size_t labelsLen;
    void* labels;

    char mutexID[64] = {0};
    pthread_mutex_t* mp_mutex;
    pthread_mutexattr_t mutexAttr;

    // munmap(allocation, 5000);
    ~sharedBuffers();
};

struct batchPath{ 
    int64_t index;
    std::string path;

    batchPath& operator=(const batchPath other)
    {
        index = other.index;
        path = other.path;
        return *this;
    }
    bool operator()(batchPath const& left, batchPath const& right) {
        return (left.index) > (right.index);
    }

    bool operator==(const batchPath& rhs) const {
        return index == rhs.index;
    }

    bool operator<(const batchPath& rhs) const {
        return index < rhs.index;
    }

    bool operator>(const batchPath& rhs) const {
        return index > rhs.index;
    }
};

struct batchData { 
    int64_t datasetSize;
    int64_t index;
    // std::vector<std::string> iidVec;
    torch::Tensor weights;
    torch::Tensor images;
    torch::Tensor labels;

    batchData& operator=(const batchData other)
    {
        datasetSize = other.datasetSize;
        index = other.index;
        weights = other.weights;
        images = other.images;
        labels = other.labels;
        return *this;
    }

    bool operator()(batchData const& left, batchData const& right)
    {
        return (left.index) > (right.index);
    }

    bool operator<(const batchData& rhs) const
    {
        return index < rhs.index;
    }

    bool operator>(const batchData& rhs) const
    {
        return index > rhs.index;
    }

    // bool operator< (batchData left, batchData right) { return (left.index) > (right.index); };
};

// class Dataset {
//  public:
//   virtual torch::data::Example<> getNext() = 0;
//   virtual size_t GetItersPerEpoch() = 0;
//   virtual bool IsDone() = 0;
//   virtual void Reset() = 0;
//   static Dataset *fromName(std::string name, size_t rank, long globalBatchSize,
//                            std::vector<long> initialBatchSizes,
//                            std::vector<long> sampleIndices,
//                            size_t fake_train_iters_per_epoch);

//   torch::data::Example<> getNextThisRank() {
//     auto ex = getNext();

//     torch::Tensor data, target;
//     if (initialBatchSizes_.at(rank_))
//       data = ex.data.split_with_sizes(initialBatchSizes_)[rank_];

//     if (sampleIndices_.size()) {
//       std::vector<long> spl(globalBatchSize_, 1);
//       auto splitsamples =
//           ex.target.split_with_sizes(spl);  // TODO make this clean....
//       std::vector<torch::Tensor> samplesOrdered;
//       for (auto &s : sampleIndices_)
//         samplesOrdered.push_back(splitsamples.at(s));
//       target = torch::cat(samplesOrdered);
//     }
//     return {data, target};
//   }
//  protected:
//   long globalBatchSize_;
//   Dataset(size_t rank, long globalBatchSize,
//           std::vector<long> initialBatchSizes, std::vector<long> sampleIndices)
//       : globalBatchSize_(globalBatchSize),
//         rank_(rank),
//         initialBatchSizes_(initialBatchSizes),
//         sampleIndices_(sampleIndices){};

//  private:
//   size_t rank_;
//   std::vector<long> initialBatchSizes_;
//   std::vector<long> sampleIndices_;

struct StreamingDataset : public Dataset
{
    private:
        bool is_eval_;
        bool is_validation_;
        size_t worldSize_ = 0;
        int64_t stopDataset_ = 0;
        bool startedDataset_ = false;
        int64_t epochLen_;
        int64_t epochCount_ = 0;
        int64_t datasetSize_;
        int64_t batchSize_;
        int64_t lastBatchIdx = -1;
        std::vector<std::thread> threads;

        // Using lambda to compare elements.

        // auto cmp = [](batchData left, batchData right) { return (left.index) > (right.index); };
        std::priority_queue<batchData, std::vector<batchData>, std::greater<batchData>> readyBatches;
        // std::deque<batchData> readyBatches;
        // torch::Tensor states_, labels_;

        std::mutex local_mutex_;
        std::mutex local_get_file_mutex_;
        char mutexID_[64] = {0};
        pthread_mutex_t* mp_mutex_;
        pthread_mutexattr_t mutexAttr_;

        int counter_ = 0;
        bool done_ = false;
        // pthread_mutex_t* mp_mutex;
        // pthread_mutexattr_t mutexAttr;
        int64_t workerID_;
        int64_t bufferIndex_;
        uint64_t numWorkers_ = 2;
        std::vector<sharedBuffers*> sharedWorkerBuffs_;
        bool moved_from_ = false;

    public:
        StreamingDataset(size_t rank, long globalBatchSize,
                           std::vector<long> initialBatchSizes,
                           std::vector<long> sampleIndices, bool is_eval,
                           size_t worldSize = 1, bool is_validation_ = false);
        ~StreamingDataset();


        // bool batchComp(batchData left, batchData right) { return (left.index) > (right.index); };

        // StreamingDataset(StreamingDataset const &) = delete;
        // StreamingDataset& operator=(StreamingDataset const &) = delete;

        // StreamingDataset(StreamingDataset &&moveable) noexcept :
        //     Dataset(moveable.rank_, moveable.globalBatchSize_, moveable.initialBatchSizes_, moveable.sampleIndices_){
        //     std::cout << "moving - StreamingDataset\n";
        //     moved_from_ = false;
        //     moveable.moved_from_ = true;
        //     // And now we spell out the explicit default move constructor
        // }

        // StreamingDataset& operator=(StreamingDataset &&moveable) noexcept {
        //     std::cout << "moving oper - StreamingDataset\n";
        //     moved_from_ = false;
        //     moveable.moved_from_ = true;
        //     // And now we spell out the explicit default move assignment operator
        //     return *this;
        // }




        torch::optional<batchData> read_batch();
        torch::optional<batchData> read_batch_worker();
        void worker_RunMain();
        void worker_BlobToTensorsThread(int wID);
        int64_t init();
        int64_t test();
        // torch::optional<batchData> get_batch(void);
        // torch::optional<batchData> 
        std::map<std::string, at::Tensor> getNextThisRank() override;
        // torch::optional<size_t> size();
        void Reset(); // {counter_ = 0; done_=false;};
        bool IsDone(){return done_;};
        size_t GetItersPerEpoch() override;
        std::map<std::string, torch::Tensor> getNext() override;
};