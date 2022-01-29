#pragma once

#include <torch/torch.h>
#include <torch/types.h>
#include <pthread.h>

struct sharedBuffers { 
    bool ready = false;
    int stopWorker = 0;
    int sock = -1;
    int64_t batchSize;
    int64_t index;

    // uint64_t numH5Files;
    // uint64_t h5FileIdx;
    // char h5FileNamesBuffer[2048] = {0};

    size_t iidVecStrLen;
    size_t iidVecLen;
    void* iidVec;

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

    char mutexID[32] = {0};
    pthread_mutex_t* mp_mutex;
    pthread_mutexattr_t mutexAttr;
};

struct batchData { 
    int64_t len;
    int64_t index;
    // std::vector<std::string> iidVec;
    torch::Tensor weights;
    torch::Tensor images;
    torch::Tensor labels;

    batchData& operator=(const batchData other)
    {
        // std::cout << "copy assignment of batchData\n";
        len = other.len;
        index = other.index;
        // iidVec = other.iidVec;
        weights = other.weights;
        images = other.images;
        labels = other.labels;
        // std::swap(n, other.n);
        // std::swap(s1, other.s1);
        return *this;
    }
};

// const int kNumberOfExamplesAfterWhichTheDatasetExhausts = 10;
// const int kNumberOfWorkers = 4;

struct StreamingDataset : public torch::data::datasets::StatefulDataset<StreamingDataset, batchData, size_t>
{
    // public:
        int rank = 0;
        int worldSize = 0;
        int stopDataset = 0;
        int64_t epochLen;
        int64_t batchSize;
        std::vector<std::thread> threads;
        std::deque<batchData> readyBatches;
        // torch::Tensor states_, labels_;


        std::mutex local_mutex;
        std::mutex local_get_file_mutex;
        char mutexID[32] = {0};
        pthread_mutex_t* mp_mutex;
        pthread_mutexattr_t mutexAttr;



        int counter = 0;
        // pthread_mutex_t* mp_mutex;
        // pthread_mutexattr_t mutexAttr;
        int64_t workerID;
        int64_t bufferIndex;
        uint64_t numWorkers = 4;
        std::vector<sharedBuffers*> sharedWorkerBuffs;

    // public:
        // bool is_stateful = false;
        explicit StreamingDataset(int rank, int worldSize);
        ~StreamingDataset(void);
        torch::optional<batchData> read_batch(void);
        torch::optional<batchData> read_batch_worker(void);
        void workerRunMain(void);
        void workerToTensorsThread(int wID);
        int64_t init(void);
        int64_t test(void);
        torch::optional<batchData> get_batch(size_t) override;
        torch::optional<size_t> size() const override;
        void reset() override {counter = 0;};
        void save(torch::serialize::OutputArchive& archive) const override{((void)archive);};
        void load(torch::serialize::InputArchive& archive) override {((void)archive);};
        // torch::Tensor read_data(const std::string& loc);
        // torch::data::Example<> get(size_t index);
        // torch::data::Example<> get(size_t index) ;//override;
        // explicit StreamingDataset(const std::string& loc_states, const std::string& loc_labels) 
        //     : states_(read_data(loc_states)),
        //       labels_(read_data(loc_labels)) {   };


        //   explicit RandomDataset(Mode mode = Mode::kTrain);
        // {  this->batchSize = batchSize; std::cout << "Hello World!\n";  };
        // torch::data::AnvilExample<> get(size_t index) override;
        // torch::data::Example<> get(size_t index) override;
        //  {
        //     std::lock_guard<std::mutex> lock(mutex);
        //     if (counter < kNumberOfExamplesAfterWhichTheDatasetExhausts) {
        //         return torch::nullopt;
        //     }
        //     return torch::nullopt;
        // }

	    /// Returns the size of the dataset.
	    // torch::optional<size_t> size() const override;
        // batchData* begin();// { batchData newbatch; auto tmp = StreamingDataset::read_batch(&newbatch); return newbatch; }
        // batchData* end();// { return iterator(val + len); }


};