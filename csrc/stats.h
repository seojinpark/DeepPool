#pragma once

#include <torch/torch.h>
#include <torch/types.h>

// #define STATS_DEBUG

using tuple2int = std::tuple<int64_t, int64_t>;
using tuple4int = std::tuple<int64_t, int64_t, int64_t, int64_t>;
using tuple2i1d1i = std::tuple<int64_t, int64_t, double, int64_t>;
using tuple3double = std::tuple<double, double, double>;
using tuple4double = std::tuple<double, double, double, double>;

struct sclass {
    int64_t count_;
    std::vector<int> detects_;
};

typedef struct{
    uint64_t numberOfResults;
} ResultsMsgHeader;

typedef struct{
    double val;
    bool truth;
} Result;

struct Stats
{
    private:
        std::vector<std::string> labels_;
        std::string background_label_;
        int64_t num_frames_;
        double threshold_;
        double display_threshold_;
        std::string mode_ = std::string("single");
        double spacing_ = 0;
        sclass sclass_;
        std::vector<Result> results_;
        std::map<std::string, sclass> pr_;
        int64_t background_index_ = -1;

    public:
    // labels, background_label, num_frames, threshold=None, display_threshold=None, mode='single', spacing=None):
        Stats(std::vector<std::string> labels, std::string background_label,
                           int64_t num_frames, double threshold=0, double display_threshold=0,
                           std::string mode = std::string("single"), double spacing = 0);
        ~Stats();

        // std::map<std::string, torch::Tensor> batch(std::map<std::string, torch::Tensor> results);
        // std::map<std::string, torch::Tensor> batch_chip(std::map<std::string, torch::Tensor> result);
        std::tuple<int64_t, int64_t, int64_t, torch::Tensor, std::vector<tuple4double>>
            batch(std::map<std::string, torch::Tensor> results);
        std::tuple<int64_t, int64_t, int64_t, torch::Tensor, std::vector<tuple4double>>
            batch_chip(std::map<std::string, torch::Tensor> result);
        torch::Tensor find_peaks(torch::Tensor image_in, c10::IntArrayRef shape, double threshold);
        std::vector<int64_t> unravel_index(int64_t index, c10::IntArrayRef shape);
        void insert_box(torch::Tensor* image, int64_t x_cen, int64_t y_cen, int64_t radius, std::string color, int64_t thickness=2);
        std::tuple<std::vector<double>, std::vector<double>> _build_roc_curve();
        double _area_under_roc(std::vector<double> det, std::vector<double> fa);
        void _plot_roc_curve(std::string filename, std::vector<double> det, std::vector<double> fa, double auc);
        void close_chip();

        // uint64_t getResultsSizeInBytes();
        std::vector<Result>* getResults() {return &results_;};

        // torch::optional<batchData> read_batch();
        // torch::optional<batchData> read_batch_worker();
        // void worker_RunMain();
        // void worker_BlobToTensorsThread(int wID);
        // int64_t init();
        // int64_t test();
        // std::map<std::string, at::Tensor> getNextThisRank() override;
        // void Reset() {counter_ = 0; done_=false;};
        // bool IsDone(){return done_;};
        // size_t GetItersPerEpoch() override;
        // std::map<std::string, torch::Tensor> getNext() override;
};