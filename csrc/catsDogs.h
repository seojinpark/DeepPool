// Copyright 2020-present pytorch-cpp Authors
#pragma once

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>
#include <cstddef>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <iostream> 
#include <sstream> 

// CatsDogs dataset
// based on: https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/data/datasets/mnist.h.
class CatsDogs : public torch::data::datasets::Dataset<CatsDogs> {
 public:
    // The supplied path should be to the csv file pointing to all images and their labels
    explicit CatsDogs(const std::string root, size_t worldSize, size_t rank);

    // Returns the `Example` at the given `index`.
    torch::data::Example<> get(size_t index) override;

    // Returns the size of the dataset.
    torch::optional<size_t> size() const override;

 private:
    std::vector<std::tuple<std::string, int>> data;
    size_t worldSize;
    size_t rank;
};
