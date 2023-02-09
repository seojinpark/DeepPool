// Copyright 2020-present pytorch-cpp Authors
#include "catsDogs.h"

// Note: we currently load from disk at every iteration, some time could be saved (not guaranteed)
// by pre-loading the entire dataset into RAM before training.

// read_data reads the CSV file and parses it to be ready when a new image is requested
std::vector<std::tuple<std::string, int>> read_data(std::string img_names_src)
{
    std::vector<std::tuple<std::string, int>> data;

    // Stream to text file
    std::ifstream name_stream;
    name_stream.open(img_names_src);
    if (!name_stream.good())
    {
        std::cout << "Could not open file " << img_names_src << std::endl;
    }

    // load file paths and labels to ram
    int lineNum = 0;
    while (name_stream.good())
    {
        lineNum++;
        std::string img_name;
        std::string label;
        int int_label;
        try
        {
            getline(name_stream, img_name, ',');
            getline(name_stream, label, '\n');
            int_label = std::stoi(label);
            data.push_back(make_tuple(img_name, int_label));
        }
        catch (...)
        {
            std::cerr << "Failed to parse line " << lineNum << " of " << img_names_src << ", got name: " << img_name << ", and label: " << label << "\n";
        }
    }

    return data;
}

CatsDogs::CatsDogs(std::string root)
{
    data = read_data(root);
}

// get returns an image-label pair, which it does by loading the image from disk, augmenting it, and converting it to a tensor.
torch::data::Example<> CatsDogs::get(size_t index)
{

    if (cache.count(index) > 0)
        return cache[index];

    // images have 3 channels with values from 0-255. Size is not consistent, seen 200-500 for width and height values.
    // store image in Mat object
    std::string image_path = std::get<0>(data[index]);
    cv::Mat img = cv::imread(image_path);
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        assert(false);
    }

    // pre-process image at train time

    // normalize to 0.0 and 1.0
    // img = img / 255.0;

    // remove border
    // int startX=10, startY=10, width=img.cols-20, height=img.rows-20;
    // img = cv::Mat(img, cv::Rect(startX, startY, width, height));

    // resize
    cv::resize(img, img, cv::Size(224, 224));

    torch::Tensor tensor_image = torch::from_blob(img.data, {img.rows, img.cols, 3}, at::kByte);

    // permute dimensions of tensor to Libtorch format [Channels, Height, Width]
    tensor_image = tensor_image.permute({2, 0, 1});

    torch::Tensor label_tensor = torch::tensor(std::get<1>(data[index]));

    // std::cerr << "Read image from " << image_path << std::endl;


    // {images.to(torch::kFloat32).div_(255), targets.to(torch::kInt64)};
    cache[index] = {tensor_image.to(torch::kFloat32).div_(255), label_tensor.to(torch::kInt64)};
    return cache[index];

    // fake data
    // torch::Tensor tensor_image = torch::zeros({3, 224, 224}, torch::kF32);
    // torch::Tensor label_tensor = torch::tensor(0); // note that tensor and Tensor are different!!

    // return {tensor_image, label_tensor};
}

torch::optional<size_t> CatsDogs::size() const
{
    return data.size();
}