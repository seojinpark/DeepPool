// # Raytheon Proprietary. Developed entirely at private expense. Patent pending
// # 18-11636-US-NP (1547.705US1)
// # This software is subject to the protections of the Associate Contractor
// # Agreement between Raytheon BBN Technologies Corp.,  Perspecta Labs, Inc and 
// # University of Southern California under the terms of DARPA prime contracts 
// # HR001120C0089 and HR001119S0082 for the FastNICs program.
// # WARNING - This document contains technology whose export or disclosure to
// # Non-U.S. persons, wherever located, is subject to the Export Administrations
// # Regulations (EAR) (15 C.F.R. Sections 730-774). Violations are subject to
// # severe criminal penalties.
// # Raytheon BBN Technologies conducted an internal review and determined this
// # information is export controlled as EAR99.

// import os
// import sys
// import numpy as np
// import json
// import time
// import pickle
// from copy import deepcopy
// from sortedcontainers import SortedList
// import matplotlib as mpl
// mpl.use('Agg')
// import matplotlib.pyplot as plt
// from skimage import io
// from src.metrics import fast_image
// from src.viewer import image_utils


#include "stats.h"
#include <tuple>
#include "utils.h"
#include <filesystem>
#include <opencv2/opencv.hpp>
// #include <Python.h>
// #include <matplotlibcpp.h>
#include <matplot/matplot.h>
// namespace plt = matplotlibcpp;

// using tuple2int = std::tuple<int64_t, int64_t>;
// using tuple4int = std::tuple<int64_t, int64_t, int64_t, int64_t>;
// using tuple3double = std::tuple<double, double, double>;
// using tuple4double = std::tuple<double, double, double, double>;

// typedef std::tuple<int64_t, int64_t> tuple2int;
// typedef std::tuple<int64_t, int64_t, int64_t, int64_t> tuple4int;
// typedef std::tuple<double, double, double> tuple3double;


auto tensorToCvImage2(at::Tensor tensor)
{
    int width = tensor.sizes()[0];
    int height = tensor.sizes()[1];
    // int channels = tensor.sizes()[2];
    auto dtype = CV_8UC3;

    if (tensor.sizes()[2] == 1)
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

Stats::Stats(std::vector<std::string> labels, std::string background_label,
                           int64_t num_frames, double threshold, double display_threshold,
                           std::string mode, double spacing) :
            labels_(labels), background_label_(background_label), num_frames_(num_frames),
            threshold_(threshold), display_threshold_(display_threshold), mode_(mode), spacing_(spacing)
{
    auto it = std::find (labels_.begin(), labels_.end(), background_label_);
    if (it != labels_.end())
        background_index_ = it - labels_.begin();

    // int64_t sclass_count_ = 0;
    int64_t label_index = 0;
    std::vector<int> sclass_detects_;
    if (mode_ == "single"){
        sclass_.count_ = 0;
        sclass_.detects_ = std::vector<int>();
    }
    else if (mode_ == "multi"){
        for (auto label : labels_){
            if (label_index == background_index_)
                continue;
            pr_[label] = sclass();
            pr_[label].count_ = 0;
            pr_[label].detects_ = std::vector<int>();
        }
    }
    else if (mode_ == "chip")
        results_.clear();
}

Stats::~Stats(){

}


std::tuple<int64_t, int64_t, int64_t, torch::Tensor, std::vector<tuple4double>>
    Stats::batch(std::map<std::string, torch::Tensor> results){
    // if (mode_ == "single")
    //     return batch_single(result)
    // else if (mode_ == "multi")
    //     return batch_multi(result)
    // if (mode_ == "chip")
    return batch_chip(results);

    // std::map<std::string, torch::Tensor> res;
    // return res;
}


std::vector<int64_t> Stats::unravel_index(int64_t index, c10::IntArrayRef shape){
    std::vector<int64_t> out;
    for (auto dim = shape.rbegin(); dim!=shape.rend(); ++dim){
        out.push_back(index % *dim);
        index = (int64_t) (index / *dim);
    }
    std::reverse(out.begin(),out.end());
    return out;
}

torch::Tensor Stats::find_peaks(torch::Tensor image_in, c10::IntArrayRef shape, double threshold){
    // cdef float[:, :] image_out = np.zeros(shape, dtype=np.float32)
    torch::Tensor image_out = torch::zeros(shape, torch::TensorOptions().dtype(torch::kFloat32));
    int32_t row, col, rp, cp;
    float value;
    int is_top;
    for (row=0; row<shape[0]; row++){
        for (col=0; col<shape[1]; col++){
            if (image_in.index({row, col}).item<double>() >= threshold){
                value = image_in.index({row, col}).item<double>();
                is_top = 1;
                for (rp=std::max(0, row - 3); rp < std::min((int32_t)shape[0], row + 4); rp++){
                    // for cp in range(c_max(0, col - 1), c_min(shape[1], col + 2)):
                    for (cp=std::max(0, col - 3); cp < std::min((int32_t)shape[1], col + 4); cp++){
                        if (image_in.index({rp, cp}).item<double>() > value){
                            is_top = 0;
                        }
                    }
                }
                if (is_top)
                    image_out.index_put_({row, col}, value);
            }
        }
    }
    return image_out;
}

template <typename T, typename rT>
rT getTupleIdx(T tup, const int64_t idx) {

    // assert(idx < std::tuple_size<decltype(tup)>::value);
    switch (idx)
    {
    case 0:
        return (rT)std::get<0>(tup);
    case 1:
        return (rT)std::get<1>(tup);
    case 2:
        return (rT)std::get<2>(tup);
    default:
        assert(idx < 3);
    }
}

void Stats::insert_box(torch::Tensor* image, int64_t x_cen, int64_t y_cen, int64_t radius, std::string color, int64_t thickness){

    tuple2int x_r = {std::max(x_cen - radius, (int64_t)0),
           std::min(x_cen + radius, image->sizes()[1] - 1)};
    tuple2int y_r = {std::max(y_cen - radius, (int64_t)0),
           std::min(y_cen + radius, image->sizes()[0] - 1)};

    // x_r = [max([x_cen - radius, 0]),
    //        min([x_cen + radius, image.shape[1] - 1])]
    // y_r = [max([y_cen - radius, 0]),
    //        min([y_cen + radius, image.shape[0] - 1])]
    
    std::tuple<double, double, double> rep;
    // if (color == "red")
    //     rep = {1.0, 0.0, 0.0};
    // else if (color == "green")
    //     rep = {0.0, 1.0, 0.0};
    // else if (color == "blue")
    //     rep = {0.0, 0.0, 1.0};
    // else if (color == "yellow")
    //     rep = {1.0, 1.0, 0.0};
    // else if (color == "cyan")
    //     rep = {0.0, 1.0, 1.0};

    if (color == "blue")
        rep = {1.0, 0.0, 0.0};
    else if (color == "green")
        rep = {0.0, 1.0, 0.0};
    else if (color == "red")
        rep = {0.0, 0.0, 1.0};
    else if (color == "cyan")
        rep = {1.0, 1.0, 0.0};
    else if (color == "yellow")
        rep = {0.0, 1.0, 1.0};
    else if (color == "orange")
        rep = {0.0, 0.65, 1.0};

    for (int64_t band=0;  band < image->sizes()[2]; band++){
        for (int64_t st=0; st < thickness; st++){
            // auto v = std::get<band>(rep);
            // auto v = getTupleIdx<tuple3double, double>(rep, band);
            // std::cout << std::get<0>(y_r) + st << " " << std::get<0>(x_r) + st << " " << std::get<1>(x_r) + 1 - st << " " << band << std::endl;
            // std::cout << std::get<1>(y_r) - st << " " << std::get<0>(x_r) + st << " " << std::get<1>(x_r) + 1 - st << " " << band << std::endl;
            // std::cout << std::get<0>(y_r) + st << " " << std::get<1>(y_r) + 1 - st << " " << std::get<0>(x_r) + st << " " << band << std::endl;
            // std::cout << std::get<0>(y_r) + st << " " << std::get<1>(y_r) + 1 - st << " " << std::get<1>(x_r) - st << " " << band << std::endl;
            image->index_put_({
                std::get<0>(y_r) + st,
                torch::indexing::Slice(std::get<0>(x_r) + st, std::get<1>(x_r) + 1 - st),
                band}, getTupleIdx<tuple3double, double>(rep, band));
            image->index_put_({
                std::get<1>(y_r) - st,
                torch::indexing::Slice(std::get<0>(x_r) + st, std::get<1>(x_r) + 1 - st),
                band}, getTupleIdx<tuple3double, double>(rep, band));
            image->index_put_({
                torch::indexing::Slice(std::get<0>(y_r) + st, std::get<1>(y_r) + 1 - st),
                std::get<0>(x_r) + st,
                band}, getTupleIdx<tuple3double, double>(rep, band));
            image->index_put_({
                torch::indexing::Slice(std::get<0>(y_r) + st, std::get<1>(y_r) + 1 - st),
                std::get<1>(x_r) - st,
                band}, getTupleIdx<tuple3double, double>(rep, band));
            // image[y_r[0] + st, (x_r[0] + st):(x_r[1] + 1 - st), band] = rep[band];
            // image[y_r[1] - st, (x_r[0] + st):(x_r[1] + 1 - st), band] = rep[band];
            // image[(y_r[0] + st):(y_r[1] + 1 - st), x_r[0] + st, band] = rep[band];
            // image[(y_r[0] + st):(y_r[1] + 1 - st), x_r[1] - st, band] = rep[band];
        }
    }
}

std::tuple<int64_t, int64_t, int64_t, torch::Tensor, std::vector<tuple4double>>
    Stats::batch_chip(std::map<std::string, torch::Tensor> result){

    torch::Tensor probability = result["pred"].clone();
    torch::Tensor image_batch = result["image"].clone();
    torch::Tensor label_batch = result["labels"].clone();

#ifdef STATS_DEBUG
    // std::cout << image_batch.index({0,0,0}) << std::endl;
    std::cout << "probability " << probability.sizes() << " " << probability.index({0, 0, torch::indexing::Slice()}) << std::endl;
    std::cout << "image_batch " << image_batch.sizes() << std::endl;
    std::cout << "label_batch " << label_batch.sizes() << std::endl;
#endif

    int64_t slack = 5; //# JHG: should be 5!
    auto bg_index = background_index_;

    double conversion = (double)image_batch.sizes()[0]/(double)label_batch.sizes()[0];

    int64_t lr_border, ud_border;
    if (spacing_ == 0){
        lr_border = 4; //# or 4
        ud_border = 4;
    }
    else{
        auto border_frac = (1 - spacing_)/2;
        auto ignore_region = torch::ceil(torch::tensor(label_batch.sizes())*border_frac).to(torch::kInt);
        lr_border = ignore_region.index({1}).item<int64_t>();
        ud_border = ignore_region.index({0}).item<int64_t>();
    }

    // #display_thresh = 0.11
    auto display_thresh = display_threshold_;
    double bare_min = 0.1;
    if (threshold_ != 0)
        bare_min = threshold_;

    auto center_frame = (int64_t) (num_frames_/2);
    // # begin multi label precision recall

    // # begin image marking for debug
    // # will not work for arbitrary classes
    int64_t found = 0;
    int64_t missClass = 0;
    int64_t missed = 0;
    int64_t fas = 0;

    torch::Tensor image;

    if (display_thresh != -1){//[:, :, (3*center_frame):(3*(center_frame + 1))]
        image = image_batch.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice((3*center_frame),(3*(center_frame + 1)))});
        image -= image.min();
        image /= image.max();
        if (image.sizes()[2] == 1)
            image = torch::cat({image.clone(), image.clone(), image.clone()}, 2);
    }
    // std::cout <<  " " << sprob << std::endl;
    auto sprob = 1.0 - probability.index({torch::indexing::Slice(), torch::indexing::Slice(), bg_index});
    // auto sprob_max = sprob.max().item<double>();
    // auto sprob_min = sprob.min().item<double>();
    // std::cout <<  "sprob max min " << sprob_max << " " << sprob_min << std::endl;

    // #print('delete me', sprob.max())
    // #from skimage import io
    // #io.imsave('prob.png', (sprob*255.999/sprob.max()).astype(np.uint8))

    std::vector<torch::Tensor> slabel_pre_stack;
    for (int label_index=0; label_index<probability.sizes()[2]; label_index++)
        if (label_index != bg_index)
            slabel_pre_stack.push_back(label_batch.index({torch::indexing::Slice(), torch::indexing::Slice(), label_index}).clone());
    auto slabel = torch::stack(slabel_pre_stack, 2);
    slabel = slabel.sum(2);

    // #print('delete me', sprob.max())
    // #io.imsave('lab.png', (slabel*255.999/slabel.max()).astype(np.uint8))

#ifdef STATS_DEBUG
    char buff[1024];
    snprintf(buff, sizeof(buff), "/DeepPool/stat_outputs/probability.jpg");
    std::string buffAsStdStr = buff;
    auto disp_sprob = torch::unsqueeze((255.999*probability/probability.max()), -1).to(torch::kUInt8).contiguous().clone();

    auto cvImg = tensorToCvImage2(disp_sprob);
    cv::imwrite(buffAsStdStr, cvImg);


    buff[1024];
    snprintf(buff, sizeof(buff), "/DeepPool/stat_outputs/sprobA.jpg");
    buffAsStdStr = buff;
    disp_sprob = torch::unsqueeze((255.999*sprob/sprob.max()), -1).to(torch::kUInt8).contiguous().clone();

    cvImg = tensorToCvImage2(disp_sprob);
    cv::imwrite(buffAsStdStr, cvImg);

    snprintf(buff, sizeof(buff), "/DeepPool/stat_outputs/sprobB.jpg");
    buffAsStdStr = buff;
    disp_sprob = torch::unsqueeze((255.999*sprob/probability.max()), -1).to(torch::kUInt8).contiguous().clone();

    cvImg = tensorToCvImage2(disp_sprob);
    cv::imwrite(buffAsStdStr, cvImg);
#endif

    // # local maxes only
    auto new_prob = find_peaks(sprob, sprob.sizes(), bare_min).clone();

    // std::cout <<  "new_prob " << new_prob << std::endl;
#ifdef STATS_DEBUG
    // std::cout <<  "new_prob " << tsrSizeToStr(new_prob).c_str() << std::endl;
    std::cout <<  "new_prob " << new_prob << std::endl;
    // char buff[1024];
    snprintf(buff, sizeof(buff), "/DeepPool/stat_outputs/new_prob.jpg");
    buffAsStdStr = buff;
    auto disp_new_prob = torch::unsqueeze((255.999*new_prob/new_prob.max()), -1).to(torch::kUInt8).contiguous().clone();
    cvImg = tensorToCvImage2(disp_new_prob);
    cv::imwrite(buffAsStdStr, cvImg);
    // new_prob = np.copy(new_prob);
#endif

    // #print('delete me', new_prob.max())
    // #io.imsave('new_prob.png', (new_prob*255.999/new_prob.max()).astype(np.uint8))

    std::vector<std::tuple<int64_t, int64_t, double, int64_t>> the_found;
    // std::cout << "torch::count_nonzero(new_prob) " << torch::count_nonzero(new_prob) << std::endl;

    while (torch::count_nonzero(new_prob).item<int32_t>() > 0){
        std::vector<int64_t> tmp = unravel_index(torch::argmax(new_prob).item<int64_t>(), sprob.sizes());
        int64_t row = tmp[0];
        int64_t col = tmp[1];

        auto by_label = probability.index({row, col, torch::indexing::Slice()}).clone();
        by_label.index_put_({bg_index}, 0.0);
        int64_t label_index = by_label.argmax().item<int64_t>();
#ifdef STATS_DEBUG
        std::cout <<  "the_found.push_back " << label_index << ", " << row <<", "<< col << ", " << new_prob.index({row, col}).item<double>() << std::endl;
        // std::cout <<  "new_prob[row, col] " << tsrSizeToStr(new_prob[row, col]).c_str() << std::endl;
#endif
        the_found.push_back({row, col, new_prob.index({row, col}).item<double>(), label_index});

        tuple2int rmm = {std::max((int64_t)0, row - slack),
                std::min(sprob.sizes()[0], row + slack + 1)};
        tuple2int cmm = {std::max((int64_t)0, col - slack),
                std::min(sprob.sizes()[1], col + slack + 1)};
        new_prob.index_put_({torch::indexing::Slice(std::get<0>(rmm),std::get<1>(rmm)),
                torch::indexing::Slice(std::get<0>(cmm),std::get<1>(cmm))}, 0.0);
    }

    std::vector<tuple4double> orig_found;
    if (false){ //# JHG: write output for Bob
        for (auto [r, c, p, l] : the_found)
            orig_found.push_back({(double)r*conversion + 0.5,
                                (double)c*conversion + 0.5, p, (double)l});
    }

    auto max_in_border = 0.0;
    auto max_out_border = 0.0;
    for (auto [r, c, p, l] : the_found){
        if (c >= lr_border && c < sprob.sizes()[1] - lr_border &&\
            r >= ud_border && r < sprob.sizes()[0] - ud_border){
            if (p > max_in_border)
                max_in_border = p;
        }
        else{
            if (p > max_out_border)
                max_out_border = p;
        }
    }

#ifdef STATS_DEBUG
    std::cout <<  "slabel " << slabel << std::endl;
#endif
    auto the_where = torch::where(slabel > 0.0);
    auto the_where_rows = the_where[0];
    auto the_where_cols = the_where[1];
    assert(the_where_rows.size(0)==the_where_cols.size(0));
    // std::cout <<  "the_where[0] " << tsrSizeToStr(the_where[0]).c_str() << std::endl;
    // std::cout <<  "the_where[0] " << the_where[0] << std::endl;
    // std::cout <<  "the_where[1] " << the_where[1] << std::endl;


    // for (int64_t idx=the_where_rows.size(0); idx >= 0; idx--){
    //     int64_t row = the_where_rows.index({idx}).item<int64_t>();
    //     int64_t col = the_where_cols.index({idx}).item<int64_t>();
    // }


    std::vector<tuple2i1d1i> to_remove;
    bool true_in_border = false;
    bool true_out_border = false;
    bool in_border;
    // for (auto row_and_col : the_where){
    std::vector<tuple2int> marked;
    for (int64_t idx=0; idx < the_where_rows.size(0); idx++){

        // std::cout <<  "row_and_col " << tsrSizeToStr(row_and_col).c_str() << std::endl;
        int64_t row = the_where_rows.index({idx}).item<int64_t>();
        int64_t col = the_where_cols.index({idx}).item<int64_t>();
        bool markedAlready = false;
        for (auto [r, c] : marked){
            auto l1 = slabel.index({r,c}).item<float>();
            auto l2 = slabel.index({row,col}).item<float>();
            if (std::abs(row-r) < 2 && std::abs(col-c) < 2 && std::abs(l1-l2) < 0.0000001){
                markedAlready=true;
                break;
            }
        }
        if(markedAlready) continue;
        marked.push_back({row, col});

        if (col >= lr_border && col < sprob.sizes()[1] - lr_border and\
            row >= ud_border && row < sprob.sizes()[0] - ud_border){
            in_border = true;
            true_in_border = true;
        }
        else{
            in_border = false;
            true_out_border = true;
        }

        tuple2int rmm = {std::max((int64_t)0, row - slack),
                std::min(new_prob.sizes()[0], row + slack + 1)};
        tuple2int cmm = {std::max((int64_t)0, col - slack),
                std::min(new_prob.sizes()[1], col + slack + 1)};
        bool has_found = false;
        bool found_but_wrongl = false;
        int64_t current = -1;
        for (int64_t chrow=std::get<0>(rmm); chrow < std::get<1>(rmm); chrow++){
            for (int64_t chcol=std::get<0>(cmm); chcol < std::get<1>(cmm); chcol++){
                int64_t findex = 0;
                for (auto [r, c, p, l] : the_found){
                    if (r == chrow && c == chcol){
                        auto otherl = (l==1) ? 2 : 1;
                        if (std::count(to_remove.begin(), to_remove.end(), tuple2i1d1i({r, c, p, l})) == 0){
                            to_remove.push_back(tuple2i1d1i({r, c, p, l}));
                            if (!has_found){
                                has_found = true;
                                current = findex;
                                found_but_wrongl = false;
                            }
                            else if (p > std::get<2>(the_found[current])){
                                current = findex;
                                found_but_wrongl = false;
                            }
                        }
                        else if (std::count(the_found.begin(), the_found.end(), tuple2i1d1i({r, c, p, otherl})) > 0 && 
                                 std::count(to_remove.begin(), to_remove.end(), tuple2i1d1i({r, c, p, otherl})) == 0){
                            to_remove.push_back(tuple2i1d1i({r, c, p, otherl}));
                            if (!has_found){
                                has_found = true;
                                current = findex;
                                found_but_wrongl = true;
                            }
                            else if (p > std::get<2>(the_found[current])){
                                current = findex;
                                found_but_wrongl = true;
                            }
                        }
                    }
                    findex++;
                }
            }
        }
        marked.push_back({row, col});
        if (has_found){
            auto adj_index = std::get<3>(the_found[current]);
            auto color = (adj_index > 1) ? "cyan" : "green";
            int64_t bsize = 10;
            if (display_thresh != 0){
                if (std::get<2>(the_found[current]) > display_thresh && !found_but_wrongl){
                    insert_box(&image,
                                (uint64_t)((double)std::get<1>(the_found[current])*conversion + 0.5),
                                (uint64_t)((double)std::get<0>(the_found[current])*conversion + 0.5),
                                bsize, color);
                    found += 1;
                }
                else if (found_but_wrongl){
                    insert_box(&image,
                                (uint64_t)((double)std::get<1>(the_found[current])*conversion + 0.5),
                                (uint64_t)((double)std::get<0>(the_found[current])*conversion + 0.5),
                                bsize, "orange");
                    found += 1;
                    missClass += 1;
                }
                else{
                    insert_box(&image,
                                (uint64_t)((double)std::get<1>(the_found[current])*conversion + 0.5),
                                (uint64_t)((double)std::get<0>(the_found[current])*conversion + 0.5),
                                bsize, "yellow");
                    missed += 1;
                }
            }
        }
        else if (display_thresh != 0){
            insert_box(&image,
                (uint64_t)((double)col*conversion + 0.5),
                (uint64_t)((double)row*conversion + 0.5),
                10, "yellow");
            missed += 1;
        }
    }

    auto max_val = std::max(max_in_border, max_out_border);
    if (true_in_border)
        results_.push_back({max_val, true});
    else if (!true_out_border)
        results_.push_back({max_val, false});

    // std::count(to_remove.begin(), to_remove.end(), tuple4int({r, c, p, l})) == 0
    // for rem in to_remove:
    for (auto rem : to_remove){
        auto it = std::find(the_found.begin(), the_found.end(), rem);
        the_found.erase(it);
    }

    for (auto [r, c, p, l] : the_found){
        if (c >= lr_border && c < sprob.sizes()[1] - lr_border &&\
            r >= ud_border && r < sprob.sizes()[0] - ud_border){
            if (display_thresh != 0){
                if (p > display_thresh){
                    insert_box(&image,
                                (uint64_t)((double)c*conversion + 0.5),
                                (uint64_t)((double)r*conversion + 0.5),
                                10, "red");
                    fas += 1;
                    // break;
                }
            }
        }
    }

    std::tuple<int64_t, int64_t, int64_t, torch::Tensor, std::vector<tuple4double>> results;
    if (display_thresh != 0){
        auto disp_image = (255*image).to(torch::kUInt8).clone();
        // found
        results = {found, missed, fas, disp_image, orig_found};
    }
    else
        // results = {'found': found, 'missed': missed, 'false': fas};
        results = {found, missed, fas, torch::Tensor(), orig_found};

    return results;
}

std::tuple<std::vector<double>, std::vector<double>> Stats::_build_roc_curve(){
    int64_t fas = 0;
    int64_t dts = 0;
    for(auto res : results_){
        if (res.truth)
            dts += 1;
        else
            fas += 1;
        // fas = sum([int(not res[1]) for res in results_])
        // dts = sum([int(res[1]) for res in results])
    }
    double dtr = 0.0;
    double far = 0.0;
    std::vector<double> det;
    std::vector<double> fa;
    std::sort(results_.begin(), results_.end(), 
    [](const Result &a, const Result &b) -> bool{ 
        if (a.val != b.val)
            return a.val > b.val; 
        else
            return a.truth > b.truth; 
    }); 
    
    for(auto res : results_){
        if (res.truth)
            dtr += 1.0/((double)dts);
        else
            far += 1.0/((double)fas);
        det.push_back(dtr);
        fa.push_back(far);
    }

    return {det, fa};

    // for prob, truth in sorted(results, reverse=True):
    //     if truth:
    //         dtr += 1/dts
    //     else:
    //         far += 1/fas
    //     det.append(dtr)
    //     fa.append(far)

    // return np.array(det), np.array(fa)
}

double Stats::_area_under_roc(std::vector<double> det, std::vector<double> fa){
    double current_auc = 0.0;
    for (uint64_t i = 0; i < det.size() - 1; i++)
    {
        double y = (det[i] + det[i + 1])/2.0;
        double dx = abs(fa[i + 1] - fa[i]);
        current_auc += (y*dx);
    }
    
    // for index in range(len(det) - 1):
    //     y = (det[index] + det[index + 1])/2.0
    //     dx = abs(fa[index + 1] - fa[index])
    //     current_auc += y*dx

    return current_auc;
}
void Stats::_plot_roc_curve(std::string filename, std::vector<double> det, std::vector<double> fa, double auc){
    std::string savename = filename;

    for_each(det.begin(), det.end(), [](double &val){ val *= 100.0; });

    char buff[64];

    // using namespace matplot;

    // plot(x, y)->color({0.f, 0.7f, 0.9f});
    auto f = matplot::figure(true);
    matplot::semilogx(fa, det);
    matplot::ylim({0.0, 100.0});
    // matplot::xlim({0.0, 1.0});
    matplot::grid(true);

    memset(buff, 0, 64);
    snprintf(buff, sizeof(buff), " auc: %f", auc);

    // text(4, 2, "Curvature 1")->alignment(labels::alignment::center);
    matplot::text(.01,10.0, buff)->alignment(matplot::labels::alignment::center);
    memset(buff, 0, 64);
    snprintf(buff, sizeof(buff), " aoc: %f", 1.0-auc);
    matplot::text(.01,90.0, buff)->alignment(matplot::labels::alignment::center);

    // title("2-D Line Plot");
    matplot::xlabel("False Alarm Rate (fraction)");
    matplot::ylabel("Detection Rate (percent)");
    // matplot::save(savename);

    std::string path = "/DeepPool/ROC_all.png";
    
    // epochLen_*epochCount_ + counter_ >= datasetSize_/worldSize_
    if (std::filesystem::exists(path))
        std::filesystem::remove(path);


    f->save(savename);

    // show();
    // return 0;

    // Py_SetProgramName(L"plot_roc");
    // Py_Initialize();
    // PyObject *pTimeModule = PyImport_ImportModule("time");
    // PyRun_SimpleString( "import matplotlib.pyplot as plt\n\n" );

    // PyObject *pModule = PyImport_ImportModule("matplotlib.pyplot");
    // PyObject *pPltModule = PyImport_ImportModule("matplotlib");
    // Py_Finalize();
              
    // plt::figure();
    // plt::
    // 

    

    
    // plt::grid(true);
    // plt::save(savename);
}
void Stats::close_chip(){

    // std::string filename = "chips.dat";
    // with open(os.path.join('output', filename), 'wb') as fid:
    //     pickle.dump(self._results, fid);
        
    auto [det, fa] = _build_roc_curve();
    std::string roc_filename = "/DeepPool/stat_outputs/ROC_all.png";//os.path.join('output', 'ROC_all.png');
    auto auc = _area_under_roc(det, fa);
    _plot_roc_curve(roc_filename, det, fa, auc);
    std::cout << "Wrote " << roc_filename << " with auc = " << auc <<std::endl;
}


// uint64_t Stats::getResultsSizeInBytes(){
//     auto rsize = results_.size();
//     return (sizeof(std::vector<std::tuple<double, bool>>) + rsize*(sizeof(double) + sizeof(bool)));
// }



// void Stats::plot_lrsched(torch::optim::StepLR lrsched){

//     std::string savename = "lrschedule.png";
//     char buff[64];

//     // using namespace matplot;

//     // plot(x, y)->color({0.f, 0.7f, 0.9f});
//     auto f = matplot::figure(true);

//     std::vector<double> x = matplot::linspace(0, lrsched);

//     // matplot::semilogx(fa, det);
//     // matplot::ylim({0.0, 100.0});
//     // matplot::xlim({0.0, 1.0});
//     matplot::grid(true);

//     // memset(buff, 0, 64);
//     // snprintf(buff, sizeof(buff), " auc: %f", auc);

//     // text(4, 2, "Curvature 1")->alignment(labels::alignment::center);
//     // matplot::text(.01,10.0, buff)->alignment(matplot::labels::alignment::center);
//     // memset(buff, 0, 64);
//     // snprintf(buff, sizeof(buff), " aoc: %f", 1.0-auc);
//     // matplot::text(.01,90.0, buff)->alignment(matplot::labels::alignment::center);

//     // title("2-D Line Plot");
//     matplot::xlabel("Epoch");
//     matplot::ylabel("Learning Rate");
//     // matplot::save(savename);
//     f->save(savename);
// }