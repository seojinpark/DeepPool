
#ifndef AUTO_LRS_CLIENT_H
#define AUTO_LRS_CLIENT_H

#include <torch/torch.h>
#include <torch/types.h>
// #include <pthread.h>
// #include "dataset.h"
#include <string>
// #include <stdio.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <filesystem>
#include "runnableModule.h"


class AutoLRSClient{

    public:
        AutoLRSClient(std::shared_ptr<RunnableModule> model, std::shared_ptr<torch::optim::RMSprop> optimizer, 
                JobContext* job, std::string checkpoint_path, std::string listening_host="localhost", 
                uint64_t listening_port=12315, uint64_t warmup_steps=0, double warmup_lr=0.0001, uint64_t summary_steps=1);
        AutoLRSClient(std::shared_ptr<RunnableModule> model, std::shared_ptr<torch::optim::RMSprop> optimizer, 
                JobContext* job, std::string checkpoint_path, unsigned step_size, double gamma, uint64_t warmup_steps=0);
        void on_train_batch_end(double loss);
        double get_last_lr();
        void save_variables(std::string inpath="");
        void restore_variables(std::string inpath="");
        void step();
        uint64_t get_warmup_steps(){return warmup_steps_;};
        // double update_lr(double newLR);

    protected:
        std::shared_ptr<torch::optim::StepLR> stepSched_;
        std::shared_ptr<RunnableModule> model_;
        std::shared_ptr<torch::optim::RMSprop> optimizer_;
        JobContext* job_;
        std::filesystem::path checkpoint_path_;
        std::string listening_host_{"localhost"};
        uint64_t listening_port_{12345};
        uint64_t warmup_steps_{0};
        double warmup_lr_{0.001};
        uint64_t summary_steps_{1};

        char buffer_[1024] = {0};
        double lr_{0.001};
        uint64_t global_step_{0};
        int socket_;
        struct sockaddr_in server_addr_;
        bool started_{false};

        // std::vector<double> lr_history; 

        void verbose_operation(std::string op_);
        void connect_server();


};

#endif