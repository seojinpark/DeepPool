
#include "AutoLRSClient.h"
// #include <netinet/in.h>
#include <netdb.h>
#include <chrono>
#include "logger.h"
#include "runnableModule.h"
        
AutoLRSClient::AutoLRSClient(std::shared_ptr<RunnableModule> model, std::shared_ptr<torch::optim::RMSprop> optimizer, 
        JobContext* job, std::string checkpoint_path, std::string listening_host, 
        uint64_t listening_port, uint64_t warmup_steps, double warmup_lr, uint64_t summary_steps) :
        model_(model), optimizer_(optimizer), job_(job), checkpoint_path_(checkpoint_path),
        listening_host_(listening_host),listening_port_(listening_port), warmup_steps_(warmup_steps), 
        warmup_lr_(warmup_lr), summary_steps_(summary_steps)  {

    stepSched_=NULL;
    checkpoint_path_ = checkpoint_path_.append(std::to_string(rtctx->rank));
    if (!std::filesystem::exists(checkpoint_path_)){
        std::filesystem::create_directory(checkpoint_path_);
    }

    // if(warmup_steps_ > 0)
    //     lr_history.push_back(warmup_lr_);

    // if(rtctx->rank==0)
    connect_server();
    // else
    //     save_variables();
}

AutoLRSClient::AutoLRSClient(std::shared_ptr<RunnableModule> model, std::shared_ptr<torch::optim::RMSprop> optimizer, 
        JobContext* job, std::string checkpoint_path, unsigned step_size, double gamma, uint64_t warmup_steps) : 
        model_(model), optimizer_(optimizer), job_(job), checkpoint_path_(checkpoint_path), warmup_steps_(warmup_steps) {
    stepSched_ = std::make_unique<torch::optim::StepLR>(*optimizer, step_size, gamma);

    checkpoint_path_ = checkpoint_path_.append(std::to_string(rtctx->rank));
    if (!std::filesystem::exists(checkpoint_path_)){
        std::filesystem::create_directory(checkpoint_path_);
    }

    lr_ = 0.001;
}

void AutoLRSClient::step(){
    if(stepSched_ != NULL){
        stepSched_->step();
    }
}

void AutoLRSClient::connect_server(){
    // self._socket.connect((self._listening_host, self._listening_port))
    // int sockfd, n;
    // struct sockaddr_in serv_addr;
    struct hostent *server;

    // int portno = atoi(argv[2]); listening_port_

    socket_ = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_ < 0){
        perror("ERROR opening socket");
        exit(-1);
    }

    server = gethostbyname(listening_host_.c_str());
    if (server == NULL) {
        fprintf(stderr,"ERROR, no such host\n");
        exit(0);
    }

    bzero((char *) &server_addr_, sizeof(server_addr_));
    server_addr_.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&server_addr_.sin_addr.s_addr, server->h_length);
    server_addr_.sin_port = htons(listening_port_);

    if (connect(socket_, (struct sockaddr *) &server_addr_, sizeof(server_addr_)) < 0) {
        perror("ERROR connecting");
        exit(-1);
    }
}

void AutoLRSClient::verbose_operation(std::string op_){
    //     if self._global_step % self._summary_steps == 0:
    //         logging.info("[AutoLRS at {}] {}".format(self._global_step, _op))
    if (global_step_ % summary_steps_ == 0)
        // std::cout << "AutoLRS at " << global_step_ << op_<< std::endl;
        DP_LOG(NOTICE, "[AutoLRS at %ld] %s", global_step_, op_.c_str());
}

void AutoLRSClient::save_variables(std::string inpath){
    // """Save model parameters and optimizer states."""
    auto start = std::chrono::high_resolution_clock::now();

    std::filesystem::path chkPath;
    if (inpath != ""){
        chkPath = std::filesystem::path(inpath);
        if (!std::filesystem::exists(chkPath)){
            std::filesystem::create_directory(chkPath);
        }
    }
    else{
        chkPath = std::filesystem::path(checkpoint_path_);
        if (!std::filesystem::exists(chkPath)){
            std::filesystem::create_directory(chkPath);
        }
    }

    model_->saveOptimizer(std::filesystem::path(chkPath.string()).append("optimizer.pt").string());
    // torch::save(*optimizer_.get(), std::filesystem::path(checkpoint_path_).append("optimizer.pt").string());
    for (auto layer : model_->layers){
        if(layer->active){
            // layer->module.train();
            layer->saveModule(std::filesystem::path(chkPath.string()).append(layer->timerkey + ".pt").string());
        }
    }

    // logging.info("[AutoLRS] backup variables, elapsed: {}s".format(time.time() - _start_time))
    auto end = std::chrono::high_resolution_clock::now();
    auto mills = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    DP_LOG(NOTICE, "[AutoLRS] backup variables, elapsed: %ldms", mills);
    std::cout << "\n[AutoLRS] backup variables, elapsed: " << mills << std::endl;
}

void AutoLRSClient::restore_variables(std::string inpath){
    auto start = std::chrono::high_resolution_clock::now();

    model_->parameters.clear();
    std::filesystem::path chkPath;
    if (inpath != "")
        chkPath = std::filesystem::path(inpath);
    else
        chkPath = checkpoint_path_;

    for (auto layer : model_->layers){
        if(layer->active){
            layer->loadModule(std::filesystem::path(chkPath.string()).append(layer->timerkey + ".pt").string());
            layer->module.to(rtctx->c10dev);
            // layer->module.train();
            for (const auto& params : layer->module.parameters()){
                // std::cout << params.requires_grad() << std::endl;
                model_->parameters.push_back(params);
            }
            // layer->module.to(rtctx->c10dev);
        }
    }
    
    model_->SetupOptimizer(std::filesystem::path(chkPath.string()).append("optimizer.pt").string());

    auto end = std::chrono::high_resolution_clock::now();
    auto mills = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    DP_LOG(NOTICE, "[AutoLRS] restoring variables, elapsed: %ldms", mills);
    std::cout << "\n[AutoLRS] restoring variables, elapsed: " << mills << std::endl;
}

std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

void AutoLRSClient::on_train_batch_end(double loss){
    if (stepSched_ != NULL)
        return;
    // if self._global_step < self._warmup_steps:
    //     # linear warmup
    //     self._lr = (self._warmup_lr / self._warmup_steps) * (self._global_step + 1)
    //     for param_group in self._optimizer.param_groups:
    //         param_group['lr'] = self._lr
    //     self._global_step += 1
    if(global_step_ < warmup_steps_){
        // lr_ = (warmup_lr_ / warmup_steps_) * (global_step_ + 1);
        // lr_history.push_back(lr_);
        lr_ = warmup_lr_;
        // for(auto param_group : optimizer_->param_groups())
        //     param_group.options().set_lr(lr_);

        for(auto param_group : optimizer_->param_groups())
            static_cast<torch::optim::RMSpropOptions &>(param_group.options()).lr(lr_);
        global_step_++;
    }
    else if(!started_){
        save_variables();
        restore_variables();
        started_ = true;

        // printf("Please enter the message: ");
        bzero(buffer_,1024);
        sprintf(buffer_, "startBO,%f", loss);
        auto n = write(socket_, buffer_, strlen(buffer_));
        if (n < 0) {
            perror("ERROR writing to socket");
            exit(-1);
        }
        // self._socket.send(",".join(('startBO', str(loss))).encode("utf-8"))
        verbose_operation("Start Bayesian Optimization(BO)");
        // data = self._socket.recv(1024).decode("utf-8")
        bzero(buffer_,1024);
        n = read(socket_, buffer_, 1024);
        if (n < 0) {
            perror("ERROR reading from socket");
            exit(-1);
        }
        std::string data(buffer_);
        verbose_operation("Received data: " + data);
        auto splitData = split(data,',');
        lr_ = std::stod(splitData.back());
        // lr_history.push_back(lr_);
        // lr_ = 0.001;

        for(auto param_group : optimizer_->param_groups())
            static_cast<torch::optim::RMSpropOptions &>(param_group.options()).lr(lr_);
    }
    else{
        bzero(buffer_,1024);
        sprintf(buffer_, "loss,%f", loss);
        auto n = write(socket_, buffer_, strlen(buffer_));
        if (n < 0) {
            perror("ERROR writing to socket");
            exit(-1);
        }
        // self._socket.send(','.join(('loss', str(loss))).encode('utf-8'))
        bzero(buffer_,1024);
        n = read(socket_, buffer_, 1024);
        if (n < 0) {
            perror("ERROR reading from socket");
            exit(-1);
        }
        //     data = self._socket.recv(1024).decode("utf-8")
        //     self._verbose_operation("Received data: " + data)
        std::string data(buffer_);
        verbose_operation("Received data: " + data);
        auto splitData = split(data,',');
        // if data.startswith("restore"):
        if(data.find("restore") == 0){
            restore_variables();
            verbose_operation("restore trainable variables");
        }
        // elif data.startswith("ckpt"):
        else if(data.find("ckpt") == 0){
            save_variables();
            restore_variables();
            verbose_operation("backup trainable variables");
        }
        // elif data.startswith('evaluate'):
        else if(data.find("evaluate") == 0){
            double val_loss = job_->Validation(global_step_);
            bzero(buffer_,1024);
            sprintf(buffer_, "val_loss,%f", val_loss);
            auto n = write(socket_, buffer_, strlen(buffer_));
            if (n < 0) {
                perror("ERROR writing to socket");
                exit(-1);
            }

            bzero(buffer_,1024);
            n = read(socket_, buffer_, 1024);
            if (n < 0) {
                perror("ERROR reading from socket");
                exit(-1);
            }
            // val_loss = self._val_fn()
            // self._socket.send(",".join(("val_loss", str(val_loss))).encode("utf-8"))
            // data = self._socket.recv(1024).decode("utf-8")
        }
        else if(data.find("save") == 0){
        }
        else{
            lr_ = std::stod(splitData.back());
            // lr_history.push_back(lr_);
            // lr_ = 0.001;
            for(auto param_group : optimizer_->param_groups())
                static_cast<torch::optim::RMSpropOptions &>(param_group.options()).lr(lr_);
                // param_group.options().set_lr(lr_);
            global_step_++;
        }
        //     elif data.startswith('save'):
        //         pass
        //     else:
        //         self._lr = (float(data.split(",")[-1]))
        //         for param_group in self._optimizer.param_groups:
        //             param_group['lr'] = self._lr
        //         self._global_step += 1
    }
}


double AutoLRSClient::get_last_lr(){
    if(stepSched_ != NULL){
        for(auto param_group : optimizer_->param_groups()){
            lr_ = param_group.options().get_lr();
            break;
        }
    }
    return lr_;
};