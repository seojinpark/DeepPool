# DeepPool

## Install

Ensure you have NVIDIA drivers available on your system, and the NVIDIA Container Toolkit installed [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker].
Download and run the NVIDIA PyTorch container with DeepPool and the data mounted:
```
docker run \
    --gpus all \
    --network="host" \
    --shm-size 4G \
    -it \
    --rm \
    -v "$(pwd)"/DeepPool:/DeepPool \
    -v "$(pwd)"/Data:/Data \
    nvcr.io/nvidia/pytorch:22.01-py3
```

Now in the container, build DeepPool:
```
cd /DeepPool
bash build.sh
```

## Data

The CatsDogs dataset was downloaded from Kaggle [https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data?select=train.zip]. The images in the Train directory were split into training data and evaluation data. A CSV file was generated for both training and evaluation, containing the filepaths and labels.

In `deeppoolexample/main.py` you provide a filepath to a CSV file that contains a list of image-label pairs. An example CSV file looks like the below
```
/Data/catsDogs/train/dog.9636.jpg,1
/Data/catsDogs/train/cat.3267.jpg,0
/Data/catsDogs/train/cat.8683.jpg,0
/Data/catsDogs/train/cat.4511.jpg,0
/Data/catsDogs/train/dog.11091.jpg,1
```

For the example of the CatsDogs dataset, the code for loading and pre-processing can be seen in `/DeepPool/csrc/catsDogs.cpp` and `/DeepPool/csrc/dataset.cpp`. Dataloading is done in parallel by A good rule of thumb is to set the number of workers equal to the number of CPU cores available. If you get an error like `Caught signal 11 (Segmentation fault: address not mapped to object at address 0x7f96280568f4)`, then you are overloading your CPU and need to either decrease the batch size, decrease the number of workers, or give more RAM to the docker container.

After changing any c++ code, you rebuild the project with `bash build.sh`.

## Run

Launch the cluster coordinator:
```
python cluster.py \
    --addrToBind 127.0.0.1:12347 \
    --hostfile hostfile.txt \
    --be_batch_size=0 \
    --cpp \
    --logdir /DeepPool/logs
```

Once you see "Now, cluster is ready to accept training jobs." you may launch a job in another terminal.
For example, to run ResNet18 with global batch size 32, run:
```
python examples/resnetImported.py 32
```

To view the results of the run, inspect the contents of `logs/cpprt0.out`. For more detailed logs, like forward and backward times, look in the `logs/runtime` logs. When the job completes, you will see a line similar to the below in `logs/cpprt0.out`.
```
A training job vgg16_8_32_2.0_DP is completed (1800 iters, 13.57 ms/iter, 73.71 iter/s, 0.00 be img/s, 32 globalBatchSize).
```

Hitting Ctrl-C should kill the cluster, but if there are ever lingering processes, run
```
pkill runtime
```

## Multi-GPU

Ensure that when you start the docker container that gives access to the gpus, most easily done with the flag `--gpus all`, or you can specify specific GPUs instead.

Update the hostfile to point to multiple GPUs. For example, 4 GPUs on the local machine, the hostfile looks like
```
#address,gpu_idx
127.0.0.1,0
127.0.0.1,1
127.0.0.1,2
127.0.0.1,3
```

Now when you generate the job for your model, pass in the number of GPUs to be used, for `examples/resnetImported.py` tht would be in line `cs.searchBestSplitsV3(GPU_count, global_batch_size)`, where `GPU_count` has a value of 4.