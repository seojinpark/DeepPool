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
    -v "$(pwd)"path/to/Data:/Data \
    nvcr.io/nvidia/pytorch:22.01-py3
```

Make sure both /DeepPool and /Data are populated. If not, then the paths in the previous command were incorrect.

Now in the container, modify pytorch, and build DeepPool:
```
cd /DeepPool
tar -xvf modifyPytorchSource.tar.gz -C /
bash build.sh
```

If during the build process you get an error that "dataloader->sampler_" is private, not public, then the tar file was incorrectly unzipped. Make sure it modifies the pytorch source code.

## Data

Locate the data from the previous DeppPool version, it has not changed for this version.

The CatsDogs dataset was downloaded from Kaggle [https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition/data?select=train.zip]. The images in the Train directory were split into training data and evaluation data. For each of these two new directories a CSV file was generated, containing the filepaths and labels for each image.

For the example of the CatsDogs dataset, the code for loading and pre-processing can be seen in `/DeepPool/csrc/catsDogs.cpp` and `/DeepPool/csrc/dataset.cpp`. Dataloading is done in parallel by multiple workers. Each GPU gets their own workers, so be careful with multiple GPUs that you don't over-subscribe your CPU cores. If you get an error like `Caught signal 11 (Segmentation fault: address not mapped to object at address 0x7f96280568f4)`, then you are overloading your CPU and need to either decrease the batch size, decrease the number of workers, or give more RAM to the docker container.

After changing any c++ code, you rebuild the project with `bash build.sh`.

## Run

Launch the cluster coordinator:
```
python cluster.py \
    --addrToBind 127.0.0.1:12347 \
    --be_batch_size=0 \
    --cpp \
    --logdir /DeepPool/logs \
    --hostfile hostfile.txt
```

Once you see "Now, cluster is ready to accept training jobs." you may launch a job in another terminal.
For example, to run ResNet18 with global batch size 32 on a single GPU, run:
```
python examples/resnetImported.py 1 32
```

To view the results of the run, inspect the contents of `logs/cpprt0.out`. For more detailed logs, like forward and backward times, look in the `logs/runtime` logs. When the job completes, you will see a line similar to the below in `logs/cpprt0.out`.
```
1677704218.374995837 runtime.cpp:222 in poll NOTICE[1]: Removed the completed job. Remaining: 0
```

Hitting Ctrl-C should kill the cluster, but if there are ever lingering processes, run twice
```
pkill runtime
```

## Multi-GPU

Ensure that when you start the docker container that you give it access to the GPUs, most easily done with the flag `--gpus all`, or you can specify specific GPUs instead.

Update the hostfile to point to multiple GPUs. For example, to specify 4 GPUs on the local machine, the hostfile looks like
```
#address,gpu_idx
127.0.0.1,0
127.0.0.1,1
127.0.0.1,2
127.0.0.1,3
```

Now when you generate the job for your model, pass in the number of GPUs to be used. For example, `python examples/resnetImported.py 4 128` will run on the 4 GPUs with a global batch size of 128, meaning the local batch size is 32.
