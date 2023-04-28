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
    -v /path/to/DeepPool:/DeepPool \
    -v /path/to/Data:/Data \
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

Once you see "Now, cluster is ready to accept training jobs." you may launch a job in another terminal. Make sure the number of GPUs you specify below matches the number of hosts in the hostfile. The GPU count is global, so make sure it's divisible by the number of GPUs you specify.

For example, to run ResNet18 with global batch size 32 on a single GPU, run:
```
python examples/resnetImported.py 1 32
```

`resnetImported.py` will contain other parameters, such as the directory to save and load the model from, the number of data workers per GPU, and how many epochs to train to. The model will not save until the final epoch.

To view the results of the run, inspect the contents of `logs/cpprt0.out`. Inspect the other logs if something goes wrong. When the job completes, you will see a line similar to the below in `logs/cpprt0.out`.
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
