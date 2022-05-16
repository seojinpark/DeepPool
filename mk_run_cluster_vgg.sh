#!/bin/sh

export LD_LIBRARY_PATH='/home/mkeaton/msccl/build/lib:/home/mkeaton/cudnn/cuda/lib64:/usr/local/cuda/lib64'

export SCCL_XML_FILES='/home/mkeaton/DeepPool/ar-c8-i1.xml'

NCCL_ALGO=SCCL python3 cluster.py --addrToBind localhost:15507 --c10dBackend nccl --be_batch_size=0 --cpp --profile --logdir /home/mkeaton/DeepPool/logs --gpus 8 --sync_coalesce --sync_tensor_pad 8
