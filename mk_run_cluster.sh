#!/bin/sh

python3 cluster.py --addrToBind localhost:15507 -c10dBackend nccl --be_batch_size=0 --cpp --profile --logdir /home/mkeaton/DeepPool/logs
