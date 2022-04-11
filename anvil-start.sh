

#!/bin/bash

cd /DeepPool
pkill -9 -f localhost:1127
pkill -9 -f debugpy
pkill -9 -f run_planes_meters_
pkill -9 -f OpenDebug
pkill -9 -f autolrs_server
pkill -9 -f cluster
pkill -9 -f runtime
pkill -9 -f RM-
pkill -9 -f python
rm /dev/shm/batches/*
rm /dev/shm/val-batches/*
rm /dev/shm/eval-batches/*
rm samples/rank0*
rm samples/rank1*
rm samples/*
rm modules/scriptmodule_*
rm logs/cpprt*.out
rm logs/lastReq*
rm logs/runtime*

source /root/miniconda3/bin/activate
conda activate base

cd /anvil
python run_planes_meters_server.py &
python run_planes_meters_test_server.py &
python run_planes_meters_test_server.py validation &


cd /autolrs
python autolrs_server.py --min_lr 0.00000001 --max_lr 0.001 --host localhost --port 22334 --ranks 4 &

cd /DeepPool
python cluster.py --addrToBind localhost:12345 --c10dBackend nccl --be_batch_size=0 --cpp --logdir /DeepPool/logs &
# python examples/anvil_resnet.py 4 256 &
# tail -f logs/runtime0.out