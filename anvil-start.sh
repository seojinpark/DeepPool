

#!/bin/bash

pkill -9 -f localhost:1127
pkill -9 -f debugpy
pkill -9 -f run_planes_meters_
pkill -9 -f OpenDebug
rm /dev/shm/batches/*
rm /dev/shm/eval-batches/*
rm samples/rank0*
rm samples/rank1*
rm samples/*
rm modules/scriptmodule_*

source /root/miniconda3/bin/activate
conda activate base

cd ../anvil
python run_planes_meters_server.py &
python run_planes_meters_test_server.py &
