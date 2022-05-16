#!/bin/sh

python3 ./examples/dlrm_s_pytorch.py 4 MP --arch-embedding-size=1000000-1000000-1000000-1000000 --arch-sparse-feature-size=128 --arch-mlp-bot=128-128-128-128 --arch-mlp-top=1024-1024-1024-1024-1 --num-indices-per-lookup=100 --arch-interaction-op="dot" --max-ind-range=40000000 --data-generation=random --loss-function=bce --round-targets=True --learning-rate=1.0 --mini-batch-size=2048 --print-freq=2 --print-time --test-freq=2 --test-mini-batch-size=2048 --memory-map --use-gpu --num-batches=1000 --dist-backend=nccl
