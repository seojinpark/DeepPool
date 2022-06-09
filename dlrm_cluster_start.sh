#!/bin/sh

/bin/rm -f logs/*.txt logs/*.out logs/*.err

export LD_LIBRARY_PATH='/msccl/build/lib:/usr/local/cuda/lib64'

export SCCL_XML_FILES='/CLOPTNet/schedules/ar-c8-i1.xml:/CLOPTNet/schedules/bipartite-k4-N8-i1-c4-AtoA-LL.xml'

NCCL_ALGO=SCCL python3 cluster.py --addrToBind localhost:12347 --c10dBackend nccl --be_batch_size=0 --cpp --profile --logdir /DeepPool/logs --gpus 8 --sync_coalesce --sync_tensor_pad 8
