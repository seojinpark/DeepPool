import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from parallelizationPlanner import CostSim
from clusterClient import ClusterClient
from torchvision.models import resnet18

import torch

input_shape = (3,224,224)
num_classes = 2
GPU_count = int(sys.argv[1])
global_batch_size = int(sys.argv[2])

model = resnet18(num_classes=num_classes)

# Register model with DeepPool and generate job
cs = CostSim()
cs.GeneralLayer(model, "model", {}, mustTrace=True)
cs.printAllLayers(silent=True)
cs.computeInputDimensions(input_shape)
job, iterMs, gpuMs, maxGpusUsed = cs.searchBestSplitsV3(GPU_count, global_batch_size)

# Push job to the cluster
cc = ClusterClient()

jobParams = {}
jobParams["catsdogs"] = True

jobParams['training_data'] = "/Data/catsDogs/train.csv"
jobParams['evaluation_data'] = "/Data/catsDogs/test.csv"
jobParams['num_workers'] = 64

jobParams['epochs_to_train'] = 2

# jobParams['checkpoint_path'] = "/DeepPool/checkpoints"
# jobParams['loading_path'] = "/DeepPool/checkpoints"

cc.submitTrainingJob("ResNet18", job.dumpInJSON(), jobParams=jobParams)
