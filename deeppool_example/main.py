import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from parallelizationPlanner import CostSim
from clusterClient import ClusterClient
import resnet34

input_shape = (3,224,224)
num_classes = 2
gloabal_batch_size = 32
GPU_count = 1

model = resnet34.ResNet34(num_classes=num_classes)

# Register model with DeepPool and generate job
cs = CostSim()
cs.GeneralLayer(model, "model", {}, mustTrace=True)

jobParams = {}
jobParams["catsdogs"] = True

jobParams['evaluation_data'] = "/Data/catsDogs/test.csv"
jobParams['training_data'] = "/Data/catsDogs/train.csv"

# Register model with DeepPool and generate job
cs.printAllLayers(silent=True)
cs.computeInputDimensions(input_shape)
job, iterMs, gpuMs, maxGpusUsed = cs.searchBestSplitsV3(GPU_count, gloabal_batch_size)

# Push job to the cluster
cc = ClusterClient()
cc.submitTrainingJob("ResNet34", job.dumpInJSON(), jobParams=jobParams)