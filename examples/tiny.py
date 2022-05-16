import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import threading
import os, sys
import time

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from parallelizationPlanner import CostSim
from clusterClient import ClusterClient
from jobDescription import TrainingJob

import torch.backends.cudnn as cudnn
cudnn.benchmark = True


class TinyModel(nn.Module):

    def __init__(self, input_size=2048, output_size=1000) -> None:
        super(TinyModel, self).__init__()
        self.layer1 = cs.Linear(int(input_size), int(output_size))

    def forward(self, x):
        x = self.layer1(x)
        return x


def tiny(**kwargs):
    """Tiny 1-layer model

    Args:
    """
    model = TinyModel(**kwargs)
    return model


def main(gpuCount, globalBatch, amplificationLimit=2.0, dataParallelBaseline=False, netBw=2.66E5, spatialSplit=False, simResultFilename=None, use_be=False):
    global cs
    cs = CostSim(None, netBw=netBw, verbose=False, gpuProfileLoc="profile/A100_vgg.prof")
    model = tiny()

    cs.printAllLayers(silent=True)
    cs.computeInputDimensions((2048, ))

    saveWholeModel = False
    if saveWholeModel:
        model.train()
        model.cuda()
        fakeInput = torch.randn(cs.layers[0].inputDim).unsqueeze(0).cuda()
        traced = torch.jit.trace(model, fakeInput)
        saveLocation = "modules/vgg16.pt"
        torch.jit.save(traced, saveLocation)
        print("Saved whole model to %s" % saveLocation)
        exit(0)

    job, iterMs, gpuMs, maxGpusUsed = cs.searchBestSplitsV3(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)
    print("Searching for parallelization strategy is completed.\n")
    cs.to_dot("Digraph", globalBatch)
    cs.to_gpuTimeline("TinyModel, Data Parallel" if dataParallelBaseline else "TinyModel, Burst Parallel", maxGpusUsed, dataParallelBaseline, xlim=11)

    jobInJson = job.dumpInJSON()

    job2 = TrainingJob("test", None, None, 0, 0, "")
    job2.loadJSON(jobInJson)
    assert(jobInJson == job2.dumpInJSON())
    print("Load/Dump returned the same output? %s" % ("true" if jobInJson == job2.dumpInJSON() else "false"))

    if not spatialSplit:
        cc = ClusterClient()
        jobName = "tiny_%d_%d_%2.1f%s" % (gpuCount, globalBatch, amplificationLimit, "_DP" if dataParallelBaseline else "")
        jobName += "_BE" if use_be else ""
        cc.submitTrainingJob(jobName, jobInJson, use_be)

    if simResultFilename != None:
        f = open(simResultFilename, "a")
        f.write("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, gpuCount, iterMs, gpuMs))
        f.close()

        if gpuCount == 8:
            f = open(simResultFilename, "r")
            print(f.read())
            f.close()

        
if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 3:
        main(int(sys.argv[1]), int(sys.argv[2]), dataParallelBaseline=True)
    else:
        print("Wrong number of arguments.\nUsage: ")
