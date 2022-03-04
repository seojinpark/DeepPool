import torch
from torch import Tensor
import torch.nn as nn
# from .utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import os, sys
import time
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from parallelizationPlanner import CostSim
from parallelizationPlanner import GpuProfiler
from clusterClient import ClusterClient
from jobDescription import TrainingJob

def upSampling2d(size=None, scale_factor=None):
    upSample = nn.UpsamplingNearest2d(size, scale_factor)
    params = {"size": size, "scale_factor": scale_factor}
    cs.GeneralLayer(upSample, "upSample2d", params, mustTrace=False)
    return upSample

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        # Encoder        
        self.encoder1 = nn.Sequential(
            cs.Conv2d(9, 32, 3, padding=1),
            cs.ReLU(),
            cs.Conv2d(32, 32, 3, padding=1),
            cs.ReLU()
        )
        # skip1 = x
        skip1 = cs.layers[-1]

        self.encoder2 = nn.Sequential(
            cs.MaxPool2d(kernel_size=2, stride=2),
            cs.Conv2d(32, 64, 3, padding=1),
            cs.ReLU(),
            cs.Conv2d(64, 64, 3, padding=1),
            cs.ReLU()
        )
        # skip2 = x
        skip2 = cs.layers[-1]

        self.encoder3 = nn.Sequential(
            cs.MaxPool2d(kernel_size=2, stride=2),
            cs.Conv2d(64, 128, 3, padding=1),
            cs.ReLU(),
            cs.Conv2d(128, 128, 3, padding=1),
            cs.ReLU()
        )

        # skip3 = x
        skip3 = cs.layers[-1]
        
        self.bottleneck = nn.Sequential(
            cs.MaxPool2d(kernel_size=2, stride=2),
            cs.Conv2d(128, 256, 3, padding=1),
            cs.ReLU(),
            cs.Conv2d(256, 256, 3, padding=1),
            cs.ReLU()
        )

        # Decoder
        self.decoderUS1 = upSampling2d(scale_factor=2)
        cs.Concat([skip3, cs.layers[-1]])
        self.decoder1 = nn.Sequential(
            cs.Conv2d(384, 128, 3, padding=1),
            cs.ReLU(),
            cs.Conv2d(128, 128, 3, padding=1),
            cs.ReLU(),
            upSampling2d(scale_factor=2)
        )

        # x = cs.UpSampling2D(2)(x)
        # x = cs.Concatenate(axis=-1)([x, skip3])
        # x = cs.Conv2d(128, 3, padding=1)
        # x = cs.ReLU()
        # x = cs.Conv2d(128, 3, padding=1)
        # x = cs.ReLU()

        cs.Concat([skip2, cs.layers[-1]])
        self.decoder2 = nn.Sequential(
            cs.Conv2d(192, 64, 3, padding=1),
            cs.ReLU(),
            cs.Conv2d(64, 64, 3, padding=1),
            cs.ReLU(),
            upSampling2d(scale_factor=2)
        )

        # x = cs.UpSampling2D(2)(x)
        # x = cs.Concatenate(axis=-1)([x, skip2])
        # x = cs.Conv2d(64, 3, padding=1)
        # x = cs.ReLU()
        # x = cs.Conv2d(64, 3, padding=1)
        # x = cs.ReLU()

        cs.Concat([skip1, cs.layers[-1]])
        self.decoder3 = nn.Sequential(
            cs.Conv2d(96, 32, 3, padding=1),
            cs.ReLU(),
            cs.Conv2d(32, 32, 3, padding=1),
            cs.ReLU(),
            cs.Conv2d(32, 1, 1, padding=0)
        )

        # x = cs.UpSampling2D(2)(x)
        # x = cs.Concatenate(axis=-1)([x, skip1])
        # x = cs.Conv2d(32, 3, padding=1)
        # x = cs.ReLU()
        # x = cs.Conv2d(32, 3, padding=1)
        # x = cs.ReLU()

        # x = cs.Conv2d(1, 1, padding=1)


    def forward(self, x):
        x = self.encoder1(x)
        skip1 = x
        x = self.encoder2(x)
        skip2 = x
        x = self.encoder3(x)
        skip3 = x
        x = self.bottleneck(x)
        x = self.decoderUS1(x)
        x = torch.cat([x, skip3], dim=1)
        x = self.decoder1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.decoder2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.decoder3(x)
        return x
        

def main(gpuCount, globalBatch, amplificationLimit=2.0, dataParallelBaseline=False, netBw=2.66E5, spatialSplit=False, simResultFilename=None, simOnly=False, use_be=False):
    profiler = GpuProfiler("cuda")
    profiler.loadProfile()
    global cs
    cs = CostSim(profiler, netBw=netBw, verbose=True, gpuProfileLoc="profile/A100_wrn.prof")
    model = Unet()
    cs.printAllLayers(slient=True)
    cs.computeInputDimensions((9,512,512))
    # job, iterMs, gpuMs = cs.searchBestSplits(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)
    job, iterMs, gpuMs, maxGpusUsed = cs.searchBestSplitsV3(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)
    jobInJson = job.dumpInJSON()
    profiler.saveProfile()
    # for rank in range(4):
    #     print("GPU rank: %d"%rank)
    #     print(job.dumpSingleRunnableModule(rank))

    job2 = TrainingJob("test", None, None, 0, 0, "")
    job2.loadJSON(jobInJson)
    assert(jobInJson == job2.dumpInJSON())
    print("Load/Dump returned the same output? %s" % ("true" if jobInJson == job2.dumpInJSON() else "false"))
    # print(jobInJson)
    
    if not spatialSplit and not simOnly:
        cc = ClusterClient()
        jobName = "unet_%d_%d_%2.1f%s" % (gpuCount, globalBatch, amplificationLimit, "_DP" if dataParallelBaseline else "")
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


def runAllConfigs(modelName: str, clusterType: str, simOnly=True):
    if clusterType == "V100":
        netBw = 22937
    elif clusterType == "A100":
        netBw = 2.66E5
    elif clusterType == "B100":
        netBw = 2.66E5 * 5
    else:
        print("Wrong cluster type. Put either V100 or A100")

    gpuCounts = [1, 2, 4, 8]
    # gpuCounts = [1, 2, 4]
    globalBatchSize = 16
    # globalBatchSize = 16
    # globalBatchSize = 8
    limitAndBaseline = [(2.0, True, False), (99, False, False), (1.5, False, False), (2.0, False, False), (2.5, False, False)]
    # limitAndBaseline = [(99, False, True)]
    # limitAndBaseline = []
    for lim, baseline, spatialSplit in limitAndBaseline:
        simResultFilename = "%s_%s_b%d_lim%2.1f_sim.data" % (modelName, "DP" if baseline else "MP", globalBatchSize, lim)
        f = open(simResultFilename, "w")
        f.write("#batch GPUs IterMs  GpuMs\n")
        f.close()

        for gpuCount in gpuCounts:
            if not simOnly:
                preSize = os.stat('runtimeResult.data').st_size
            main(gpuCount, globalBatchSize, amplificationLimit=lim, dataParallelBaseline=baseline, netBw=netBw, spatialSplit=spatialSplit, simResultFilename=simResultFilename, simOnly=simOnly)
            # check exp finished.
            if not simOnly:
                print("runtimeResult.data's original size: ", preSize)
                while os.stat('runtimeResult.data').st_size == preSize and not spatialSplit:
                    time.sleep(10)
                print("runtimeResult.data's current size: ", os.stat('runtimeResult.data').st_size)
        
        if not spatialSplit and not simOnly:
            fw = open("%s_%s_b%d_lim%2.1f_run.data" % (modelName, "DP" if baseline else "MP", globalBatchSize, lim), "w")
            fr = open('runtimeResult.data', "r")
            fw.write("#batch GPUs IterMs  GpuMs\n")
            fw.write(fr.read())
            fw.close()
            fr.close()

        fr = open('runtimeResult.data', "w")
        fr.close()

def generateJit():
    global cs
    netBw = 2.66E5
    cs = CostSim(GpuProfiler("cuda"), netBw=netBw, verbose=False)

    fakeInputSize = (16,9,512,512)
    # fakeInput = torch.zeros(fakeInputSize)
    fakeInput = torch.randn(fakeInputSize)

    # model = resnet152()
    # traced = torch.jit.trace(model, fakeInput)
    # torch.jit.save(traced, "beModules/resnet152.jit")
    
    # model = resnext101_32x8d()
    # traced = torch.jit.trace(model, fakeInput)
    # torch.jit.save(traced, "beModules/resnext101_32x8d.jit")

    model = Unet()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters: ", pytorch_total_params)

    output = model.forward(fakeInput)
    print("Output dimensions: ", output.size())
    print("Output: ", output)

    traced = torch.jit.trace(model, fakeInput)
    torch.jit.save(traced, "beModules/unet.jit")


if __name__ == "__main__":
    print(len(sys.argv))
    if len(sys.argv) == 3:
        main(int(sys.argv[1]), int(sys.argv[2]), dataParallelBaseline=True)
    elif len(sys.argv) >= 4:
        use_be = len(sys.argv) > 4 and int(sys.argv[4]) == 1
        if sys.argv[3] == "DP":
            main(int(sys.argv[1]), int(sys.argv[2]), dataParallelBaseline=True, use_be=use_be)
        else:
            main(int(sys.argv[1]), int(sys.argv[2]), amplificationLimit=float(sys.argv[3]), use_be=use_be)
    elif len(sys.argv) == 2:
        print("Run all configs")
        runAllConfigs("unet", sys.argv[1])
    elif len(sys.argv) == 1:
        generateJit()
    else:
        print("Wrong number of arguments.\nUsage: ")
