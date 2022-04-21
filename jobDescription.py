# Copyright (c) 2021 MIT
# 
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR(S) DISCLAIM ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL AUTHORS BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import json
import torch
import io
import os
from os.path import exists
from typing import Optional, IO, List, Any

from collections import defaultdict
import copy
import sys
import hashlib

class TensorProperties:
    def __init__(self, tensor: torch.Tensor = None, batch_dim=0):
        self.props = {}
        if tensor is not None:
            self.props["tensor_shape"] = tuple(tensor.shape)
            self.props["dtype"] = str(tensor.dtype)
            self.props["batch_dim"] = batch_dim

    def __repr__(self):
        return json.dumps(self.props, sort_keys=True)

    def __str__(self):
        return self.__repr__()

    def fromStr(self, jstr):
        self.props = json.loads(jstr)

    def tensor_shape(self):
        return tuple(self.props["tensor_shape"])

    def tensor_shape_nobatch(self):
        assert self.batch_dim() == 0
        return self.tensor_shape()[1:]

    def batch_dim(self):
        return self.props["batch_dim"]

    def dtype(self):
        return eval(self.props["dtype"])

    def genRand(self, batchsize, device):
        shape = list(self.tensor_shape())
        shape = shape[:self.batch_dim()] + [batchsize] + shape[self.batch_dim() + 1:]
        return torch.zeros(shape, dtype=self.dtype()).to(device)

class Layer:
    def __init__(self, module, name: str, params: tuple, prevLayers: list, device: int = None):
        self.id = None      # Assigned later by calling printAllLayers.
        self.name = name
        self.params = params
        self.prevLayers = prevLayers
        for prevLayer in prevLayers or []:
            prevLayer.nextLayers.append(self)
        self.nextLayers = []
        self.module = module
        self.moduleSavedLocation = None
        self.losslayer = ""
        self.device = device
        self.initial_inputs = []
        self.distribute = False
        self.traced_once = False

        # self.inputDim = (0, 0, 0)   # (Channel, Width, Height) for 2d convolution
        # self.outputDim = (0, 0, 0)  # (Channel, Width, Height)
        self.must_trace = False

    def addInputShape(self, example_tensor_with_batch: torch.Tensor, *args, **kwargs):
        assert not self.prevLayers
        self.initial_inputs.append(TensorProperties(example_tensor_with_batch, *args, **kwargs))

    def getInputShapes(self) -> List[TensorProperties]:
        if not self.prevLayers:
            return self.initial_inputs

        shapes = []
        for layer in self.prevLayers:
            if layer.distribute:
                assert self.id is not None and layer.id is not None
                nextlayerids = sorted([l.id for l in layer.nextLayers])
                idx = nextlayerids.index(self.id)
                tothisI = layer.getOutputShapes()[idx]
                if type(tothisI) in [list, tuple]:
                    shapes += list(tothisI)
                else:
                    shapes.append(tothisI)
                continue
            shapes += layer.getOutputShapes()
        return shapes

    def getOutputShapes(self, device="cuda") -> List[TensorProperties]:
        if hasattr(self, "outputShapes"):
            return self.outputShapes

        out = self.scriptModule()(*self.getRandomInputs(1, device))
        if type(out) in [tuple, list]:
            self.outputShapes = [TensorProperties(t) for t in out]
        else:
            self.outputShapes = [TensorProperties(out)]

        # set inputDim and outputDim for backwards compatibility
        self.inputDim = self.getInputShapes()[0].tensor_shape_nobatch()
        if len(self.inputDim) == 1:
            self.inputDim = self.inputDim[0]
        self.outputDim = self.outputShapes[0].tensor_shape_nobatch()
        if len(self.outputDim) == 1:
            self.outputDim = self.outputDim[0]
        return self.outputShapes

    def getOutputShape(self, *args, **kwargs) -> TensorProperties:
        ret = self.getOutputShapes(*args, **kwargs)
        assert len(ret) == 1
        return ret[0]

    def getParameters(self):
        pass

    def getRandomInputs(self, batchsize, device="cuda"):
        return [shape.genRand(batchsize, device) for shape in self.getInputShapes()]

    def getModuleId(self):
        if hasattr(self.module, 'weight'):
            h = hashlib.md5()
            h.update(self.module.weight.cpu().detach().numpy())
            return self.name +\
                json.dumps(self.params, sort_keys=True, separators=('_', '-'))+str(h.hexdigest())
        else:
            return self.name +\
                json.dumps(self.params, sort_keys=True, separators=('_', '-'))

    def scriptModule(self):
        if not self.moduleSavedLocation:
            moduleId = self.getModuleId()
            saveLocation = os.getcwd() + f"/modules/scriptmodule_{moduleId}.pt"
            self.moduleSavedLocation = saveLocation
        if not self.module or self.traced_once:
            return torch.jit.load(self.moduleSavedLocation).to("cuda")

        self.traced_once = True

        fakeInput = self.getRandomInputs(1, "cpu")
        if self.must_trace:
            print("jit tracing...", self.name)
            traced = torch.jit.trace(self.module.to("cpu"), fakeInput)
        else:
            print("jit scripting...", self.name)
            traced = torch.jit.script(self.module, fakeInput)

        # saveLocation = "modules/scriptmodule_%d.pt"%self.id
        torch.jit.save(traced, self.moduleSavedLocation)
        return torch.load(self.moduleSavedLocation).to("cuda")

    def getInitialConfig(self, globalBatch: int):
        inputDim = self.getInputShapes()[0].tensor_shape_nobatch()
        if self.device is not None:
            initCfg = (globalBatch, *inputDim)
        elif self.name in ["conv2d"]:
            outputDim = self.getOutputShape().tensor_shape_nobatch()
            initCfg = (globalBatch, inputDim[1], inputDim[2], inputDim[0], outputDim[2]) # (batch, width, height, channel, filter)
        elif self.name in ["linear", "ReLU1d"]:
            outputDim = self.getOutputShape().tensor_shape_nobatch()
            initCfg = (globalBatch, *inputDim, *outputDim)
        elif self.name in ["flatten", "maxPool2d", "avgPool2d", "adAvgPool2d", "ReLU2d", "concat"]:
            initCfg = (globalBatch, inputDim[1], inputDim[2], inputDim[0]) # (batch, width, height, channel, filter)
        else:
            initCfg = (globalBatch, *inputDim) # (batch, width, height, channel)
        return initCfg

    def dumpForJSON(self):
        prop = {}
        if self.id == None:
            raise Exception("layer is not yet initialized.")

        prop["id"] = self.id
        prop["name"] = self.name
        prop["params"] = self.params
        prop["gpuTime"] = getattr(self, "gpuTime", (1,1))
        prop["prevLayers"] = []
        if self.prevLayers != None:
            for prevLayer in self.prevLayers:
                prop["prevLayers"].append(prevLayer.id)
        prop["prevLayers"] = sorted(prop["prevLayers"])
        prop["nextLayers"] = []
        if self.nextLayers != None:
            for nextLayer in self.nextLayers:
                prop["nextLayers"].append(nextLayer.id)
        prop["nextLayers"] = sorted(prop["nextLayers"])
        prop["inputDim"] = self.inputDim
        prop["outputDim"] = self.outputDim
        if hasattr(self, 'gpuAssignment'):
            prop["gpuAssignment"] = self.gpuAssignment

        if self.moduleSavedLocation:
            prop["moduleSavedLocation"] = self.moduleSavedLocation
        elif self.module != None:
            self.scriptModule()
            prop["moduleSavedLocation"] = saveLocation

        if not self.prevLayers:
            prop["initial_inputs"] = [str(a) for a in self.initial_inputs]

        prop["distribute"] = self.distribute
        return prop


class TrainingJob:
    def __init__(self, name: str, layers: List[Layer], layerConfigs: List[tuple], globalBatchSize: int, maxGpusUsed: int, datasetDir: str):
        self.name = name
        self.layers = layers
        self.layerConfigs = layerConfigs
        self.globalBatchSize = globalBatchSize
        self.maxGpusUsed = maxGpusUsed
        self.datasetDir = datasetDir
        self.bytesPerParam = 4
        self.initialBatchSizes = None
        self.sampleIndicesList = None
        self.perRankConfigCache = []
        self.losslayer = None
    
    def loadJSON(self, jobInJson: str):
        job = json.loads(jobInJson)
        self.globalBatchSize = job["globalBatchSize"]
        self.maxGpusUsed = job["maxGpusUsed"]
        self.layers = []
        self.layerConfigs = []
        def loadlayer(ldsc):
            # print(ldsc)
            prevLayers = [self.layers[prevLayerId] for prevLayerId in ldsc["prevLayers"]]
            l = Layer(None, ldsc["name"], ldsc["params"], prevLayers)
            if 'gpuTime' in ldsc:
                l.gpuTime = ldsc["gpuTime"]
            l.id = ldsc["id"]
            # l.nextLayers = ldsc["nextLayers"]
            l.inputDim = ldsc["inputDim"]
            l.outputDim = ldsc["outputDim"]
            if 'gpuAssignment' in ldsc:
                l.gpuAssignment = ldsc["gpuAssignment"]
            l.bestCfg = ldsc["config"]
            if 'moduleSavedLocation' in ldsc:
                l.moduleSavedLocation = ldsc["moduleSavedLocation"]
            for inputdesc in ldsc.get('initial_inputs', []):
                prop = TensorProperties()
                prop.fromStr(inputdesc)
                l.initial_inputs.append(prop)
            if "distribute" in ldsc:
                l.distribute = ldsc["distribute"]
            return l
        for ldsc in job["layers"]:
            l = loadlayer(ldsc)
            config = ldsc["config"]
            self.layers.append(l)
            self.layerConfigs.append(config)
    
        if job.get("losslayer"):
            self.losslayer = loadlayer(job["losslayer"])

    def getGpusUsed(self):
        return self.maxGpusUsed

    def dumpInJSON(self, layers: List[Layer] = None, layerConfigs: list = None):
        if layers is None:
            layers = self.layers
        if layerConfigs is None:
            layerConfigs = self.layerConfigs

        allProps = []
        for l, config in zip(layers, layerConfigs):
            prop = l.dumpForJSON()
            prop["config"] = config
            allProps.append(prop)
        fullDesc = {"globalBatchSize": self.globalBatchSize, "maxGpusUsed": self.maxGpusUsed, "layers": allProps}
        if self.losslayer is not None:
            self.losslayer.id = -1
            prop = self.losslayer.dumpForJSON()
            prop["config"] = config # reuse last config
            fullDesc["losslayer"] = prop
        # return json.dumps(fullDesc, indent=1, sort_keys=False)
        return json.dumps(fullDesc, sort_keys=False)

    def dumpSingleRunnableModule(self, targetRank: int) -> str: # Only supports DP now.
        if len(self.perRankConfigCache) == 0:
            self.perRankConfigCache = [self.dumpSingleRunnableModuleHelper(rank) for rank in range(self.getGpusUsed())]
        fullDesc = self.perRankConfigCache[targetRank]
        fullDesc["initialBatchSizes"] = self.initialBatchSizes
        fullDesc["sampleIndices"] = self.sampleIndicesList[targetRank]
        dumpedStr = json.dumps(fullDesc, sort_keys=False)
        return dumpedStr

    def computeXfers(self):

        if getattr(self, "xferSamplesDone", False):
            return

        self.xferSamplesDone = True

        initBSize = self.layers[0].bestCfg[0]
        initAssigned = self.layers[0].gpuAssignment
        self.initialBatchSizes = [initBSize if g in initAssigned else 0 for g in range(self.getGpusUsed())]

        totalSamples = 0
        xferCounter = 0

        self.all_xfers = []

        for l in self.layers:

            l.byGpu = defaultdict(list)
            l.bySample = {}
            if not isinstance(l.gpuAssignment, list):
                l.gpuAssignmen = [l.gpuAssignmen]

            if not l.prevLayers: # 1st layer.
                totalSamples = 0
                for i in range(self.getGpusUsed()):
                    if i not in l.gpuAssignment: continue
                    for _ in range(l.bestCfg[0]):
                        l.byGpu[i].append(totalSamples)
                        l.bySample[totalSamples] = i
                        totalSamples += 1
                continue

            # find a prevLayer with a matching config for matching sample assignment
            found = False
            for prevLayer in l.prevLayers:
                if set(prevLayer.gpuAssignment) == set(l.gpuAssignment):
                    assert prevLayer.bestCfg[0] == l.bestCfg[0]
                    l.byGpu = copy.deepcopy(prevLayer.byGpu)
                    l.bySample = copy.deepcopy(prevLayer.bySample)
                    found = True
                    break

            # if no matching prevlayer, try to reassign samples, minimizing bandwidth
            if not found:
                assert(len(l.byGpu) == 0 and len(l.bySample) == 0)
                lastGpuAssigned = 0
                def tryAssign(idx, gpu):
                    nonlocal lastGpuAssigned
                    assert idx not in l.bySample
                    if gpu in l.gpuAssignment and len(l.byGpu[gpu]) < l.bestCfg[0]:
                        l.bySample[idx] = gpu
                        l.byGpu[gpu].append(idx)
                        lastGpuAssigned = gpu
                        return True
                    return False

                for i in range(totalSamples):
                    if tryAssign(i, prevLayer.bySample[i]):
                        continue

                    if tryAssign(i, lastGpuAssigned):
                        continue

                    for gpu in range(self.getGpusUsed()):
                        if tryAssign(i, gpu):
                            break
                    assert i in l.bySample

            for prevLayer in l.prevLayers:

                if l.bySample == prevLayer.bySample:
                    continue

                # TODO: find a way to make sure that we dont have to shuffle samples when GPU assignments dont change
                # assert set(prevLayer.gpuAssignment) != set(l.gpuAssignment)

                xfer_to_from_pairs = defaultdict(lambda: defaultdict(list))

                # Arrange by RX
                lastSrcG, lastDstG = prevLayer.bySample[0], l.bySample[0]
                rangeStart = 0
                for i in range(1, totalSamples):
                    curSrcG = prevLayer.bySample[i]
                    curDstG = l.bySample[i]
                    if curSrcG == lastSrcG and curDstG == lastDstG: continue
                    xfer_to_from_pairs[lastDstG][lastSrcG].append((rangeStart, i - rangeStart))
                    lastSrcG = curSrcG
                    lastDstG = curDstG
                    rangeStart = i
                xfer_to_from_pairs[lastDstG][lastSrcG].append((rangeStart, totalSamples - rangeStart))

                # TODO: pickout special scatters/gathers someday?

                for receiver, srcs in xfer_to_from_pairs.items():
                    for src, samples in srcs.items():
                        for sample_start, nr_sample in samples:
                            assert sorted(prevLayer.byGpu[src]) == prevLayer.byGpu[src]
                            assert sorted(l.byGpu[src]) == l.byGpu[src]
                            rxOffset = l.byGpu[receiver].index(sample_start)
                            txOffset = prevLayer.byGpu[src].index(sample_start)
                            xfer = {
                                "name": f"{l.id}_from_{prevLayer.id}_{receiver}:{rxOffset}_{src}:{txOffset}_sample_{xferCounter}",
                                "prop": {
                                    "rxSampleOffset": rxOffset,
                                    "txSampleOffset": txOffset,
                                    "xferSamples": nr_sample,
                                    "prevLayerId": prevLayer.id,
                                    "nextLayerId": l.id,
                                },
                                "dest": receiver,
                                "src": src,
                                "bytes": 1 # fix
                            }
                            self.all_xfers.append(xfer)
                            xferCounter += 1

                for i in range(self.getGpusUsed()):
                    for j in range(self.getGpusUsed()):
                        if not xfer_to_from_pairs[i][j]: continue


        self.sampleIndicesList = []
        for i in range(self.getGpusUsed()):
            self.sampleIndicesList.append(sorted(self.layers[-1].byGpu[i]))

    def dumpSingleRunnableModuleHelper(self, targetRank: int) -> str:

        self.computeXfers()
        allProps = []
        for l in self.layers:
            if not self.isConfigDataParallelOnly(l, l.bestCfg, self.globalBatchSize):
                assert False, "WHAT IS THIS?"
                print("[dumpSingleRunnableModule] config was not DP-only.")
                return None
            prop = l.dumpForJSON()
            cfg = list(l.bestCfg)
            if targetRank not in l.gpuAssignment:
                cfg[0] = 0
            prop["config"] = tuple(cfg)
            allProps.append(prop)

        for xfer in self.all_xfers:
            if xfer["src"] == targetRank or xfer["dest"] == targetRank:
                d = allProps[xfer["prop"]["nextLayerId"]]
                if not d.get("xfers"):
                    d["xfers"] = []
                d["xfers"].append(xfer)

        fullDesc = {"rank": targetRank,
                    "maxGpusUsed": self.maxGpusUsed,
                    "globalBatchSize": self.globalBatchSize,
                    "layers": allProps}
        if self.losslayer:
            lastlayer = self.layers[-1]
            cfg = list(lastlayer.bestCfg)
            if targetRank not in lastlayer.gpuAssignment:
                cfg[0] = 0
            fullDesc["losslayer"] = self.losslayer.dumpForJSON()
            fullDesc["losslayer"]["config"] = cfg
        return fullDesc

    def calcGpusNeeded(self, layer: Layer, config: tuple, globalBatch: int):
        initCfg = layer.getInitialConfig(globalBatch)
        gpuCount = 1
        # if len(config) != len(initCfg):
        #     print("[calcGpusNeeded] dimension of configs doesn't match!! %20s layer len(config):%d != len(initCfg):%d" % (layer.name, len(config), len(initCfg)))
        for i in range(len(initCfg)):
            gpuCount *= int(initCfg[i] / config[i])
        return gpuCount

    def isConfigDataParallelOnly(self, layer: Layer, config: tuple, globalBatch: int):
        initCfg = layer.getInitialConfig(globalBatch)
        dpOnly = True
        for i in range(1, len(config)):
            if config[i] != initCfg[i]:
                dpOnly = True
        return dpOnly

def test():
    return

if __name__ == "__main__":
    test()
