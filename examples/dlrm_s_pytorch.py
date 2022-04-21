# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: an implementation of a deep learning recommendation model (DLRM)
# The model input consists of dense and sparse features. The former is a vector
# of floating point values. The latter is a list of sparse indices into
# embedding tables, which consist of vectors of floating point values.
# The selected vectors are passed to mlp networks denoted by triangles,
# in some cases the vectors are interacted through operators (Ops).
#
# output:
#                         vector of values
# model:                        |
#                              /\
#                             /__\
#                               |
#       _____________________> Op  <___________________
#     /                         |                      \
#    /\                        /\                      /\
#   /__\                      /__\           ...      /__\
#    |                          |                       |
#    |                         Op                      Op
#    |                    ____/__\_____           ____/__\____
#    |                   |_Emb_|____|__|    ...  |_Emb_|__|___|
# input:
# [ dense features ]     [sparse indices] , ..., [sparse indices]
#
# More precise definition of model layers:
# 1) fully connected layers of an mlp
# z = f(y)
# y = Wx + b
#
# 2) embedding lookup (for a list of sparse indices p=[p1,...,pk])
# z = Op(e1,...,ek)
# obtain vectors e1=E[:,p1], ..., ek=E[:,pk]
#
# 3) Operator Op can be one of the following
# Sum(e1,...,ek) = e1 + ... + ek
# Dot(e1,...,ek) = [e1'e1, ..., e1'ek, ..., ek'e1, ..., ek'ek]
# Cat(e1,...,ek) = [e1', ..., ek']'
# where ' denotes transpose operation
#
# References:
# [1] Maxim Naumov, Dheevatsa Mudigere, Hao-Jun Michael Shi, Jianyu Huang,
# Narayanan Sundaram, Jongsoo Park, Xiaodong Wang, Udit Gupta, Carole-Jean Wu,
# Alisson G. Azzolini, Dmytro Dzhulgakov, Andrey Mallevich, Ilia Cherniavskii,
# Yinghai Lu, Raghuraman Krishnamoorthi, Ansha Yu, Volodymyr Kondratenko,
# Stephanie Pereira, Xianjie Chen, Wenlin Chen, Vijay Rao, Bill Jia, Liang Xiong,
# Misha Smelyanskiy, "Deep Learning Recommendation Model for Personalization and
# Recommendation Systems", CoRR, arXiv:1906.00091, 2019

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

# miscellaneous
import builtins
import datetime
import json
import sys
import time
import bisect
from collections import deque

# onnx
# The onnx import causes deprecation warnings every time workers
# are spawned during testing. So, we filter out those warnings.
import warnings

# data generation
# import dlrm_data_pytorch as dp

# For distributed run
# import extend_distributed as ext_dist
# import mlperf_logger

# numpy
import numpy as np
from numpy import random as ra
import sklearn.metrics

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, RandomSampler
from torch._ops import ops
from torch.autograd.profiler import record_function
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.scatter_gather import gather, scatter
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler
# import optim.rwsadagrad as RowWiseSparseAdagrad
from torch.utils.tensorboard import SummaryWriter

# mixed-dimension trick
# from tricks.md_embedding_bag import PrEmbeddingBag, md_solver

# quotient-remainder trick
# from tricks.qr_embedding_bag import QREmbeddingBag

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    try:
        import onnx
    except ImportError as error:
        print("Unable to import onnx. ", error)

# from torchviz import make_dot
# import torch.nn.functional as Functional
# from torch.nn.parameter import Parameter
import os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
# sys.path.append('.')
from parallelizationPlanner import CostSim
from parallelizationPlanner import GpuProfiler
from clusterClient import ClusterClient
from jobDescription import TrainingJob


exc = getattr(builtins, "IOError", "FileNotFoundError")


do_embed_layer_mp = False #True
bbn_max_gpus = None



# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Mixed-Dimensions Trick
#
# Description: Applies mixed dimension trick to embeddings to reduce
# embedding sizes.
#
# References:
# [1] Antonio Ginart, Maxim Naumov, Dheevatsa Mudigere, Jiyan Yang, James Zou,
# "Mixed Dimension Embeddings with Application to Memory-Efficient Recommendation
# Systems", CoRR, arXiv:1909.11810, 2019
def md_solver(n, alpha, d0=None, B=None, round_dim=True, k=None):
    '''
    An external facing function call for mixed-dimension assignment
    with the alpha power temperature heuristic
    Inputs:
    n -- (torch.LongTensor) ; Vector of num of rows for each embedding matrix
    alpha -- (torch.FloatTensor); Scalar, non-negative, controls dim. skew
    d0 -- (torch.FloatTensor); Scalar, baseline embedding dimension
    B -- (torch.FloatTensor); Scalar, parameter budget for embedding layer
    round_dim -- (bool); flag for rounding dims to nearest pow of 2
    k -- (torch.LongTensor) ; Vector of average number of queries per inference
    '''
    n, indices = torch.sort(n)
    k = k[indices] if k is not None else torch.ones(len(n))
    d = alpha_power_rule(n.type(torch.float) / k, alpha, d0=d0, B=B)
    if round_dim:
        d = pow_2_round(d)
    undo_sort = [0] * len(indices)
    for i, v in enumerate(indices):
        undo_sort[v] = i
    return d[undo_sort]


def alpha_power_rule(n, alpha, d0=None, B=None):
    if d0 is not None:
        lamb = d0 * (n[0].type(torch.float) ** alpha)
    elif B is not None:
        lamb = B / torch.sum(n.type(torch.float) ** (1 - alpha))
    else:
        raise ValueError("Must specify either d0 or B")
    d = torch.ones(len(n)) * lamb * (n.type(torch.float) ** (-alpha))
    for i in range(len(d)):
        if i == 0 and d0 is not None:
            d[i] = d0
        else:
            d[i] = 1 if d[i] < 1 else d[i]
    return (torch.round(d).type(torch.long))


def pow_2_round(dims):
    return 2 ** torch.round(torch.log2(dims.type(torch.float)))


class PrEmbeddingBag(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, base_dim):
        super(PrEmbeddingBag, self).__init__()
        self.embs = cs.EmbeddingBag(
            num_embeddings, embedding_dim, mode="sum", sparse=True)
        torch.nn.init.xavier_uniform_(self.embs.weight)
        if embedding_dim < base_dim:
            self.proj = cs.Linear(embedding_dim, base_dim, bias=False)
            torch.nn.init.xavier_uniform_(self.proj.weight)
        elif embedding_dim == base_dim:
            self.proj = cs.Identity()
        else:
            raise ValueError(
                "Embedding dim " + str(embedding_dim) + " > base dim " + str(base_dim)
            )

    def forward(self, input, offsets=None, per_sample_weights=None):
        return self.proj(self.embs(
            input, offsets=offsets, per_sample_weights=per_sample_weights))
# end mixed-dimension trick







# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Quotient-Remainder Trick
#
# Description: Applies quotient remainder-trick to embeddings to reduce
# embedding sizes.
#
# References:
# [1] Hao-Jun Michael Shi, Dheevatsa Mudigere, Maxim Naumov, Jiyan Yang,
# "Compositional Embeddings Using Complementary Partitions for Memory-Efficient
# Recommendation Systems", CoRR, arXiv:1909.02107, 2019

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np


class QREmbeddingBag(nn.Module):
    r"""Computes sums or means over two 'bags' of embeddings, one using the quotient
    of the indices and the other using the remainder of the indices, without
    instantiating the intermediate embeddings, then performs an operation to combine these.

    For bags of constant length and no :attr:`per_sample_weights`, this class

        * with ``mode="sum"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.sum(dim=0)``,
        * with ``mode="mean"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.mean(dim=0)``,
        * with ``mode="max"`` is equivalent to :class:`~torch.nn.Embedding` followed by ``torch.max(dim=0)``.

    However, :class:`~torch.nn.EmbeddingBag` is much more time and memory efficient than using a chain of these
    operations.

    QREmbeddingBag also supports per-sample weights as an argument to the forward
    pass. This scales the output of the Embedding before performing a weighted
    reduction as specified by ``mode``. If :attr:`per_sample_weights`` is passed, the
    only supported ``mode`` is ``"sum"``, which computes a weighted sum according to
    :attr:`per_sample_weights`.

    Known Issues:
    Autograd breaks with multiple GPUs. It breaks only with multiple embeddings.

    Args:
        num_categories (int): total number of unique categories. The input indices must be in
                              0, 1, ..., num_categories - 1.
        embedding_dim (list): list of sizes for each embedding vector in each table. If ``"add"``
                              or ``"mult"`` operation are used, these embedding dimensions must be
                              the same. If a single embedding_dim is used, then it will use this
                              embedding_dim for both embedding tables.
        num_collisions (int): number of collisions to enforce.
        operation (string, optional): ``"concat"``, ``"add"``, or ``"mult". Specifies the operation
                                      to compose embeddings. ``"concat"`` concatenates the embeddings,
                                      ``"add"`` sums the embeddings, and ``"mult"`` multiplies
                                      (component-wise) the embeddings.
                                      Default: ``"mult"``
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (string, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 ``"sum"`` computes the weighted sum, taking :attr:`per_sample_weights`
                                 into consideration. ``"mean"`` computes the average of the values
                                 in the bag, ``"max"`` computes the max value over each bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` matrix will be a sparse tensor. See
                                 Notes for more details regarding sparse gradients. Note: this option is not
                                 supported when ``mode="max"``.

    Attributes:
        weight (Tensor): the learnable weights of each embedding table is the module of shape
                         `(num_embeddings, embedding_dim)` initialized using a uniform distribution
                         with sqrt(1 / num_categories).

    Inputs: :attr:`input` (LongTensor), :attr:`offsets` (LongTensor, optional), and
        :attr:`per_index_weights` (Tensor, optional)

        - If :attr:`input` is 2D of shape `(B, N)`,

          it will be treated as ``B`` bags (sequences) each of fixed length ``N``, and
          this will return ``B`` values aggregated in a way depending on the :attr:`mode`.
          :attr:`offsets` is ignored and required to be ``None`` in this case.

        - If :attr:`input` is 1D of shape `(N)`,

          it will be treated as a concatenation of multiple bags (sequences).
          :attr:`offsets` is required to be a 1D tensor containing the
          starting index positions of each bag in :attr:`input`. Therefore,
          for :attr:`offsets` of shape `(B)`, :attr:`input` will be viewed as
          having ``B`` bags. Empty bags (i.e., having 0-length) will have
          returned vectors filled by zeros.

        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
            to indicate all weights should be taken to be ``1``. If specified, :attr:`per_sample_weights`
            must have exactly the same shape as input and is treated as having the same
            :attr:`offsets`, if those are not ``None``. Only supported for ``mode='sum'``.


    Output shape: `(B, embedding_dim)`

    """
    __constants__ = ['num_categories', 'embedding_dim', 'num_collisions',
                     'operation', 'max_norm', 'norm_type', 'scale_grad_by_freq',
                     'mode', 'sparse']

    def __init__(self, num_categories, embedding_dim, num_collisions,
                 operation='mult', max_norm=None, norm_type=2.,
                 scale_grad_by_freq=False, mode='mean', sparse=False,
                 _weight=None):
        super(QREmbeddingBag, self).__init__()

        assert operation in ['concat', 'mult', 'add'], 'Not valid operation!'

        self.num_categories = num_categories
        if isinstance(embedding_dim, int) or len(embedding_dim) == 1:
            self.embedding_dim = [embedding_dim, embedding_dim]
        else:
            self.embedding_dim = embedding_dim
        self.num_collisions = num_collisions
        self.operation = operation
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq

        if self.operation == 'add' or self.operation == 'mult':
            assert self.embedding_dim[0] == self.embedding_dim[1], \
                'Embedding dimensions do not match!'

        self.num_embeddings = [int(np.ceil(num_categories / num_collisions)),
            num_collisions]

        if _weight is None:
            self.weight_q = Parameter(torch.Tensor(self.num_embeddings[0], self.embedding_dim[0]))
            self.weight_r = Parameter(torch.Tensor(self.num_embeddings[1], self.embedding_dim[1]))
            self.reset_parameters()
        else:
            assert list(_weight[0].shape) == [self.num_embeddings[0], self.embedding_dim[0]], \
                'Shape of weight for quotient table does not match num_embeddings and embedding_dim'
            assert list(_weight[1].shape) == [self.num_embeddings[1], self.embedding_dim[1]], \
                'Shape of weight for remainder table does not match num_embeddings and embedding_dim'
            self.weight_q = Parameter(_weight[0])
            self.weight_r = Parameter(_weight[1])
        self.mode = mode
        self.sparse = sparse

    def reset_parameters(self):
        nn.init.uniform_(self.weight_q, np.sqrt(1 / self.num_categories))
        nn.init.uniform_(self.weight_r, np.sqrt(1 / self.num_categories))

    def forward(self, input, offsets=None, per_sample_weights=None):
        input_q = (input / self.num_collisions).long()
        input_r = torch.remainder(input, self.num_collisions).long()

        embed_q = F.embedding_bag(input_q, self.weight_q, offsets, self.max_norm,
                                  self.norm_type, self.scale_grad_by_freq, self.mode,
                                  self.sparse, per_sample_weights)
        embed_r = F.embedding_bag(input_r, self.weight_r, offsets, self.max_norm,
                                  self.norm_type, self.scale_grad_by_freq, self.mode,
                                  self.sparse, per_sample_weights)

        if self.operation == 'concat':
            embed = torch.cat((embed_q, embed_r), dim=1)
        elif self.operation == 'add':
            embed = embed_q + embed_r
        elif self.operation == 'mult':
            embed = embed_q * embed_r

        return embed

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        s += ', mode={mode}'
        return s.format(**self.__dict__)
# end quotient-remainder trick










def time_wrap(use_gpu):
    if use_gpu:
        torch.cuda.synchronize()
    return time.time()


def dlrm_wrap(X, lS_o, lS_i, use_gpu, device, ndevices=1):
    with record_function("DLRM forward"):
        if use_gpu:  # .cuda()
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            if ndevices == 1:
                lS_i = (
                    [S_i.to(device) for S_i in lS_i]
                    if isinstance(lS_i, list)
                    else lS_i.to(device)
                )
                lS_o = (
                    [S_o.to(device) for S_o in lS_o]
                    if isinstance(lS_o, list)
                    else lS_o.to(device)
                )
        return dlrm(X.to(device), lS_o, lS_i)


def loss_fn_wrap(Z, T, use_gpu, device):
    with record_function("DLRM loss compute"):
        if args.loss_function == "mse" or args.loss_function == "bce":
            return dlrm.loss_fn(Z, T.to(device))
        elif args.loss_function == "wbce":
            loss_ws_ = dlrm.loss_ws[T.data.view(-1).long()].view_as(T).to(device)
            loss_fn_ = dlrm.loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            return loss_sc_.mean()

def loss_fn_wrap_nodevice(Z, T):
    with record_function("DLRM loss compute"):
        if args.loss_function == "mse" or args.loss_function == "bce":
            return dlrm.loss_fn(Z, T)
        elif args.loss_function == "wbce":
            loss_ws_ = dlrm.loss_ws[T.data.view(-1).long()].view_as(T)
            loss_fn_ = dlrm.loss_fn(Z, T)
            loss_sc_ = loss_ws_ * loss_fn_
            return loss_sc_.mean()


# The following function is a wrapper to avoid checking this multiple times in th
# loop below.
def unpack_batch(b):
    # Experiment with unweighted samples
    return b[0], b[1], b[2], b[3], torch.ones(b[3].size()), None


class LRPolicyScheduler(_LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, decay_start_step, num_decay_steps):
        self.num_warmup_steps = num_warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_start_step + num_decay_steps
        self.num_decay_steps = num_decay_steps

        if self.decay_start_step < self.num_warmup_steps:
            sys.exit("Learning rate warmup must finish before the decay starts")

        super(LRPolicyScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_count = self._step_count
        if step_count < self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        elif self.decay_start_step <= step_count and step_count < self.decay_end_step:
            # decay
            decayed_steps = step_count - self.decay_start_step
            scale = ((self.num_decay_steps - decayed_steps) / self.num_decay_steps) ** 2
            min_lr = 0.0000001
            lr = [max(min_lr, base_lr * scale) for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            if self.num_decay_steps > 0:
                # freeze at last, either because we're after decay
                # or because we're between warmup and decay
                lr = self.last_lr
            else:
                # do not adjust
                lr = self.base_lrs
        return lr

"""
This super hacky module helps get fake data into the right place in the runtime.
Currently the runtime supports a single data Tensor as input to the model, and
only supports a single "first" node. The runtime's dataset generator generates
the dense features, and embedding indexes/offsets in a single tensor per sample;
this input node unpacks the data and forwards the relevant parts to the relevant
operators.
"""

class FakeDataDistributionNode(nn.Module):
    def __init__(self, dense_features, nl_bags, average_lS_i_len):
        super(FakeDataDistributionNode, self).__init__()
        self.dense_features = dense_features
        self.nl_bags = nl_bags
        self.average_lS_i_len = average_lS_i_len

    def forward(self, input_data):
        # [Batch Dim, DATA]
        # DATA := [nr_dense_features] + [nr_bags] + [nr_bags * average_lS_i_len]
        dense_x = input_data[:, :self.dense_features]
        offsets_by_batch = input_data[:, self.dense_features:self.dense_features + self.nl_bags]
        offsets = torch.transpose(offsets_by_batch, 0, 1).contiguous().unsqueeze(2).to(torch.int32)

        idx = input_data[:, self.dense_features + self.nl_bags:]
        idx = idx.view(-1, self.nl_bags, self.average_lS_i_len)
        idx = torch.transpose(idx, 0, 1).contiguous().to(torch.int32) # TODO: should these be int64

        tempcat = torch.cat([offsets, idx], 2)

        return [dense_x] + list(tempcat.split([1] * self.nl_bags))

"""
This class unpacks the offsets and indexes for each sample for the embedding bag
"""
class EmbeddingRepackWrap(nn.Module):
    def __init__(self, bag):
        super(EmbeddingRepackWrap, self).__init__()
        self.bag = bag

    def forward(self, input_data):
        # [1, Batch, [offset, indexes]]
        idata = input_data.squeeze(0)
        offsets = torch.flatten(idata[:, :1])
        indexes = torch.flatten(idata[:, 1:])
        return self.bag(indexes, offsets)

class DLRMDotModule(nn.Module):
    def __init__(self, arch_interaction_itself: bool = False, nr_emb: int = 1):
        super(DLRMDotModule, self).__init__()
        self.arch_interaction_itself = arch_interaction_itself

        # cache the tensors used for the index below on the device
        ni, nj = nr_emb + 1, nr_emb + 1
        offset = 1 if self.arch_interaction_itself else 0
        li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
        lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
        self.register_buffer("li", li)
        self.register_buffer("lj", lj)

    def forward(self, *inputList):
        x=inputList[0]
        ly=list(inputList[1:])
        (batch_size, d) = x.shape
        T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
        # perform a dot product
        Z = torch.bmm(T, torch.transpose(T, 1, 2))
        # append dense feature with the interactions (into a row vector)
        # approach 1: all
        # Zflat = Z.view((batch_size, -1))
        # approach 2: unique
        # _, ni, nj = Z.shape
        # approach 1: tril_indices
        # offset = 0 if self.arch_interaction_itself else -1
        # li, lj = torch.tril_indices(ni, nj, offset=offset)
        # approach 2: custom
        # offset = 1 if self.arch_interaction_itself else 0
        li = self.get_buffer("li")
        lj = self.get_buffer("lj")
        Zflat = Z[:, li, lj]
        # concatenate dense features and interactions
        R = torch.cat([x] + [Zflat], dim=1)

        return R

### define dlrm in PyTorch ###
class DLRM_Net(nn.Module):
    def create_mlp(self, ln, sigmoid_layer, mlp_in_shape=None, custom_previous_layers = None):
        # build MLP layer by layer
        layers = nn.ModuleList()
        for i in range(0, ln.size - 1):
            n = ln[i]
            m = ln[i + 1]

            # construct fully connected operator
            if i == 0 and custom_previous_layers is not None:
                LL = cs.Linear(int(n), int(m), bias=True, custom_previous_layers=custom_previous_layers)
            else:
                LL = cs.Linear(int(n), int(m), bias=True)

            # initialize the weights
            # with torch.no_grad():
            # custom Xavier input, output or two-sided fill
            mean = 0.0  # std_dev = np.sqrt(variance)
            std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
            W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
            std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
            bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
            # approach 1
            LL.weight.data = torch.tensor(W, requires_grad=True)
            LL.bias.data = torch.tensor(bt, requires_grad=True)
            # approach 2
            # LL.weight.data.copy_(torch.tensor(W))
            # LL.bias.data.copy_(torch.tensor(bt))
            # approach 3
            # LL.weight = Parameter(torch.tensor(W),requires_grad=True)
            # LL.bias = Parameter(torch.tensor(bt),requires_grad=True)
            layers.append(LL)

            # construct sigmoid or relu operator
            if i == sigmoid_layer:
                layers.append(cs.Sigmoid())
            else:
                layers.append(cs.ReLU())

        # approach 1: use ModuleList
        # return layers
        # approach 2: use Sequential container to wrap all layers
        return torch.nn.Sequential(*layers)

    def create_emb(self, m, ln, weighted_pooling=None):
        emb_l = nn.ModuleList()
        v_W_l = []
        for i in range(0, ln.size):
            # if ext_dist.my_size > 1:
            #     if i not in self.local_emb_indices:
            #         continue
            n = ln[i]

            # construct embedding operator
            if self.qr_flag and n > self.qr_threshold:
                EE = QREmbeddingBag(
                    n,
                    m,
                    self.qr_collisions,
                    operation=self.qr_operation,
                    mode="sum",
                    sparse=True,
                )
            elif self.md_flag and n > self.md_threshold:
                base = max(m)
                _m = m[i] if n > self.md_threshold else base
                EE = PrEmbeddingBag(n, _m, base)
                # use np initialization as below for consistency...
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, _m)
                ).astype(np.float32)
                EE.embs.weight.data = torch.tensor(W, requires_grad=True)
            else:
                global do_embed_layer_mp
                if do_embed_layer_mp:
                    global bbn_max_gpus
                    ngpus = bbn_max_gpus
                    EE = cs.EmbeddingBag(n, m, mode="sum", sparse=True, custom_previous_layers=[], device=i%ngpus)
                else:
                    EE = nn.EmbeddingBag(n, m, mode="sum", sparse=True)
                    mod = EmbeddingRepackWrap(EE)
                    cs.GeneralLayer(mod, "EmbeddingBag", {}, custom_previous_layers=[self.distlayer], mustTrace=True)

                self.merge_layers.append(cs.layers[-1])
                # initialize embeddings
                # nn.init.uniform_(EE.weight, a=-np.sqrt(1 / n), b=np.sqrt(1 / n))
                W = np.random.uniform(
                    low=-np.sqrt(1 / n), high=np.sqrt(1 / n), size=(n, m)
                ).astype(np.float32)
                # approach 1
                EE.weight.data = torch.tensor(W, requires_grad=True)
                # approach 2
                # EE.weight.data.copy_(torch.tensor(W))
                # approach 3
                # EE.weight = Parameter(torch.tensor(W),requires_grad=True)
            if weighted_pooling is None:
                v_W_l.append(None)
            else:
                v_W_l.append(torch.ones(n, dtype=torch.float32))
            emb_l.append(EE).to(device=0)
        return emb_l, v_W_l

    def __init__(
        self,
        m_spa=None,
        ln_emb=None,
        ln_bot=None,
        ln_top=None,
        arch_interaction_op=None,
        arch_interaction_itself=False,
        sigmoid_bot=-1,
        sigmoid_top=-1,
        sync_dense_params=True,
        loss_threshold=0.0,
        ndevices=-1,
        qr_flag=False,
        qr_operation="mult",
        qr_collisions=0,
        qr_threshold=200,
        md_flag=False,
        md_threshold=200,
        weighted_pooling=None,
        loss_function="bce",
        mlp_in_shape = None,
        lS_o_shape = None,
        lS_i_shape = None
    ):
        super(DLRM_Net, self).__init__()

        self.merge_layers = []

        if (
            (m_spa is not None)
            and (ln_emb is not None)
            and (ln_bot is not None)
            and (ln_top is not None)
            and (arch_interaction_op is not None)
        ):

            # save arguments
            self.ndevices = ndevices
            self.output_d = 0
            self.parallel_model_batch_size = -1
            self.parallel_model_is_not_prepared = True
            self.arch_interaction_op = arch_interaction_op
            self.arch_interaction_itself = arch_interaction_itself
            self.sync_dense_params = sync_dense_params
            self.loss_threshold = loss_threshold
            self.loss_function=loss_function
            if weighted_pooling is not None and weighted_pooling != "fixed":
                self.weighted_pooling = "learned"
            else:
                self.weighted_pooling = weighted_pooling
            # create variables for QR embedding if applicable
            self.qr_flag = qr_flag
            if self.qr_flag:
                self.qr_collisions = qr_collisions
                self.qr_operation = qr_operation
                self.qr_threshold = qr_threshold
            # create variables for MD embedding if applicable
            self.md_flag = md_flag
            if self.md_flag:
                self.md_threshold = md_threshold

            # If running distributed, get local slice of embedding tables
            # if ext_dist.my_size > 1:
            #     n_emb = len(ln_emb)
            #     if n_emb < ext_dist.my_size:
            #         sys.exit(
            #             "only (%d) sparse features for (%d) devices, table partitions will fail"
            #             % (n_emb, ext_dist.my_size)
            #         )
            #     self.n_global_emb = n_emb
            #     self.n_local_emb, self.n_emb_per_rank = ext_dist.get_split_lengths(
            #         n_emb
            #     )
            #     self.local_emb_slice = ext_dist.get_my_slice(n_emb)
            #     self.local_emb_indices = list(range(n_emb))[self.local_emb_slice]

            distmod = FakeDataDistributionNode(mlp_in_shape[1], len(ln_emb), lS_i_shape)
            self.distlayer = cs.GeneralDistributionLayer(distmod, "inputsplit")
            self.distlayer.addInputShape(torch.zeros(1,  mlp_in_shape[1] + len(ln_emb) + len(ln_emb) * lS_i_shape))
            self.bot_l = self.create_mlp(ln_bot, sigmoid_bot, mlp_in_shape).to(device=0)
            self.merge_layers.append(cs.layers[-1])
            # create operators
            if ndevices <= 1:
                self.emb_l, w_list = self.create_emb(m_spa, ln_emb, weighted_pooling)
                if self.weighted_pooling == "learned":
                    self.v_W_l = nn.ParameterList()
                    for w in w_list:
                        self.v_W_l.append(Parameter(w))
                else:
                    self.v_W_l = w_list
            mod = DLRMDotModule(self.arch_interaction_itself, len(ln_emb))
            layer = cs.GeneralLayer(mod, "dlrm_dot", {"arch_interaction_itself": arch_interaction_itself}, custom_previous_layers = self.merge_layers)
            layer.must_trace = True
            self.top_l = self.create_mlp(ln_top, sigmoid_top)

            # quantization
            self.quantize_emb = False
            self.emb_l_q = []
            self.quantize_bits = 32

            # specify the loss function
            if self.loss_function == "mse":
                self.loss_fn = torch.nn.MSELoss(reduction="mean")
            elif self.loss_function == "bce":
                self.loss_fn = torch.nn.BCELoss(reduction="mean")
            elif self.loss_function == "wbce":
                self.loss_ws = torch.tensor(
                    np.fromstring(args.loss_weights, dtype=float, sep="-")
                )
                self.loss_fn = torch.nn.BCELoss(reduction="none")
            else:
                sys.exit(
                    "ERROR: --loss-function=" + self.loss_function + " is not supported"
                )

    def apply_mlp(self, x, layers):
        # approach 1: use ModuleList
        # for layer in layers:
        #     x = layer(x)
        # return x
        # approach 2: use Sequential container to wrap all layers
        return layers(x)

    def apply_emb(self, lS_o, lS_i, emb_l, v_W_l):
        # WARNING: notice that we are processing the batch at once. We implicitly
        # assume that the data is laid out such that:
        # 1. each embedding is indexed with a group of sparse indices,
        #   corresponding to a single lookup
        # 2. for each embedding the lookups are further organized into a batch
        # 3. for a list of embedding tables there is a list of batched lookups

        ly = []
        for k, sparse_index_group_batch in enumerate(lS_i):
            sparse_offset_group_batch = lS_o[k]

            # embedding lookup
            # We are using EmbeddingBag, which implicitly uses sum operator.
            # The embeddings are represented as tall matrices, with sum
            # happening vertically across 0 axis, resulting in a row vector
            # E = emb_l[k]

            if v_W_l[k] is not None:
                per_sample_weights = v_W_l[k].gather(0, sparse_index_group_batch)
            else:
                per_sample_weights = None

            if self.quantize_emb:
                s1 = self.emb_l_q[k].element_size() * self.emb_l_q[k].nelement()
                s2 = self.emb_l_q[k].element_size() * self.emb_l_q[k].nelement()
                print("quantized emb sizes:", s1, s2)

                if self.quantize_bits == 4:
                    QV = ops.quantized.embedding_bag_4bit_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )
                elif self.quantize_bits == 8:
                    QV = ops.quantized.embedding_bag_byte_rowwise_offsets(
                        self.emb_l_q[k],
                        sparse_index_group_batch,
                        sparse_offset_group_batch,
                        per_sample_weights=per_sample_weights,
                    )

                ly.append(QV)
            else:
                E = emb_l[k]
                V = E(
                    sparse_index_group_batch,
                    sparse_offset_group_batch,
                    per_sample_weights=per_sample_weights,
                )

                ly.append(V)

        # print(ly)
        return ly

    #  using quantizing functions from caffe2/aten/src/ATen/native/quantized/cpu
    def quantize_embedding(self, bits):

        n = len(self.emb_l)
        self.emb_l_q = [None] * n
        for k in range(n):
            if bits == 4:
                self.emb_l_q[k] = ops.quantized.embedding_bag_4bit_prepack(
                    self.emb_l[k].weight
                )
            elif bits == 8:
                self.emb_l_q[k] = ops.quantized.embedding_bag_byte_prepack(
                    self.emb_l[k].weight
                )
            else:
                return
        self.emb_l = None
        self.quantize_emb = True
        self.quantize_bits = bits

    def interact_features(self, x, ly):

        if self.arch_interaction_op == "dot":
            # concatenate dense and sparse features
            (batch_size, d) = x.shape
            T = torch.cat([x] + ly, dim=1).view((batch_size, -1, d))
            # perform a dot product
            Z = torch.bmm(T, torch.transpose(T, 1, 2))
            # append dense feature with the interactions (into a row vector)
            # approach 1: all
            # Zflat = Z.view((batch_size, -1))
            # approach 2: unique
            _, ni, nj = Z.shape
            # approach 1: tril_indices
            # offset = 0 if self.arch_interaction_itself else -1
            # li, lj = torch.tril_indices(ni, nj, offset=offset)
            # approach 2: custom
            offset = 1 if self.arch_interaction_itself else 0
            li = torch.tensor([i for i in range(ni) for j in range(i + offset)])
            lj = torch.tensor([j for i in range(nj) for j in range(i + offset)])
            Zflat = Z[:, li, lj]
            # concatenate dense features and interactions
            R = torch.cat([x] + [Zflat], dim=1)
        elif self.arch_interaction_op == "cat":
            # concatenation features (into a row vector)
            R = torch.cat([x] + ly, dim=1)
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + self.arch_interaction_op
                + " is not supported"
            )

        return R

    def forward(self, dense_x, lS_o, lS_i):
        # if ext_dist.my_size > 1:
        #     # multi-node multi-device run
        #     return self.distributed_forward(dense_x, lS_o, lS_i)
        if self.ndevices <= 1:
            # single device run
            return self.sequential_forward(dense_x, lS_o, lS_i)
        else:
            # single-node multi-device run
            return self.parallel_forward(dense_x, lS_o, lS_i)

    def distributed_forward(self, dense_x, lS_o, lS_i):
        assert False
        # batch_size = dense_x.size()[0]
        # # WARNING: # of ranks must be <= batch size in distributed_forward call
        # if batch_size < ext_dist.my_size:
        #     sys.exit(
        #         "ERROR: batch_size (%d) must be larger than number of ranks (%d)"
        #         % (batch_size, ext_dist.my_size)
        #     )
        # if batch_size % ext_dist.my_size != 0:
        #     sys.exit(
        #         "ERROR: batch_size %d can not split across %d ranks evenly"
        #         % (batch_size, ext_dist.my_size)
        #     )

        # # dense_x = dense_x[ext_dist.get_my_slice(batch_size)]
        # lS_o = lS_o[self.local_emb_slice]
        # lS_i = lS_i[self.local_emb_slice]

        # if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
        #     sys.exit(
        #         "ERROR: corrupted model input detected in distributed_forward call"
        #     )

        # # embeddings
        # with record_function("DLRM embedding forward"):
        #     ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)

        # # WARNING: Note that at this point we have the result of the embedding lookup
        # # for the entire batch on each rank. We would like to obtain partial results
        # # corresponding to all embedding lookups, but part of the batch on each rank.
        # # Therefore, matching the distribution of output of bottom mlp, so that both
        # # could be used for subsequent interactions on each device.
        # if len(self.emb_l) != len(ly):
        #     sys.exit("ERROR: corrupted intermediate result in distributed_forward call")

        # a2a_req = ext_dist.alltoall(ly, self.n_emb_per_rank)

        # with record_function("DLRM bottom nlp forward"):
        #     x = self.apply_mlp(dense_x, self.bot_l)

        # ly = a2a_req.wait()
        # ly = list(ly)

        # # interactions
        # with record_function("DLRM interaction forward"):
        #     z = self.interact_features(x, ly)

        # # top mlp
        # with record_function("DLRM top nlp forward"):
        #     p = self.apply_mlp(z, self.top_l)

        # # clamp output if needed
        # if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
        #     z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        # else:
        #     z = p

        # return z

    def sequential_forward(self, dense_x, lS_o, lS_i):
        # process dense features (using bottom mlp), resulting in a row vector
        x = self.apply_mlp(dense_x, self.bot_l)
        # debug prints
        # print("intermediate")
        # print(x.detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        # for y in ly:
        #     print(y.detach().cpu().numpy())

        # interact features (dense and sparse)
        z = self.interact_features(x, ly)
        # print(z.detach().cpu().numpy())

        # obtain probability of a click (using top mlp)
        p = self.apply_mlp(z, self.top_l)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z = torch.clamp(p, min=self.loss_threshold, max=(1.0 - self.loss_threshold))
        else:
            z = p

        return z

    def parallel_forward(self, dense_x, lS_o, lS_i):
        ### prepare model (overwrite) ###
        # WARNING: # of devices must be >= batch size in parallel_forward call
        batch_size = dense_x.size()[0]
        ndevices = min(self.ndevices, batch_size, len(self.emb_l))
        device_ids = range(ndevices)
        # WARNING: must redistribute the model if mini-batch size changes(this is common
        # for last mini-batch, when # of elements in the dataset/batch size is not even
        if self.parallel_model_batch_size != batch_size:
            self.parallel_model_is_not_prepared = True

        if self.parallel_model_is_not_prepared or self.sync_dense_params:
            # replicate mlp (data parallelism)
            self.bot_l_replicas = replicate(self.bot_l, device_ids)
            self.top_l_replicas = replicate(self.top_l, device_ids)
            self.parallel_model_batch_size = batch_size

        if self.parallel_model_is_not_prepared:
            # distribute embeddings (model parallelism)
            t_list = []
            w_list = []
            for k, emb in enumerate(self.emb_l):
                d = torch.device("cuda:" + str(k % ndevices))
                t_list.append(emb.to(d))
                if self.weighted_pooling == "learned":
                    w_list.append(Parameter(self.v_W_l[k].to(d)))
                elif self.weighted_pooling == "fixed":
                    w_list.append(self.v_W_l[k].to(d))
                else:
                    w_list.append(None)
            self.emb_l = nn.ModuleList(t_list)
            if self.weighted_pooling == "learned":
                self.v_W_l = nn.ParameterList(w_list)
            else:
                self.v_W_l = w_list
            self.parallel_model_is_not_prepared = False

        ### prepare input (overwrite) ###
        # scatter dense features (data parallelism)
        # print(dense_x.device)
        dense_x = scatter(dense_x, device_ids, dim=0)
        # distribute sparse features (model parallelism)
        if (len(self.emb_l) != len(lS_o)) or (len(self.emb_l) != len(lS_i)):
            sys.exit("ERROR: corrupted model input detected in parallel_forward call")

        t_list = []
        i_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            t_list.append(lS_o[k].to(d))
            i_list.append(lS_i[k].to(d))
        lS_o = t_list
        lS_i = i_list

        ### compute results in parallel ###
        # bottom mlp
        # WARNING: Note that the self.bot_l is a list of bottom mlp modules
        # that have been replicated across devices, while dense_x is a tuple of dense
        # inputs that has been scattered across devices on the first (batch) dimension.
        # The output is a list of tensors scattered across devices according to the
        # distribution of dense_x.
        x = parallel_apply(self.bot_l_replicas, dense_x, None, device_ids)
        # debug prints
        # print(x)

        # embeddings
        ly = self.apply_emb(lS_o, lS_i, self.emb_l, self.v_W_l)
        # debug prints
        # print(ly)

        # butterfly shuffle (implemented inefficiently for now)
        # WARNING: Note that at this point we have the result of the embedding lookup
        # for the entire batch on each device. We would like to obtain partial results
        # corresponding to all embedding lookups, but part of the batch on each device.
        # Therefore, matching the distribution of output of bottom mlp, so that both
        # could be used for subsequent interactions on each device.
        if len(self.emb_l) != len(ly):
            sys.exit("ERROR: corrupted intermediate result in parallel_forward call")

        t_list = []
        for k, _ in enumerate(self.emb_l):
            d = torch.device("cuda:" + str(k % ndevices))
            y = scatter(ly[k], device_ids, dim=0)
            t_list.append(y)
        # adjust the list to be ordered per device
        ly = list(map(lambda y: list(y), zip(*t_list)))
        # debug prints
        # print(ly)

        # interactions
        z = []
        for k in range(ndevices):
            zk = self.interact_features(x[k], ly[k])
            z.append(zk)
        # debug prints
        # print(z)

        # top mlp
        # WARNING: Note that the self.top_l is a list of top mlp modules that
        # have been replicated across devices, while z is a list of interaction results
        # that by construction are scattered across devices on the first (batch) dim.
        # The output is a list of tensors scattered across devices according to the
        # distribution of z.
        p = parallel_apply(self.top_l_replicas, z, None, device_ids)

        ### gather the distributed results ###
        p0 = gather(p, self.output_d, dim=0)

        # clamp output if needed
        if 0.0 < self.loss_threshold and self.loss_threshold < 1.0:
            z0 = torch.clamp(
                p0, min=self.loss_threshold, max=(1.0 - self.loss_threshold)
            )
        else:
            z0 = p0

        return z0


def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value


def dash_separated_floats(value):
    vals = value.split("-")
    for val in vals:
        try:
            float(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of floats" % value
            )

    return value


def inference(
    args,
    dlrm,
    best_acc_test,
    best_auc_test,
    test_ld,
    device,
    use_gpu,
    log_iter=-1,
):
    test_accu = 0
    test_samp = 0

    if args.mlperf_logging:
        scores = []
        targets = []

    for i, testBatch in enumerate(test_ld):
        # early exit if nbatches was set by the user and was exceeded
        if nbatches > 0 and i >= nbatches:
            break

        X_test, lS_o_test, lS_i_test, T_test, W_test, CBPP_test = unpack_batch(
            testBatch
        )

        # Skip the batch if batch size not multiple of total ranks
        # if ext_dist.my_size > 1 and X_test.size(0) % ext_dist.my_size != 0:
        #     print("Warning: Skiping the batch %d with size %d" % (i, X_test.size(0)))
        #     continue

        # forward pass
        Z_test = dlrm_wrap(
            X_test,
            lS_o_test,
            lS_i_test,
            use_gpu,
            device,
            ndevices=ndevices,
        )
        ### gather the distributed results on each rank ###
        # For some reason it requires explicit sync before all_gather call if
        # tensor is on GPU memory
        if Z_test.is_cuda:
            torch.cuda.synchronize()
        # (_, batch_split_lengths) = ext_dist.get_split_lengths(X_test.size(0))
        (_, batch_split_lengths) = (X_test.size(0), None)
        # if ext_dist.my_size > 1:
        #     Z_test = ext_dist.all_gather(Z_test, batch_split_lengths)

        if args.mlperf_logging:
            S_test = Z_test.detach().cpu().numpy()  # numpy array
            T_test = T_test.detach().cpu().numpy()  # numpy array
            scores.append(S_test)
            targets.append(T_test)
        else:
            with record_function("DLRM accuracy compute"):
                # compute loss and accuracy
                S_test = Z_test.detach().cpu().numpy()  # numpy array
                T_test = T_test.detach().cpu().numpy()  # numpy array

                mbs_test = T_test.shape[0]  # = mini_batch_size except last
                A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))

                test_accu += A_test
                test_samp += mbs_test

    if args.mlperf_logging:
        with record_function("DLRM mlperf sklearn metrics compute"):
            scores = np.concatenate(scores, axis=0)
            targets = np.concatenate(targets, axis=0)

            metrics = {
                "recall": lambda y_true, y_score: sklearn.metrics.recall_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "precision": lambda y_true, y_score: sklearn.metrics.precision_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "f1": lambda y_true, y_score: sklearn.metrics.f1_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
                "ap": sklearn.metrics.average_precision_score,
                "roc_auc": sklearn.metrics.roc_auc_score,
                "accuracy": lambda y_true, y_score: sklearn.metrics.accuracy_score(
                    y_true=y_true, y_pred=np.round(y_score)
                ),
            }

        validation_results = {}
        for metric_name, metric_function in metrics.items():
            validation_results[metric_name] = metric_function(targets, scores)
            writer.add_scalar(
                "mlperf-metrics-test/" + metric_name,
                validation_results[metric_name],
                log_iter,
            )
        acc_test = validation_results["accuracy"]
    else:
        acc_test = test_accu / test_samp
        writer.add_scalar("Test/Acc", acc_test, log_iter)

    model_metrics_dict = {
        "nepochs": args.nepochs,
        "nbatches": nbatches,
        "nbatches_test": nbatches_test,
        "state_dict": dlrm.state_dict(),
        "test_acc": acc_test,
    }

    if args.mlperf_logging:
        is_best = validation_results["roc_auc"] > best_auc_test
        if is_best:
            best_auc_test = validation_results["roc_auc"]
            model_metrics_dict["test_auc"] = best_auc_test
        print(
            "recall {:.4f}, precision {:.4f},".format(
                validation_results["recall"],
                validation_results["precision"],
            )
            + " f1 {:.4f}, ap {:.4f},".format(
                validation_results["f1"], validation_results["ap"]
            )
            + " auc {:.4f}, best auc {:.4f},".format(
                validation_results["roc_auc"], best_auc_test
            )
            + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                validation_results["accuracy"] * 100, best_acc_test * 100
            ),
            flush=True,
        )
    else:
        is_best = acc_test > best_acc_test
        if is_best:
            best_acc_test = acc_test
        print(
            " accuracy {:3.3f} %, best {:3.3f} %".format(
                acc_test * 100, best_acc_test * 100
            ),
            flush=True,
        )
    return model_metrics_dict, is_best



# WARNING: global define, must be consistent across all synthetic functions
cache_line_size = 1



def generate_stack_distance(cumm_val, cumm_dist, max_i, i, enable_padding=False):
    u = ra.rand(1)
    if i < max_i:
        # only generate stack distances up to the number of new references seen so far
        j = bisect.bisect(cumm_val, i) - 1
        fi = cumm_dist[j]
        u *= fi  # shrink distribution support to exclude last values
    elif enable_padding:
        # WARNING: disable generation of new references (once all have been seen)
        fi = cumm_dist[0]
        u = (1.0 - fi) * u + fi  # remap distribution support to exclude first value

    for (j, f) in enumerate(cumm_dist):
        if u <= f:
            return cumm_val[j]

def trace_generate_lru(
    line_accesses, list_sd, cumm_sd, out_trace_len, enable_padding=False
):
    max_sd = list_sd[-1]
    l = len(line_accesses)
    i = 0
    ztrace = deque()
    for _ in range(out_trace_len):
        sd = generate_stack_distance(list_sd, cumm_sd, max_sd, i, enable_padding)
        mem_ref_within_line = 0  # floor(ra.rand(1)*cache_line_size) #0

        # generate memory reference
        if sd == 0:  # new reference #
            line_ref = line_accesses[0]
            del line_accesses[0]
            line_accesses.append(line_ref)
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            i += 1
        else:  # existing reference #
            line_ref = line_accesses[l - sd]
            mem_ref = np.uint64(line_ref * cache_line_size + mem_ref_within_line)
            del line_accesses[l - sd]
            line_accesses.append(line_ref)
        # save generated memory reference
        ztrace.append(mem_ref)

    return ztrace

def read_dist_from_file(file_path):
    try:
        with open(file_path, "r") as f:
            lines = f.read().splitlines()
    except Exception:
        print("{file_path} Wrong file or file path")
    # read unique accesses
    unique_accesses = [int(el) for el in lines[0].split(", ")]
    # read cumulative distribution (elements are passed as two separate lists)
    list_sd = [int(el) for el in lines[1].split(", ")]
    cumm_sd = [float(el) for el in lines[2].split(", ")]

    return unique_accesses, list_sd, cumm_sd

# synthetic distribution (input data)
def generate_synthetic_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    trace_file,
    enable_padding=False,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for i, size in enumerate(ln_emb):
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    max(1, np.round(r * min(size, num_indices_per_lookup))[0])
                )
            # sparse indices to be used per embedding
            file_path = trace_file
            line_accesses, list_sd, cumm_sd = read_dist_from_file(
                file_path.replace("j", str(i))
            )
            # debug prints
            # print("input")
            # print(line_accesses); print(list_sd); print(cumm_sd);
            # print(sparse_group_size)
            # approach 1: rand
            # r = trace_generate_rand(
            #     line_accesses, list_sd, cumm_sd, sparse_group_size, enable_padding
            # )
            # approach 2: lru
            r = trace_generate_lru(
                line_accesses, list_sd, cumm_sd, sparse_group_size, enable_padding
            )
            # WARNING: if the distribution in the file is not consistent
            # with embedding table dimensions, below mod guards against out
            # of range access
            sparse_group = np.unique(r).astype(np.int64)
            minsg = np.min(sparse_group)
            maxsg = np.max(sparse_group)
            if (minsg < 0) or (size <= maxsg):
                print(
                    "WARNING: distribution is inconsistent with embedding "
                    + "table size (using mod to recover and continue)"
                )
                sparse_group = np.mod(sparse_group, size).astype(np.int64)
            # sparse_group = np.unique(np.array(np.mod(r, size-1)).astype(np.int64))
            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)

# random data from uniform or gaussian ditribution (input data)
def generate_dist_input_batch(
    m_den,
    ln_emb,
    n,
    num_indices_per_lookup,
    num_indices_per_lookup_fixed,
    rand_data_dist,
    rand_data_min,
    rand_data_max,
    rand_data_mu,
    rand_data_sigma,
):
    # dense feature
    Xt = torch.tensor(ra.rand(n, m_den).astype(np.float32))

    # sparse feature (sparse indices)
    lS_emb_offsets = []
    lS_emb_indices = []
    # for each embedding generate a list of n lookups,
    # where each lookup is composed of multiple sparse indices
    for size in ln_emb:
        lS_batch_offsets = []
        lS_batch_indices = []
        offset = 0
        for _ in range(n):
            # num of sparse indices to be used per embedding (between
            if num_indices_per_lookup_fixed:
                sparse_group_size = np.int64(num_indices_per_lookup)
            else:
                # random between [1,num_indices_per_lookup])
                r = ra.random(1)
                sparse_group_size = np.int64(
                    np.round(max([1.0], r * min(size, num_indices_per_lookup)))
                )
            # sparse indices to be used per embedding
            if rand_data_dist == "gaussian":
                if rand_data_mu == -1:
                    rand_data_mu = (rand_data_max + rand_data_min) / 2.0
                r = ra.normal(rand_data_mu, rand_data_sigma, sparse_group_size)
                sparse_group = np.clip(r, rand_data_min, rand_data_max)
                sparse_group = np.unique(sparse_group).astype(np.int64)
            elif rand_data_dist == "uniform":
                r = ra.random(sparse_group_size)
                sparse_group = np.unique(np.round(r * (size - 1)).astype(np.int64))
            else:
                raise(rand_data_dist, "distribution is not supported. \
                     please select uniform or gaussian")

            # reset sparse_group_size in case some index duplicates were removed
            sparse_group_size = np.int64(sparse_group.size)
            # store lengths and indices
            lS_batch_offsets += [offset]
            lS_batch_indices += sparse_group.tolist()
            # update offset for next iteration
            offset += sparse_group_size
        lS_emb_offsets.append(torch.tensor(lS_batch_offsets))
        lS_emb_indices.append(torch.tensor(lS_batch_indices))

    return (Xt, lS_emb_offsets, lS_emb_indices)

def generate_random_output_batch(n, num_targets, round_targets=False):
    # target (probability of a click)
    if round_targets:
        P = np.round(ra.rand(n, num_targets).astype(np.float32)).astype(np.float32)
    else:
        P = ra.rand(n, num_targets).astype(np.float32)

    return torch.tensor(P)

# Conversion from offset to length
def offset_to_length_converter(lS_o, lS_i):
    def diff(tensor):
        return tensor[1:] - tensor[:-1]

    return torch.stack(
        [
            diff(torch.cat((S_o, torch.tensor(lS_i[ind].shape))).int())
            for ind, S_o in enumerate(lS_o)
        ]
    )

# uniform ditribution (input data)
class RandomDataset(Dataset):

    def __init__(
            self,
            m_den,
            ln_emb,
            data_size,
            num_batches,
            mini_batch_size,
            num_indices_per_lookup,
            num_indices_per_lookup_fixed,
            num_targets=1,
            round_targets=False,
            data_generation="random",
            trace_file="",
            enable_padding=False,
            reset_seed_on_access=False,
            rand_data_dist="uniform",
            rand_data_min=1,
            rand_data_max=1,
            rand_data_mu=-1,
            rand_data_sigma=1,
            rand_seed=0
    ):
        # compute batch size
        nbatches = int(np.ceil((data_size * 1.0) / mini_batch_size))
        if num_batches != 0:
            nbatches = num_batches
            data_size = nbatches * mini_batch_size
            # print("Total number of batches %d" % nbatches)

        # save args (recompute data_size if needed)
        self.m_den = m_den
        self.ln_emb = ln_emb
        self.data_size = data_size
        self.num_batches = nbatches
        self.mini_batch_size = mini_batch_size
        self.num_indices_per_lookup = num_indices_per_lookup
        self.num_indices_per_lookup_fixed = num_indices_per_lookup_fixed
        self.num_targets = num_targets
        self.round_targets = round_targets
        self.data_generation = data_generation
        self.trace_file = trace_file
        self.enable_padding = enable_padding
        self.reset_seed_on_access = reset_seed_on_access
        self.rand_seed = rand_seed
        self.rand_data_dist = rand_data_dist
        self.rand_data_min = rand_data_min
        self.rand_data_max = rand_data_max
        self.rand_data_mu = rand_data_mu
        self.rand_data_sigma = rand_data_sigma

    def reset_numpy_seed(self, numpy_rand_seed):
        np.random.seed(numpy_rand_seed)
        # torch.manual_seed(numpy_rand_seed)

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx] for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        # WARNING: reset seed on access to first element
        # (e.g. if same random samples needed across epochs)
        if self.reset_seed_on_access and index == 0:
            self.reset_numpy_seed(self.rand_seed)

        # number of data points in a batch
        n = min(self.mini_batch_size, self.data_size - (index * self.mini_batch_size))

        # generate a batch of dense and sparse features
        if self.data_generation == "random":
            (X, lS_o, lS_i) = generate_dist_input_batch(
                self.m_den,
                self.ln_emb,
                n,
                self.num_indices_per_lookup,
                self.num_indices_per_lookup_fixed,
                rand_data_dist=self.rand_data_dist,
                rand_data_min=self.rand_data_min,
                rand_data_max=self.rand_data_max,
                rand_data_mu=self.rand_data_mu,
                rand_data_sigma=self.rand_data_sigma,
            )
        elif self.data_generation == "synthetic":
            (X, lS_o, lS_i) = generate_synthetic_input_batch(
                self.m_den,
                self.ln_emb,
                n,
                self.num_indices_per_lookup,
                self.num_indices_per_lookup_fixed,
                self.trace_file,
                self.enable_padding
            )
        else:
            sys.exit(
                "ERROR: --data-generation=" + self.data_generation + " is not supported"
            )

        # generate a batch of target (probability of a click)
        T = generate_random_output_batch(n, self.num_targets, self.round_targets)

        return (X, lS_o, lS_i, T)

    def __len__(self):
        # WARNING: note that we produce bacthes of outputs in __getitem__
        # therefore we should use num_batches rather than data_size below
        return self.num_batches


def collate_wrapper_random_offset(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X,
            torch.stack(lS_o),
            lS_i,
            T)


def collate_wrapper_random_length(list_of_tuples):
    # where each tuple is (X, lS_o, lS_i, T)
    (X, lS_o, lS_i, T) = list_of_tuples[0]
    return (X,
            offset_to_length_converter(torch.stack(lS_o), lS_i),
            lS_i,
            T)



def make_random_data_and_loader(args, ln_emb, m_den,
    offset_to_length_converter=False,
):

    train_data = RandomDataset(
        m_den,
        ln_emb,
        args.data_size,
        args.num_batches,
        args.mini_batch_size,
        args.num_indices_per_lookup,
        args.num_indices_per_lookup_fixed,
        1,  # num_targets
        args.round_targets,
        args.data_generation,
        args.data_trace_file,
        args.data_trace_enable_padding,
        reset_seed_on_access=True,
        rand_data_dist=args.rand_data_dist,
        rand_data_min=args.rand_data_min,
        rand_data_max=args.rand_data_max,
        rand_data_mu=args.rand_data_mu,
        rand_data_sigma=args.rand_data_sigma,
        rand_seed=args.numpy_rand_seed
    )  # WARNING: generates a batch of lookups at once

    test_data = RandomDataset(
        m_den,
        ln_emb,
        args.data_size,
        args.num_batches,
        args.mini_batch_size,
        args.num_indices_per_lookup,
        args.num_indices_per_lookup_fixed,
        1,  # num_targets
        args.round_targets,
        args.data_generation,
        args.data_trace_file,
        args.data_trace_enable_padding,
        reset_seed_on_access=True,
        rand_data_dist=args.rand_data_dist,
        rand_data_min=args.rand_data_min,
        rand_data_max=args.rand_data_max,
        rand_data_mu=args.rand_data_mu,
        rand_data_sigma=args.rand_data_sigma,
        rand_seed=args.numpy_rand_seed
    )

    collate_wrapper_random = collate_wrapper_random_offset
    if offset_to_length_converter:
        collate_wrapper_random = collate_wrapper_random_length

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=False,
        drop_last=False,  # True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_random,
        pin_memory=False,
        drop_last=False,  # True
    )
    return train_data, train_loader, test_data, test_loader










def run(gpuCount, amplificationLimit=2.0, dataParallelBaseline=False, netBw=2.66E5, spatialSplit=False, simResultFilename=None, simOnly=False, use_be=False):
    global cs
    cs = CostSim(None, netBw=netBw, verbose=True, gpuProfileLoc="profile/A100_anvil.prof")#, gpuProfileLocSub="resnetLayerGpuProfileA100.txt")
    global bbn_max_gpus
    bbn_max_gpus = gpuCount
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Train Deep Learning Recommendation Model (DLRM)"
    )
    # model related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=2)
    parser.add_argument(
        "--arch-embedding-size", type=dash_separated_ints, default="4-3-2"
    )
    # j will be replaced with the table number
    parser.add_argument("--arch-mlp-bot", type=dash_separated_ints, default="4-3-2")
    parser.add_argument("--arch-mlp-top", type=dash_separated_ints, default="4-2-1")
    parser.add_argument(
        "--arch-interaction-op", type=str, choices=["dot", "cat"], default="dot"
    )
    parser.add_argument("--arch-interaction-itself", action="store_true", default=False)
    parser.add_argument("--weighted-pooling", type=str, default=None)
    # embedding table options
    parser.add_argument("--md-flag", action="store_true", default=False)
    parser.add_argument("--md-threshold", type=int, default=200)
    parser.add_argument("--md-temperature", type=float, default=0.3)
    parser.add_argument("--md-round-dims", action="store_true", default=False)
    parser.add_argument("--qr-flag", action="store_true", default=False)
    parser.add_argument("--qr-threshold", type=int, default=200)
    parser.add_argument("--qr-operation", type=str, default="mult")
    parser.add_argument("--qr-collisions", type=int, default=4)
    # activations and loss
    parser.add_argument("--activation-function", type=str, default="relu")
    parser.add_argument("--loss-function", type=str, default="mse")  # or bce or wbce
    parser.add_argument(
        "--loss-weights", type=dash_separated_floats, default="1.0-1.0"
    )  # for wbce
    parser.add_argument("--loss-threshold", type=float, default=0.0)  # 1.0e-7
    parser.add_argument("--round-targets", type=bool, default=False)
    # data
    parser.add_argument("--data-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=0)
    parser.add_argument(
        "--data-generation", type=str, default="random"
    )  # synthetic or dataset
    parser.add_argument(
        "--rand-data-dist", type=str, default="uniform"
    )  # uniform or gaussian
    parser.add_argument("--rand-data-min", type=float, default=0)
    parser.add_argument("--rand-data-max", type=float, default=1)
    parser.add_argument("--rand-data-mu", type=float, default=-1)
    parser.add_argument("--rand-data-sigma", type=float, default=1)
    parser.add_argument("--data-trace-file", type=str, default="./input/dist_emb_j.log")
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-randomize", type=str, default="total")  # or day or none
    parser.add_argument("--data-trace-enable-padding", type=bool, default=False)
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--num-indices-per-lookup", type=int, default=10)
    parser.add_argument("--num-indices-per-lookup-fixed", type=bool, default=False)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--memory-map", action="store_true", default=False)
    # training
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--nepochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--print-precision", type=int, default=5)
    parser.add_argument("--numpy-rand-seed", type=int, default=123)
    parser.add_argument("--sync-dense-params", type=bool, default=True)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument(
        "--dataset-multiprocessing",
        action="store_true",
        default=False,
        help="The Kaggle dataset can be multiprocessed in an environment \
                        with more than 7 CPU cores and more than 20 GB of memory. \n \
                        The Terabyte dataset can be multiprocessed in an environment \
                        with more than 24 CPU cores and at least 1 TB of memory.",
    )
    # inference
    parser.add_argument("--inference-only", action="store_true", default=False)
    # quantize
    parser.add_argument("--quantize-mlp-with-bit", type=int, default=32)
    parser.add_argument("--quantize-emb-with-bit", type=int, default=32)
    # onnx
    parser.add_argument("--save-onnx", action="store_true", default=False)
    # gpu
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # distributed
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dist-backend", type=str, default="")
    # debugging and profiling
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--test-freq", type=int, default=-1)
    parser.add_argument("--test-mini-batch-size", type=int, default=-1)
    parser.add_argument("--test-num-workers", type=int, default=-1)
    parser.add_argument("--print-time", action="store_true", default=False)
    parser.add_argument("--print-wall-time", action="store_true", default=False)
    parser.add_argument("--debug-mode", action="store_true", default=False)
    parser.add_argument("--enable-profiling", action="store_true", default=False)
    parser.add_argument("--plot-compute-graph", action="store_true", default=True)
    parser.add_argument("--tensor-board-filename", type=str, default="run_kaggle_pt")
    # store/load model
    parser.add_argument("--save-model", type=str, default="")
    parser.add_argument("--load-model", type=str, default="")
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)
    # stop at target accuracy Kaggle 0.789, Terabyte (sub-sampled=0.875) 0.8107
    parser.add_argument("--mlperf-acc-threshold", type=float, default=0.0)
    # stop at target AUC Terabyte (no subsampling) 0.8025
    parser.add_argument("--mlperf-auc-threshold", type=float, default=0.0)
    parser.add_argument("--mlperf-bin-loader", action="store_true", default=False)
    parser.add_argument("--mlperf-bin-shuffle", action="store_true", default=False)
    # mlperf gradient accumulation iterations
    parser.add_argument("--mlperf-grad-accum-iter", type=int, default=1)
    # LR policy
    parser.add_argument("--lr-num-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-start-step", type=int, default=0)
    parser.add_argument("--lr-num-decay-steps", type=int, default=0)

    global args
    global nbatches
    global nbatches_test
    global writer
    args = parser.parse_args()

    if args.dataset_multiprocessing:
        assert float(sys.version[:3]) > 3.7, "The dataset_multiprocessing " + \
        "flag is susceptible to a bug in Python 3.7 and under. " + \
        "https://github.com/facebookresearch/dlrm/issues/172"

    # if args.mlperf_logging:
    #     mlperf_logger.log_event(key=mlperf_logger.constants.CACHE_CLEAR, value=True)
    #     mlperf_logger.log_start(
    #         key=mlperf_logger.constants.INIT_START, log_all_ranks=True
    #     )

    if args.weighted_pooling is not None:
        if args.qr_flag:
            sys.exit("ERROR: quotient remainder with weighted pooling is not supported")
        if args.md_flag:
            sys.exit("ERROR: mixed dimensions with weighted pooling is not supported")
    if args.quantize_emb_with_bit in [4, 8]:
        if args.qr_flag:
            sys.exit(
                "ERROR: 4 and 8-bit quantization with quotient remainder is not supported"
            )
        if args.md_flag:
            sys.exit(
                "ERROR: 4 and 8-bit quantization with mixed dimensions is not supported"
            )
        if args.use_gpu:
            sys.exit(
                "ERROR: 4 and 8-bit quantization on GPU is not supported"
            )

    globalBatch = args.mini_batch_size

    ### some basic setup ###
    np.random.seed(args.numpy_rand_seed)
    np.set_printoptions(precision=args.print_precision)
    torch.set_printoptions(precision=args.print_precision)
    torch.manual_seed(args.numpy_rand_seed)

    if args.test_mini_batch_size < 0:
        # if the parameter is not set, use the training batch size
        args.test_mini_batch_size = args.mini_batch_size
    if args.test_num_workers < 0:
        # if the parameter is not set, use the same parameter for training
        args.test_num_workers = args.num_workers

    use_gpu = args.use_gpu and torch.cuda.is_available()

    # if not args.debug_mode:
    #     ext_dist.init_distributed(local_rank=args.local_rank, use_gpu=use_gpu, backend=args.dist_backend)

    if use_gpu:
        torch.cuda.manual_seed_all(args.numpy_rand_seed)
        torch.backends.cudnn.deterministic = True
        # if ext_dist.my_size <= 1:
        #     ngpus = 1
        #     device = torch.device("cuda", ext_dist.my_local_rank)
        # else:
        if True:
            ngpus = 1 #torch.cuda.device_count()
            device = torch.device("cuda", 0)
        print("Using {} GPU(s)...".format(ngpus))
    else:
        device = torch.device("cpu")
        print("Using CPU...")

    ### prepare training data ###
    ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
    # input data

    # if args.mlperf_logging:
    #     mlperf_logger.barrier()
    #     mlperf_logger.log_end(key=mlperf_logger.constants.INIT_STOP)
    #     mlperf_logger.barrier()
    #     mlperf_logger.log_start(key=mlperf_logger.constants.RUN_START)
    #     mlperf_logger.barrier()

    # if args.data_generation == "dataset":
    #     train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)
    #     table_feature_map = {idx: idx for idx in range(len(train_data.counts))}
    #     nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
    #     nbatches_test = len(test_ld)

    #     ln_emb = train_data.counts
    #     # enforce maximum limit on number of vectors per embedding
    #     if args.max_ind_range > 0:
    #         ln_emb = np.array(
    #             list(
    #                 map(
    #                     lambda x: x if x < args.max_ind_range else args.max_ind_range,
    #                     ln_emb,
    #                 )
    #             )
    #         )
    #     else:
    #         ln_emb = np.array(ln_emb)
    #     m_den = train_data.m_den
    #     ln_bot[0] = m_den
    if True:
        # input and target at random
        ln_emb = np.fromstring(args.arch_embedding_size, dtype=int, sep="-")
        m_den = ln_bot[0]
        train_data, train_ld, test_data, test_ld = make_random_data_and_loader(args, ln_emb, m_den)
        nbatches = args.num_batches if args.num_batches > 0 else len(train_ld)
        nbatches_test = len(test_ld)

    args.ln_emb = ln_emb.tolist()
    if args.mlperf_logging:
        print("command line args: ", json.dumps(vars(args)))

    ### parse command line arguments ###
    m_spa = args.arch_sparse_feature_size
    ln_emb = np.asarray(ln_emb)
    num_fea = ln_emb.size + 1  # num sparse + num dense features

    m_den_out = ln_bot[ln_bot.size - 1]
    if args.arch_interaction_op == "dot":
        # approach 1: all
        # num_int = num_fea * num_fea + m_den_out
        # approach 2: unique
        if args.arch_interaction_itself:
            num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
        else:
            num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
    elif args.arch_interaction_op == "cat":
        num_int = num_fea * m_den_out
    else:
        sys.exit(
            "ERROR: --arch-interaction-op="
            + args.arch_interaction_op
            + " is not supported"
        )
    arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
    ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    # sanity check: feature sizes and mlp dimensions must match
    # if m_den != ln_bot[0]:
    #     sys.exit(
    #         "ERROR: arch-dense-feature-size "
    #         + str(m_den)
    #         + " does not match first dim of bottom mlp "
    #         + str(ln_bot[0])
    #     )
    if args.qr_flag:
        if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
            sys.exit(
                "ERROR: 2 arch-sparse-feature-size "
                + str(2 * m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
                + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
            )
        if args.qr_operation != "concat" and m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    else:
        if m_spa != m_den_out:
            sys.exit(
                "ERROR: arch-sparse-feature-size "
                + str(m_spa)
                + " does not match last dim of bottom mlp "
                + str(m_den_out)
            )
    if num_int != ln_top[0]:
        sys.exit(
            "ERROR: # of feature interactions "
            + str(num_int)
            + " does not match first dimension of top mlp "
            + str(ln_top[0])
        )

    # assign mixed dimensions if applicable
    if args.md_flag:
        m_spa = md_solver(
            torch.tensor(ln_emb),
            args.md_temperature,  # alpha
            d0=m_spa,
            round_dim=args.md_round_dims,
        ).tolist()

    # test prints (model arch)
    #if args.debug_mode:
    if True:
        # print("model arch:")
        # print(
        #     "mlp top arch "
        #     + str(ln_top.size - 1)
        #     + " layers, with input to output dimensions:"
        # )
        # print(ln_top)
        # print("# of interactions")
        # print(num_int)
        # print(
        #     "mlp bot arch "
        #     + str(ln_bot.size - 1)
        #     + " layers, with input to output dimensions:"
        # )
        # print(ln_bot)
        # print("# of features (sparse and dense)")
        # print(num_fea)
        # print("dense feature size")
        # print(m_den)
        # print("sparse feature size")
        # print(m_spa)
        # print(
        #     "# of embeddings (= # of sparse features) "
        #     + str(ln_emb.size)
        #     + ", with dimensions "
        #     + str(m_spa)
        #     + "x:"
        # )
        # print(ln_emb)

        print("data (inputs and targets):")
        exampleBatch = None
        average_lS_i_len = 0
        for j, inputBatch in enumerate(train_ld):
            if j >= 10:
                break
            exampleBatch = inputBatch
            X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)
            print('X.shape',X.shape)
            print('lS_o.shape',lS_o.shape)
            average_lS_i_len+=lS_i[0].shape[0]
            print('len(lS_i)',len(lS_i), lS_i[0].shape)
            print('T.shape',T.shape)
            print('W.shape',W.shape)
            print('CBPP',CBPP)

            # torch.set_printoptions(precision=4)
            # # early exit if nbatches was set by the user and has been exceeded
            # if nbatches > 0 and j >= nbatches:
            #     break
            # print("mini-batch: %d" % j)
            # print(X.detach().cpu())
            # # transform offsets to lengths when printing
            # print(
            #     torch.IntTensor(
            #         [
            #             np.diff(
            #                 S_o.detach().cpu().tolist() + list(lS_i[i].shape)
            #             ).tolist()
            #             for i, S_o in enumerate(lS_o)
            #         ]
            #     )
            # )
            # print([S_i.detach().cpu() for S_i in lS_i])
            # print(T.detach().cpu())
            # break
    average_lS_i_len //= j + 1
    X, lS_o, lS_i, T, W, CBPP = unpack_batch(exampleBatch)
    
    global ndevices
    # ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1
    ndevices = 1
    
    ### construct the neural network specified above ###
    # WARNING: to obtain exactly the same initialization for
    # the weights we need to start from the same random seed.
    # np.random.seed(args.numpy_rand_seed)
    global dlrm
    dlrm = DLRM_Net(
        m_spa,
        ln_emb,
        ln_bot,
        ln_top,
        arch_interaction_op=args.arch_interaction_op,
        arch_interaction_itself=args.arch_interaction_itself,
        sigmoid_bot=-1,
        sigmoid_top=ln_top.size - 2,
        sync_dense_params=args.sync_dense_params,
        loss_threshold=args.loss_threshold,
        ndevices=ndevices,
        qr_flag=args.qr_flag,
        qr_operation=args.qr_operation,
        qr_collisions=args.qr_collisions,
        qr_threshold=args.qr_threshold,
        md_flag=args.md_flag,
        md_threshold=args.md_threshold,
        weighted_pooling=args.weighted_pooling,
        loss_function=args.loss_function,
        mlp_in_shape=X.shape,
        lS_o_shape = lS_o.shape,
        lS_i_shape = (average_lS_i_len)
    )
    
    # Z = dlrm_wrap(
    #     X,
    #     lS_o,
    #     lS_i,
    #     use_gpu,
    #     device,
    #     ndevices=ndevices,
    # )
    

    # test prints
    if args.debug_mode:
        print("initial parameters (weights and bias):")
        for param in dlrm.parameters():
            print(param.detach().cpu().numpy())
        # print(dlrm)

    # model = dlrm
    X, lS_o, lS_i, T, W, CBPP = unpack_batch(exampleBatch)

    cs.printAllLayers()
    # cs.computeInputDimensions((2,240,240))
    cs.to_dot("Digraph", globalBatch, justdag=True)

    # job, iterMs, gpuMs, maxGpusUsed = cs.JustDoDP(gpuCount, globalBatch)
    # print("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, maxGpusUsed, iterMs, gpuMs))

    # jobInJson = job.dumpInJSON()

    cs.setLossFunction(lambda a,b: loss_fn_wrap_nodevice(a, b), "dlrm_loss", torch.zeros((1,1)))
    job, iterMs, gpuMs, maxGpusUsed = cs.searchBestSplitsV3(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)
    print("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, maxGpusUsed, iterMs, gpuMs))
    cs.to_dot("Digraph", globalBatch)
    # cs.to_gpuTimeline("Inception v3, Burst Parallel", maxGpusUsed, dataParallelBaseline)
    jobInJson = job.dumpInJSON()

    # for rank in range(gpuCount):
    #     print("GPU rank: %d"%rank)
    #     print(job.dumpSingleRunnableModule(rank))

    job2 = TrainingJob("test", None, None, 0, 0, "")
    job2.loadJSON(jobInJson)
    assert(jobInJson == job2.dumpInJSON())
    print("Load/Dump returned the same output? %s" % ("true" if jobInJson == job2.dumpInJSON() else "false"))
    # print(jobInJson)
    
    if maxGpusUsed > 8:
        print("maxGpusUsed: ", maxGpusUsed, " is bigger than 8. Can't schedule this job.")
        exit(-1)
    
    if not spatialSplit and not simOnly:
        cc = ClusterClient()
        jobName = "DLRM_%d_%d_%2.1f%s" % (gpuCount, globalBatch, amplificationLimit, "_DP" if dataParallelBaseline else "")
        jobName += "_BE" if use_be else ""
        cc.submitTrainingJob(jobName, jobInJson, use_be)

    # cs.printAllLayers(slient=False)
    # cs.computeInputDimensions((2,240,240))
    # cs.to_dot("Digraph", globalBatch, justdag=True)

    # if use_gpu:
    #     # Custom Model-Data Parallel
    #     # the mlps are replicated and use data parallelism, while
    #     # the embeddings are distributed and use model parallelism
    #     dlrm = dlrm.to(device)  # .cuda()
    #     if dlrm.ndevices > 1:
    #         dlrm.emb_l, dlrm.v_W_l = dlrm.create_emb(
    #             m_spa, ln_emb, args.weighted_pooling
    #         )
    #     else:
    #         if dlrm.weighted_pooling == "fixed":
    #             for k, w in enumerate(dlrm.v_W_l):
    #                 dlrm.v_W_l[k] = w.cuda()

    # # distribute data parallel mlps
    # if ext_dist.my_size > 1:
    #     if use_gpu:
    #         device_ids = [ext_dist.my_local_rank]
    #         dlrm.bot_l = ext_dist.DDP(dlrm.bot_l, device_ids=device_ids)
    #         dlrm.top_l = ext_dist.DDP(dlrm.top_l, device_ids=device_ids)
    #     else:
    #         dlrm.bot_l = ext_dist.DDP(dlrm.bot_l)
    #         dlrm.top_l = ext_dist.DDP(dlrm.top_l)

    # if not args.inference_only:
    #     if use_gpu and args.optimizer in ["rwsadagrad", "adagrad"]:
    #         sys.exit("GPU version of Adagrad is not supported by PyTorch.")
    #     # specify the optimizer algorithm
    #     opts = {
    #         "sgd": torch.optim.SGD,
    #         # "rwsadagrad": RowWiseSparseAdagrad.RWSAdagrad,
    #         "adagrad": torch.optim.Adagrad,
    #     }

    #     parameters = (
    #         dlrm.parameters()
    #         if ext_dist.my_size == 1
    #         else [
    #             {
    #                 "params": [p for emb in dlrm.emb_l for p in emb.parameters()],
    #                 "lr": args.learning_rate,
    #             },
    #             # TODO check this lr setup
    #             # bottom mlp has no data parallelism
    #             # need to check how do we deal with top mlp
    #             {
    #                 "params": dlrm.bot_l.parameters(),
    #                 "lr": args.learning_rate,
    #             },
    #             {
    #                 "params": dlrm.top_l.parameters(),
    #                 "lr": args.learning_rate,
    #             },
    #         ]
    #     )
    #     optimizer = opts[args.optimizer](parameters, lr=args.learning_rate)
    #     lr_scheduler = LRPolicyScheduler(
    #         optimizer,
    #         args.lr_num_warmup_steps,
    #         args.lr_decay_start_step,
    #         args.lr_num_decay_steps,
    #     )

    ### main loop ###

    # # training or inference
    # best_acc_test = 0
    # best_auc_test = 0
    # skip_upto_epoch = 0
    # skip_upto_batch = 0
    # total_time = 0
    # total_loss = 0
    # total_iter = 0
    # total_samp = 0

    # if args.mlperf_logging:
    #     mlperf_logger.mlperf_submission_log("dlrm")
    #     mlperf_logger.log_event(
    #         key=mlperf_logger.constants.SEED, value=args.numpy_rand_seed
    #     )
    #     mlperf_logger.log_event(
    #         key=mlperf_logger.constants.GLOBAL_BATCH_SIZE, value=args.mini_batch_size
    #     )

    # # Load model is specified
    # if not (args.load_model == ""):
    #     print("Loading saved model {}".format(args.load_model))
    #     if use_gpu:
    #         if dlrm.ndevices > 1:
    #             # NOTE: when targeting inference on multiple GPUs,
    #             # load the model as is on CPU or GPU, with the move
    #             # to multiple GPUs to be done in parallel_forward
    #             ld_model = torch.load(args.load_model)
    #         else:
    #             # NOTE: when targeting inference on single GPU,
    #             # note that the call to .to(device) has already happened
    #             ld_model = torch.load(
    #                 args.load_model,
    #                 map_location=torch.device("cuda")
    #                 # map_location=lambda storage, loc: storage.cuda(0)
    #             )
    #     else:
    #         # when targeting inference on CPU
    #         ld_model = torch.load(args.load_model, map_location=torch.device("cpu"))
    #     dlrm.load_state_dict(ld_model["state_dict"])
    #     ld_j = ld_model["iter"]
    #     ld_k = ld_model["epoch"]
    #     ld_nepochs = ld_model["nepochs"]
    #     ld_nbatches = ld_model["nbatches"]
    #     ld_nbatches_test = ld_model["nbatches_test"]
    #     ld_train_loss = ld_model["train_loss"]
    #     ld_total_loss = ld_model["total_loss"]
    #     if args.mlperf_logging:
    #         ld_gAUC_test = ld_model["test_auc"]
    #     ld_acc_test = ld_model["test_acc"]
    #     if not args.inference_only:
    #         optimizer.load_state_dict(ld_model["opt_state_dict"])
    #         best_acc_test = ld_acc_test
    #         total_loss = ld_total_loss
    #         skip_upto_epoch = ld_k  # epochs
    #         skip_upto_batch = ld_j  # batches
    #     else:
    #         args.print_freq = ld_nbatches
    #         args.test_freq = 0

    #     print(
    #         "Saved at: epoch = {:d}/{:d}, batch = {:d}/{:d}, ntbatch = {:d}".format(
    #             ld_k, ld_nepochs, ld_j, ld_nbatches, ld_nbatches_test
    #         )
    #     )
    #     print(
    #         "Training state: loss = {:.6f}".format(
    #             ld_train_loss,
    #         )
    #     )
    #     if args.mlperf_logging:
    #         print(
    #             "Testing state: accuracy = {:3.3f} %, auc = {:.3f}".format(
    #                 ld_acc_test * 100, ld_gAUC_test
    #             )
    #         )
    #     else:
    #         print("Testing state: accuracy = {:3.3f} %".format(ld_acc_test * 100))

    # if args.inference_only:
    #     # Currently only dynamic quantization with INT8 and FP16 weights are
    #     # supported for MLPs and INT4 and INT8 weights for EmbeddingBag
    #     # post-training quantization during the inference.
    #     # By default we don't do the quantization: quantize_{mlp,emb}_with_bit == 32 (FP32)
    #     assert args.quantize_mlp_with_bit in [
    #         8,
    #         16,
    #         32,
    #     ], "only support 8/16/32-bit but got {}".format(args.quantize_mlp_with_bit)
    #     assert args.quantize_emb_with_bit in [
    #         4,
    #         8,
    #         32,
    #     ], "only support 4/8/32-bit but got {}".format(args.quantize_emb_with_bit)
    #     if args.quantize_mlp_with_bit != 32:
    #         if args.quantize_mlp_with_bit in [8]:
    #             quantize_dtype = torch.qint8
    #         else:
    #             quantize_dtype = torch.float16
    #         dlrm = torch.quantization.quantize_dynamic(
    #             dlrm, {torch.nn.Linear}, quantize_dtype
    #         )
    #     if args.quantize_emb_with_bit != 32:
    #         dlrm.quantize_embedding(args.quantize_emb_with_bit)
    #         # print(dlrm)

    # print("time/loss/accuracy (if enabled):")

    # if args.mlperf_logging:
    #     # LR is logged twice for now because of a compliance checker bug
    #     mlperf_logger.log_event(
    #         key=mlperf_logger.constants.OPT_BASE_LR, value=args.learning_rate
    #     )
    #     mlperf_logger.log_event(
    #         key=mlperf_logger.constants.OPT_LR_WARMUP_STEPS,
    #         value=args.lr_num_warmup_steps,
    #     )

    #     # use logging keys from the official HP table and not from the logging library
    #     mlperf_logger.log_event(
    #         key="sgd_opt_base_learning_rate", value=args.learning_rate
    #     )
    #     mlperf_logger.log_event(
    #         key="lr_decay_start_steps", value=args.lr_decay_start_step
    #     )
    #     mlperf_logger.log_event(
    #         key="sgd_opt_learning_rate_decay_steps", value=args.lr_num_decay_steps
    #     )
    #     mlperf_logger.log_event(key="sgd_opt_learning_rate_decay_poly_power", value=2)

    # tb_file = "./" + args.tensor_board_filename
    # writer = SummaryWriter(tb_file)

    # ext_dist.barrier()
    # with torch.autograd.profiler.profile(
    #     args.enable_profiling, use_cuda=use_gpu, record_shapes=True
    # ) as prof:
    #     if not args.inference_only:
    #         k = 0
    #         total_time_begin = 0
    #         while k < args.nepochs:
    #             if args.mlperf_logging:
    #                 mlperf_logger.barrier()
    #                 mlperf_logger.log_start(
    #                     key=mlperf_logger.constants.BLOCK_START,
    #                     metadata={
    #                         mlperf_logger.constants.FIRST_EPOCH_NUM: (k + 1),
    #                         mlperf_logger.constants.EPOCH_COUNT: 1,
    #                     },
    #                 )
    #                 mlperf_logger.barrier()
    #                 mlperf_logger.log_start(
    #                     key=mlperf_logger.constants.EPOCH_START,
    #                     metadata={mlperf_logger.constants.EPOCH_NUM: (k + 1)},
    #                 )

    #             if k < skip_upto_epoch:
    #                 continue

    #             if args.mlperf_logging:
    #                 previous_iteration_time = None

    #             for j, inputBatch in enumerate(train_ld):
    #                 if j == 0 and args.save_onnx:
    #                     X_onnx, lS_o_onnx, lS_i_onnx, _, _, _ = unpack_batch(inputBatch)

    #                 if j < skip_upto_batch:
    #                     continue

    #                 X, lS_o, lS_i, T, W, CBPP = unpack_batch(inputBatch)

    #                 if args.mlperf_logging:
    #                     current_time = time_wrap(use_gpu)
    #                     if previous_iteration_time:
    #                         iteration_time = current_time - previous_iteration_time
    #                     else:
    #                         iteration_time = 0
    #                     previous_iteration_time = current_time
    #                 else:
    #                     t1 = time_wrap(use_gpu)

    #                 # early exit if nbatches was set by the user and has been exceeded
    #                 if nbatches > 0 and j >= nbatches:
    #                     break

    #                 # Skip the batch if batch size not multiple of total ranks
    #                 if ext_dist.my_size > 1 and X.size(0) % ext_dist.my_size != 0:
    #                     print(
    #                         "Warning: Skiping the batch %d with size %d"
    #                         % (j, X.size(0))
    #                     )
    #                     continue

    #                 mbs = T.shape[0]  # = args.mini_batch_size except maybe for last

    #                 # forward pass
    #                 Z = dlrm_wrap(
    #                     X,
    #                     lS_o,
    #                     lS_i,
    #                     use_gpu,
    #                     device,
    #                     ndevices=ndevices,
    #                 )

    #                 if ext_dist.my_size > 1:
    #                     T = T[ext_dist.get_my_slice(mbs)]
    #                     W = W[ext_dist.get_my_slice(mbs)]

    #                 # loss
    #                 E = loss_fn_wrap(Z, T, use_gpu, device)

    #                 # compute loss and accuracy
    #                 L = E.detach().cpu().numpy()  # numpy array
    #                 # training accuracy is not disabled
    #                 # S = Z.detach().cpu().numpy()  # numpy array
    #                 # T = T.detach().cpu().numpy()  # numpy array

    #                 # # print("res: ", S)

    #                 # # print("j, train: BCE ", j, L)

    #                 # mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
    #                 # A = np.sum((np.round(S, 0) == T).astype(np.uint8))

    #                 with record_function("DLRM backward"):
    #                     # scaled error gradient propagation
    #                     # (where we do not accumulate gradients across mini-batches)
    #                     if (args.mlperf_logging and (j + 1) % args.mlperf_grad_accum_iter == 0) or not args.mlperf_logging:
    #                         optimizer.zero_grad()
    #                     # backward pass
    #                     E.backward(retain_graph=True)

    #                     # optimizer
    #                     if (args.mlperf_logging and (j + 1) % args.mlperf_grad_accum_iter == 0) or not args.mlperf_logging:
    #                         optimizer.step()
    #                         lr_scheduler.step()

    #                 if args.mlperf_logging:
    #                     total_time += iteration_time
    #                 else:
    #                     t2 = time_wrap(use_gpu)
    #                     total_time += t2 - t1

    #                 total_loss += L * mbs
    #                 total_iter += 1
    #                 total_samp += mbs

    #                 should_print = ((j + 1) % args.print_freq == 0) or (
    #                     j + 1 == nbatches
    #                 )
    #                 should_test = (
    #                     (args.test_freq > 0)
    #                     and (args.data_generation in ["dataset", "random"])
    #                     and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
    #                 )

    #                 # print time, loss and accuracy
    #                 if should_print or should_test:
    #                     gT = 1000.0 * total_time / total_iter if args.print_time else -1
    #                     total_time = 0

    #                     train_loss = total_loss / total_samp
    #                     total_loss = 0

    #                     str_run_type = (
    #                         "inference" if args.inference_only else "training"
    #                     )

    #                     wall_time = ""
    #                     if args.print_wall_time:
    #                         wall_time = " ({})".format(time.strftime("%H:%M"))

    #                     print(
    #                         "Finished {} it {}/{} of epoch {}, {:.2f} ms/it,".format(
    #                             str_run_type, j + 1, nbatches, k, gT
    #                         )
    #                         + " loss {:.6f}".format(train_loss)
    #                         + wall_time,
    #                         flush=True,
    #                     )

    #                     log_iter = nbatches * k + j + 1
    #                     writer.add_scalar("Train/Loss", train_loss, log_iter)

    #                     total_iter = 0
    #                     total_samp = 0

    #                 # testing
    #                 if should_test:
    #                     epoch_num_float = (j + 1) / len(train_ld) + k + 1
    #                     if args.mlperf_logging:
    #                         mlperf_logger.barrier()
    #                         mlperf_logger.log_start(
    #                             key=mlperf_logger.constants.EVAL_START,
    #                             metadata={
    #                                 mlperf_logger.constants.EPOCH_NUM: epoch_num_float
    #                             },
    #                         )

    #                     # don't measure training iter time in a test iteration
    #                     if args.mlperf_logging:
    #                         previous_iteration_time = None
    #                     print(
    #                         "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k)
    #                     )
    #                     model_metrics_dict, is_best = inference(
    #                         args,
    #                         dlrm,
    #                         best_acc_test,
    #                         best_auc_test,
    #                         test_ld,
    #                         device,
    #                         use_gpu,
    #                         log_iter,
    #                     )

    #                     if (
    #                         is_best
    #                         and not (args.save_model == "")
    #                         and not args.inference_only
    #                     ):
    #                         model_metrics_dict["epoch"] = k
    #                         model_metrics_dict["iter"] = j + 1
    #                         model_metrics_dict["train_loss"] = train_loss
    #                         model_metrics_dict["total_loss"] = total_loss
    #                         model_metrics_dict[
    #                             "opt_state_dict"
    #                         ] = optimizer.state_dict()
    #                         print("Saving model to {}".format(args.save_model))
    #                         torch.save(model_metrics_dict, args.save_model)

    #                     if args.mlperf_logging:
    #                         mlperf_logger.barrier()
    #                         mlperf_logger.log_end(
    #                             key=mlperf_logger.constants.EVAL_STOP,
    #                             metadata={
    #                                 mlperf_logger.constants.EPOCH_NUM: epoch_num_float
    #                             },
    #                         )

    #                     # Uncomment the line below to print out the total time with overhead
    #                     # print("Total test time for this group: {}" \
    #                     # .format(time_wrap(use_gpu) - accum_test_time_begin))

    #                     if (
    #                         args.mlperf_logging
    #                         and (args.mlperf_acc_threshold > 0)
    #                         and (best_acc_test > args.mlperf_acc_threshold)
    #                     ):
    #                         print(
    #                             "MLPerf testing accuracy threshold "
    #                             + str(args.mlperf_acc_threshold)
    #                             + " reached, stop training"
    #                         )
    #                         break

    #                     if (
    #                         args.mlperf_logging
    #                         and (args.mlperf_auc_threshold > 0)
    #                         and (best_auc_test > args.mlperf_auc_threshold)
    #                     ):
    #                         print(
    #                             "MLPerf testing auc threshold "
    #                             + str(args.mlperf_auc_threshold)
    #                             + " reached, stop training"
    #                         )
    #                         if args.mlperf_logging:
    #                             mlperf_logger.barrier()
    #                             mlperf_logger.log_end(
    #                                 key=mlperf_logger.constants.RUN_STOP,
    #                                 metadata={
    #                                     mlperf_logger.constants.STATUS: mlperf_logger.constants.SUCCESS
    #                                 },
    #                             )
    #                         break

    #             if args.mlperf_logging:
    #                 mlperf_logger.barrier()
    #                 mlperf_logger.log_end(
    #                     key=mlperf_logger.constants.EPOCH_STOP,
    #                     metadata={mlperf_logger.constants.EPOCH_NUM: (k + 1)},
    #                 )
    #                 mlperf_logger.barrier()
    #                 mlperf_logger.log_end(
    #                     key=mlperf_logger.constants.BLOCK_STOP,
    #                     metadata={mlperf_logger.constants.FIRST_EPOCH_NUM: (k + 1)},
    #                 )
    #             k += 1  # nepochs
    #         if args.mlperf_logging and best_auc_test <= args.mlperf_auc_threshold:
    #             mlperf_logger.barrier()
    #             mlperf_logger.log_end(
    #                 key=mlperf_logger.constants.RUN_STOP,
    #                 metadata={
    #                     mlperf_logger.constants.STATUS: mlperf_logger.constants.ABORTED
    #                 },
    #             )
    #     else:
    #         print("Testing for inference only")
    #         inference(
    #             args,
    #             dlrm,
    #             best_acc_test,
    #             best_auc_test,
    #             test_ld,
    #             device,
    #             use_gpu,
    #         )

    # # profiling
    # if args.enable_profiling:
    #     time_stamp = str(datetime.datetime.now()).replace(" ", "_")
    #     with open("dlrm_s_pytorch" + time_stamp + "_shape.prof", "w") as prof_f:
    #         prof_f.write(
    #             prof.key_averages(group_by_input_shape=True).table(
    #                 sort_by="self_cpu_time_total"
    #             )
    #         )
    #     with open("dlrm_s_pytorch" + time_stamp + "_total.prof", "w") as prof_f:
    #         prof_f.write(prof.key_averages().table(sort_by="self_cpu_time_total"))
    #     prof.export_chrome_trace("dlrm_s_pytorch" + time_stamp + ".json")
    #     # print(prof.key_averages().table(sort_by="cpu_time_total"))

    # # plot compute graph
    # if args.plot_compute_graph:# and args.local_rank == 0:
    #     sys.exit(
    #         "ERROR: Please install pytorchviz package in order to use the"
    #         + " visualization. Then, uncomment its import above as well as"
    #         + " three lines below and run the code again."
    #     )
    #     # V = Z.mean() if args.inference_only else E
    #     # dot = make_dot(V, params=dict(dlrm.named_parameters()))
    #     # dot.render('dlrm_s_pytorch_graph') # write .pdf file

    # # test prints
    # if not args.inference_only and args.debug_mode:
    #     print("updated parameters (weights and bias):")
    #     for param in dlrm.parameters():
    #         print(param.detach().cpu().numpy())

    # # export the model in onnx
    # if args.save_onnx:
    #     """
    #     # workaround 1: tensor -> list
    #     if torch.is_tensor(lS_i_onnx):
    #         lS_i_onnx = [lS_i_onnx[j] for j in range(len(lS_i_onnx))]
    #     # workaound 2: list -> tensor
    #     lS_i_onnx = torch.stack(lS_i_onnx)
    #     """
    #     # debug prints
    #     # print("inputs", X_onnx, lS_o_onnx, lS_i_onnx)
    #     # print("output", dlrm_wrap(X_onnx, lS_o_onnx, lS_i_onnx, use_gpu, device))
    #     dlrm_pytorch_onnx_file = "dlrm_s_pytorch.onnx"
    #     batch_size = X_onnx.shape[0]
    #     print("X_onnx.shape", X_onnx.shape)
    #     if torch.is_tensor(lS_o_onnx):
    #         print("lS_o_onnx.shape", lS_o_onnx.shape)
    #     else:
    #         for oo in lS_o_onnx:
    #             print("oo.shape", oo.shape)
    #     if torch.is_tensor(lS_i_onnx):
    #         print("lS_i_onnx.shape", lS_i_onnx.shape)
    #     else:
    #         for ii in lS_i_onnx:
    #             print("ii.shape", ii.shape)

    #     # name inputs and outputs
    #     o_inputs = (
    #         ["offsets"]
    #         if torch.is_tensor(lS_o_onnx)
    #         else ["offsets_" + str(i) for i in range(len(lS_o_onnx))]
    #     )
    #     i_inputs = (
    #         ["indices"]
    #         if torch.is_tensor(lS_i_onnx)
    #         else ["indices_" + str(i) for i in range(len(lS_i_onnx))]
    #     )
    #     all_inputs = ["dense_x"] + o_inputs + i_inputs
    #     # debug prints
    #     print("inputs", all_inputs)

    #     # create dynamic_axis dictionaries
    #     do_inputs = (
    #         [{"offsets": {1: "batch_size"}}]
    #         if torch.is_tensor(lS_o_onnx)
    #         else [
    #             {"offsets_" + str(i): {0: "batch_size"}} for i in range(len(lS_o_onnx))
    #         ]
    #     )
    #     di_inputs = (
    #         [{"indices": {1: "batch_size"}}]
    #         if torch.is_tensor(lS_i_onnx)
    #         else [
    #             {"indices_" + str(i): {0: "batch_size"}} for i in range(len(lS_i_onnx))
    #         ]
    #     )
    #     dynamic_axes = {"dense_x": {0: "batch_size"}, "pred": {0: "batch_size"}}
    #     for do in do_inputs:
    #         dynamic_axes.update(do)
    #     for di in di_inputs:
    #         dynamic_axes.update(di)
    #     # debug prints
    #     print(dynamic_axes)
    #     # export model
    #     torch.onnx.export(
    #         dlrm,
    #         (X_onnx, lS_o_onnx, lS_i_onnx),
    #         dlrm_pytorch_onnx_file,
    #         verbose=True,
    #         use_external_data_format=True,
    #         opset_version=11,
    #         input_names=all_inputs,
    #         output_names=["pred"],
    #         dynamic_axes=dynamic_axes,
    #     )
    #     # recover the model back
    #     dlrm_pytorch_onnx = onnx.load("dlrm_s_pytorch.onnx")
    #     # check the onnx model
    #     onnx.checker.check_model(dlrm_pytorch_onnx)
    # total_time_end = time_wrap(use_gpu)



# def main(gpuCount, globalBatch, amplificationLimit=2.0, dataParallelBaseline=False, netBw=2.66E5, spatialSplit=False, simResultFilename=None, simOnly=False, use_be=False):
#     # profiler = GpuProfiler("cuda")
#     # profiler.loadProfile()
#     global cs
#     cs = CostSim(None, netBw=netBw, verbose=True, gpuProfileLoc="profile/A100_anvil.prof")#, gpuProfileLocSub="resnetLayerGpuProfileA100.txt")
#     model = resnet34(pretrained='/anvil/data/pretrained/resnet34.pth')
#     # model.load_state_dict(torch.load('/anvil/planes_checkpoints_0_997AUC/weights_0499.pth'), strict=True)
#     # model.load_state_dict(torch.load('/DeepPool/weights_0499.pth'), strict=True)
#     # model = resnet152()
#     # model = wide_resnet101_2()
#     # cs.printAllLayers(slient=True)
#     cs.printAllLayers(slient=False)
#     cs.computeInputDimensions((2,240,240))
#     cs.to_dot("Digraph", globalBatch, justdag=True)

#     # job, iterMs, gpuMs = cs.searchBestSplits(gpuCount, globalBatch, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit, lossfn=0)
#     # job, iterMs, gpuMs, maxGpusUsed = cs.searchBestSplitsV3(gpuCount, globalBatch, lossfn=2, amplificationLimit=amplificationLimit, dataParallelBaseline=dataParallelBaseline, spatialSplit=spatialSplit)
#     job, iterMs, gpuMs, maxGpusUsed = cs.JustDoDP(gpuCount, globalBatch, lossfn=2)
#     print("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, maxGpusUsed, iterMs, gpuMs))

#     jobInJson = job.dumpInJSON()
#     # profiler.saveProfile()
#     # for rank in range(4):
#     #     print("GPU rank: %d"%rank)
#     #     print(job.dumpSingleRunnableModule(rank))

#     # job2 = TrainingJob("test", None, None, 0, 0, "")
#     # job2.loadJSON(jobInJson)
#     # assert(jobInJson == job2.dumpInJSON())
#     # print("Load/Dump returned the same output? %s" % ("true" if jobInJson == job2.dumpInJSON() else "false"))
#     # print(jobInJson)
    
#     if not spatialSplit and not simOnly:
#         cc = ClusterClient()
#         jobName = "anvil_%d_%d_%2.1f%s" % (gpuCount, globalBatch, amplificationLimit, "_DP" if dataParallelBaseline else "")
#         jobName += "_BE" if use_be else ""
#         cc.submitTrainingJob(jobName, jobInJson, use_be)

#     if simResultFilename != None:
#         f = open(simResultFilename, "a")
#         f.write("  %2d    %2d   %4.1f  %4.1f\n" % (globalBatch, gpuCount, iterMs, gpuMs))
#         f.close()

#         if gpuCount == 8:
#             f = open(simResultFilename, "r")
#             print(f.read())
#             f.close()


if __name__ == "__main__":

    print(len(sys.argv))
    # if len(sys.argv) == 3:
    run(4, dataParallelBaseline=True)
    # elif len(sys.argv) >= 4:
    #     use_be = len(sys.argv) > 4 and int(sys.argv[4]) == 1
    #     if sys.argv[3] == "DP":
    #         main(int(sys.argv[1]), int(sys.argv[2]), dataParallelBaseline=True, use_be=use_be)
    #     else:
    #         main(int(sys.argv[1]), int(sys.argv[2]), amplificationLimit=float(sys.argv[3]), use_be=use_be)
    # # elif len(sys.argv) == 2:
    # #     print("Run all configs")
    # #     runAllConfigs("anvil", sys.argv[1])
    # # elif len(sys.argv) == 1:
    # #     generateJit()
    # else:
    #     print("Wrong number of arguments.\nUsage: ")

    # run()
