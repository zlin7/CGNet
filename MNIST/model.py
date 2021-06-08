import sys
import os

CUR_DIR = os.path.dirname(os.path.realpath(__file__))
REAL_PART = 0
IMAG_PART = 1
import torch.nn as nn
import torch
import numpy as np
import ipdb
from importlib import reload
import CGNet.CGNet as cgnet
import functools


@functools.lru_cache(10)
def define_taus(mode, param, lmax):
    if mode == 1:
        taus = [param for l in range(lmax + 1)]
    elif mode == 2:
        taus = [int(np.ceil(param / (2 * l + 1.))) for l in range(lmax + 1)]
    elif mode == 3:
        taus = [int(np.ceil(param / np.sqrt(2 * l + 1.))) for l in range(lmax + 1)]
    else:
        raise ValueError("only defined 3 modes: 1,2,3")
    return taus

class MNIST_Net(nn.Module):
    def __init__(self, lmax, tau_type=3, tau_man=12,
                 nlayers=5,
                 skipconn=True, norm=True, cuda=True,
                 dropout=0.5, nfc = 1, sparse=False):
        super(MNIST_Net, self).__init__()

        taus = define_taus(tau_type, tau_man, lmax)

        taus = [[1 for _ in range(lmax + 1)]] + [taus for _ in range(nlayers)]
        assert cuda and norm and skipconn,  "Do not support these parameters yet"
        self.SphericalCNN = cgnet.SphericalCNN(lmax, taus, cuda=cuda, norm=norm, skipconn=skipconn, sparse_flag=sparse)

        if norm:
            self.bm1 = nn.BatchNorm1d(self.SphericalCNN.output_length)
        else:
            self.bm1 = None

        self.fcs = nn.ModuleList([])
        self.dropout_layers = None if dropout is None else nn.ModuleList([])
        for layer in range(nfc):
            self.fcs.append(nn.Linear(self.SphericalCNN.output_length if layer == 0 else 256, 256))
            if dropout is not None:
                self.dropout_layers.append(nn.Dropout(p=dropout))
        print("Fully connected layser\n", self.fcs)
        print("Paired with dropout layers? \n", self.dropout_layers)
        self.final = nn.Linear(256, 10)
        if cuda:
            self.cuda()

    def forward(self, x):
        x = self.SphericalCNN(x)
        if self.bm1 is not None:
            x = self.bm1(x)
        for layer in range(len(self.fcs)):
            x = torch.relu(self.fcs[layer](x))
            if self.dropout_layers is not None:
                x = self.dropout_layers[layer](x)
        x = nn.LogSoftmax(dim=1)(self.final(x))
        return x


def show_num_parameters(net):
    shapes = [x.cpu().detach().numpy().shape for x in list(filter(lambda p: p.requires_grad, net.parameters()))]
    def get_volume(shape):
        v = 1
        for i in range(len(shape)):
            v *= shape[i]
        return v
    print("Shapes of parameters", shapes, np.sum(np.asarray([get_volume(s) for s in shapes])))
    return net