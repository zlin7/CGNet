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

import CGNet.CG_layers as cglayers#; reload(cglayers)

class SphericalCNN(nn.Module):
    def __init__(self, lmax, taus,
                 cuda=True,
                 norm=True,
                 skipconn=True,
                 sparse_flag=False):
        """

        :param lmax:
        :param taus:  list of list
                taus[i] is the # of fragments of each l for (intput of) the i-th layer.
        :param n_layers:
        :param cuda:
        :param norm:
        :param skipconn:
        """
        assert cuda and norm and skipconn,  "Do not support these parameters yet"
        # the maximum l is lmax (i.e. l in range(lmax+1))
        super(SphericalCNN, self).__init__()
        self.lmax = lmax
        self.taus = taus

        # the rest of the layers are like in CGnet (all in cuda)
        self.n_layers = len(taus) - 1
        self.cgs = nn.ModuleList([cglayers.CGBN_cuda2(lmax, taus[layer_i], taus[layer_i + 1], batchnorm=norm,
                                                      sparse_flag=sparse_flag, pythoncg=False)
                                  # self.cgs = nn.ModuleList([cglayers.CGBN_cuda2(lmax, taus[layer_i], taus[layer_i + 1], batchnorm=norm, sparse_flag=sparse_flag, pythoncg=True)
                                  # self.cgs = nn.ModuleList([cglayers.CGBN_cuda(lmax, taus[layer_i], taus[layer_i + 1], batchnorm=norm)
                                  # self.cgs = nn.ModuleList([cglayers.CGBN_base(lmax, taus[layer_i], taus[layer_i + 1], batchnorm=norm, sparse_flag=sparse_flag)
                                  # self.cgs = nn.ModuleList([cglayers.CGBN_base(lmax, taus[layer_i], taus[layer_i + 1], batchnorm=norm, fully_python=True, sparse_flag=True)
                                  for layer_i in range(self.n_layers)])

        # for the skip connection..
        self.skipconn = skipconn
        if self.skipconn:
            self.output_length = 2 * sum([_taus[0] for _taus in taus])
        else:
            self.output_length = 2 * taus[-1][0]

        if cuda:
            self.cuda()

    def forward(self, f_input):
        embedding = []
        if isinstance(f_input, list):
            B = f_input[0].shape[0]
            if self.skipconn: embedding = [f_input[0].view(B, -1)]
        else:
            B = f_input.shape[0]
            if self.skipconn: embedding = [f_input[:, 0:(self.taus[0][0] * (2 * 0 + 1)), :].view(B, -1)]
        fs = f_input

        for i in range(self.n_layers):
            fs = self.cgs[i](fs, straight_output=False)
            # if self.skipconn: embedding.append(fs[:, 0:(self.taus[i+1][0] * (2 * 0 + 1)), :].view(B, -1))
            if self.skipconn: embedding.append(fs[0].view(B, -1))
        if self.skipconn:
            embedding = torch.cat(embedding, 1)
        else:
            # embedding = fs[-1][:, 0:(self.taus[-1][0] * (2 * 0 + 1)), :].view(B, -1)
            embedding = fs[0].view(B, -1)
        return embedding


