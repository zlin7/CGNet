# functions/add.py
import torch
from _ext import cudaUpBatchNorm
import pdb
from torch.nn.modules.module import Module
#import ClebschGordan
import numpy as np
import torch.nn as nn
import time

REAL_PART,IMAG_PART=0,1

class fUpFunction(torch.autograd.Function):
    def __init__(self, maxL, tauIn, tauMid, length, tauOut, 
                    moving_std=None, bmcount=None, eps=1e-5, update_std=False):
        super(fUpFunction, self).__init__()
        self.maxL = maxL
        self.tauIn = tauIn
        self.tauMid = tauMid
        self.middle_length = length

        self.tauOut = tauOut

        self.output_length = np.sum(np.asarray([tauOut[l]*(2*l+1) for l in range(maxL+1)]))

        assert(moving_std is not None)
        self.moving_std = moving_std
        self.bmcount = bmcount
        self.bm_eps = eps
        self.update_std = update_std

    def forward(self, weights, activations):
        assert(weights.is_cuda and weights.requires_grad)

        tauIn = torch.tensor(self.tauIn, dtype=torch.int)
        tauOut = torch.tensor(self.tauOut, dtype=torch.int)
    
        middle = torch.zeros(activations.shape[0], 
                                self.middle_length, 
                                2, dtype=torch.float,
                                device=torch.device('cuda'))

        output = torch.zeros(activations.shape[0], 
                                self.output_length,
                                2, dtype=torch.float,
                                device=torch.device('cuda'),
                                requires_grad=True)
        #print(self.bm_eps)
        moving_std = self.moving_std
        assert(not moving_std.requires_grad)
        #print("input", activations)
        #print("UPDATING????\n",1 if self.update_std else 0)
        #print("moving_std_pre", moving_std)
        #print("input", activations[0][0:5])
        #print("weights", weights[0:5])
        #print("Pre", activations[:10], middle[:10], output[:10])
        cudaUpBatchNorm.cudaUpBatchNorm_forward(activations,
                                                middle, 
                                                output,
                                                weights, 
                                                self.maxL, 
                                                activations.shape[0], 
                                                tauIn, 
                                                tauOut,
                                                moving_std,
                                                self.bmcount,
                                                self.bm_eps,
                                                1 if self.update_std else 0)
        self.bmcount+=1
        self.save_for_backward(weights, activations, moving_std.clone())
        #print("Post", activations[:10], middle[:10], output[:10])

        #print("middle output", middle)
        #print("moving_std", moving_std)
        #print("final output", output)
        del middle
        return output

    def backward(self, grad_output):
        #print(grad_output)
        #print("backward")
        starttime = time.time()
        weights, activations, moving_std = self.saved_tensors

        tauIn = torch.tensor(self.tauIn, dtype=torch.int)
        tauOut = torch.tensor(self.tauOut, dtype=torch.int)

        grad_input = torch.zeros(activations.shape,dtype=torch.float,device=torch.device('cuda'))
        grad_weight = torch.zeros(weights.shape,dtype=torch.float,device=torch.device('cuda'))
        
        #grad_middle = torch.zeros(activations.shape[0], 
        #                        self.middle_length, 
        #                        2, device=torch.device('cuda'))
        grad_middle = torch.zeros(activations.shape[0] * self.middle_length, 2, dtype=torch.float, device=torch.device('cuda'))
        CGlength = 0
        for l1 in range(self.maxL+1):
            for l2 in range(l1+1):
                for l in range(l1-l2, min(l1+l2,self.maxL)+1):
                    CGlength += (2*l+1)*(2*l2+1)
        CGspace = torch.zeros(CGlength, dtype=torch.float, device=torch.device('cuda'))
        #print("backward init", time.time() - starttime)
        starttime = time.time()
        cudaUpBatchNorm.cudaUpBatchNorm_backward(weights,
                                                activations,
                                                grad_input,
                                                grad_weight,
                                                grad_middle,
                                                grad_output,
                                                CGspace,
                                                self.maxL,
                                                activations.shape[0],
                                                tauIn,tauOut,
                                                moving_std,
                                                self.bm_eps)
        """
        pos = 0
        for l1 in range(self.maxL+1):
            for l2 in range(l1+1):
                for l in range(l1-l2, min(l1+l2,self.maxL)+1):
                    for m2 in range(2*l2+1):
                        for m in range(2*l+1):
                            m1 = (m-l)-(m2-l2)
                            if (-l1 <= m1 and m1 <= l1):
                                #print("l1={},l2={},l={},m1={},m2={},m={},CG={}".format(l1,l2,l,m1+l1,m2,m,CGspace[pos]))
                                print("{}{}{}.{}{}{} vs CG={}".format(l1,l2,l,m1+l1,m2,m,CGspace[pos]))
                            pos += 1
        """
        #print("moving_std", moving_std)
        #print("weights", weights)
        #print("activations", activations)
        #print("grad_input", grad_input)
        #print("grad_weight", grad_weight)
        #print("grad_middle", grad_middle)
        #print("grad_output", grad_output)
        #print("GRAD?????",grad_weight, grad_input)
        #print("backward comp", time.time() - starttime)
        del grad_middle
        del CGspace
        #print("backward return", time.time() - starttime)
        return grad_weight, grad_input

class fUpModule(Module):
    def __init__(self, maxL, taus, out_taus, 
                    batchnorm=True, 
                    layername="defaultname",
                    weight_scale=0.05):
        super(fUpModule, self).__init__()
        out_taus = taus if out_taus is None else out_taus
        self.maxL = maxL
        self.taus = taus
        self.cum_taus = np.concatenate([[0], (self.taus * (1+2*np.arange(self.maxL+1))).cumsum()])
        
        self.middle_taus = self.cal_middle_taus(taus)

        self.cum_middle_taus = np.concatenate([[0], (self.middle_taus * (1+2*np.arange(self.maxL+1))).cumsum()])
        
        self.out_taus = out_taus
        self.cum_out_taus = np.concatenate([[0], (self.out_taus * (1+2*np.arange(self.maxL+1))).cumsum()])

        #np.random.seed(0)
        #ws_np = [np.random.rand(self.out_taus[l], self.middle_taus[l], 2) * 0.05 for l in range(maxL+1)]
        #self.weights = nn.Parameter(torch.cat([torch.tensor(w.view(-1, 2),requires_grad=True) for w in self.wrap_listof_np(ws_np)], dim=0))

        #torch.manual_seed(0)
        wlength = np.sum(self.out_taus * self.middle_taus)
        self.weights = nn.Parameter(weight_scale*torch.rand(wlength,2,device=torch.device('cuda'), 
                                                        dtype=torch.float),
                                    requires_grad=True)
        #print(layername+"new weight", self.weights)

        #print(self.weights.shape)
        #print("weights", self.weights[0:10,:])
        np.random.seed(0)
        self.batchnorm=batchnorm
        if self.batchnorm:
            bm_np = np.random.rand(np.sum(self.middle_taus))
            #bm_np = np.ones(np.sum(self.middle_taus),dtype=float)
            self.moving_std = nn.Parameter(torch.tensor(bm_np,device=torch.device('cuda'), dtype=torch.float), requires_grad=False)
            #print(layername+"new bm", self.moving_std)
            #print("bm", self.moving_std[0:3], self.moving_std[9:12])
        else:
            self.moving_std = None
        self.bm_eps = 1e-5
        self.bm_cnt = 1.

        self.layername = layername
        self.cuda()

    def wrap_listof_np(self, npv, dtype=torch.float):
        return nn.ParameterList([nn.Parameter(torch.tensor(npv[l],
                    dtype=dtype, requires_grad=True)) for l in range(self.maxL+1)])


    def cal_middle_taus(self, taus):
        #print(taus)
        middle_taus = np.zeros(self.maxL + 1, dtype=int)
        for l1 in range(self.maxL + 1):
            for l2 in range(l1 + 1):
                for l in range(l1-l2, min(self.maxL, l1+l2)+1):
                    middle_taus[l] += taus[l1] * taus[l2]
        return middle_taus

    def reshpae_in(self, in_list):
        assert(len(in_list) == self.maxL + 1)
        batch_size = in_list[0].shape[0]
        #to_cat = [torch.tensor(v.view(batch_size,-1,2),requires_grad=True,dtype=torch.float) for v in in_list]
        to_cat = [v.view(batch_size,-1,2) for v in in_list]
        return torch.cat(to_cat,dim=1)

    def reshape_out(self,output):
        ret = []
        st = 0
        for l in range(self.maxL+1):
            ed = (2*l+1)*self.out_taus[l] + st
            ret.append(output[:,st:ed,:].view(-1,self.out_taus[l],2*l+1,2))
            st = ed
        return ret

    def forward(self, activations, straight_output=False):
        weights = self.weights
        new_activations = self.reshpae_in(activations) if isinstance(activations,list) else activations
        #print(new_activations.shape)
        #print(self.cum_taus)
        assert(new_activations.shape[1] == self.cum_taus[-1])
        #print(self.layername)
        #print("pre", self.moving_std[0:10])
        #print("Training? {}\n\n\n\n\n".format(self.training))
        output = fUpFunction(self.maxL, self.taus, 
                                self.middle_taus, self.cum_middle_taus[-1],
                                self.out_taus,
                                self.moving_std, self.bm_cnt, self.bm_eps,
                                self.training)(weights,new_activations)
        #THIS IS VERY IMPORTANT!!
        #print(1/0)
        self.bm_cnt+=1
        if straight_output:
            return output
        output = self.reshape_out(output)
        return output

#TODO: free cpu memory