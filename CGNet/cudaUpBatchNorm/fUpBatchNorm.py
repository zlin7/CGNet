# functions/add.py
import torch
import CGNetLayer_cuda
import pdb
import numpy as np
import torch.nn as nn
import time

class fUpFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, maxL, tauIn, tauMid, middle_length, tauOut,
                weights, activations,
                moving_std, bmcount, bm_eps, update_std):
        assert(weights.is_cuda and weights.requires_grad)
        assert len(tauIn) == len(tauOut) == len(tauMid)

        tauIn = torch.tensor(tauIn, dtype=torch.int)
        tauOut = torch.tensor(tauOut, dtype=torch.int)

        output_length = np.sum(np.asarray([tauOut[l] * (2 * l + 1) for l in range(maxL + 1)]))
    
        middle = torch.zeros(activations.shape[0], middle_length,
                             2, dtype=torch.float, device=torch.device('cuda'))

        output = torch.zeros(activations.shape[0], output_length,
                             2, dtype=torch.float, device=torch.device('cuda'),
                             requires_grad=True)
        assert(not moving_std.requires_grad)
        CGNetLayer_cuda.forward(activations, middle, output, weights,
                                maxL, activations.shape[0], tauIn, tauOut,
                                moving_std, bmcount, bm_eps,
                                1 if update_std else 0)
        ctx.save_for_backward(tauIn, tauOut, torch.tensor(middle.shape, dtype=torch.int),
                              weights, activations, moving_std.clone(), torch.tensor([bm_eps], dtype=torch.float))
        del middle
        return output

    @staticmethod
    def backward(ctx, grad_output):
        tauIn, tauOut, middle_shape, weights, activations, moving_std, bm_eps = ctx.saved_tensors
        maxL = len(tauIn) - 1 #TODO: check this!!!
        bm_eps = bm_eps.item()
        assert isinstance(bm_eps, float)

        grad_input = torch.zeros(activations.shape,dtype=torch.float,device=torch.device('cuda'))
        grad_weight = torch.zeros(weights.shape,dtype=torch.float,device=torch.device('cuda'))

        grad_middle = torch.zeros(middle_shape[0]*middle_shape[1], 2, dtype=torch.float, device=torch.device('cuda'))
        CGlength = 0
        for l1 in range(maxL+1):
            for l2 in range(l1+1):
                for l in range(l1-l2, min(l1+l2, maxL)+1):
                    CGlength += (2*l+1)*(2*l2+1)
        CGspace = torch.zeros(CGlength, dtype=torch.float, device=torch.device('cuda'))
        CGNetLayer_cuda.backward(weights, activations, grad_input,
                                 grad_weight, grad_middle, grad_output, CGspace,
                                 maxL, activations.shape[0], tauIn, tauOut,
                                 moving_std, bm_eps)
        del grad_middle
        del CGspace
        return None, None, None, None, None, grad_weight, grad_input, None, None, None, None


class fUpModule(nn.Module):
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

        torch.manual_seed(0)
        wlength = np.sum(self.out_taus * self.middle_taus)
        self.weights = nn.Parameter(weight_scale*torch.rand(wlength,2,device=torch.device('cuda'), 
                                                        dtype=torch.float),
                                    requires_grad=True)

        np.random.seed(0)
        self.batchnorm=batchnorm
        if self.batchnorm:
            bm_np = np.random.rand(np.sum(self.middle_taus))
            self.moving_std = nn.Parameter(torch.tensor(bm_np,device=torch.device('cuda'), dtype=torch.float), requires_grad=False)
        else:
            self.moving_std = None
        self.bm_eps = 1e-5
        self.bm_cnt = 1.

        self.layername = layername
        self.cuda()

    def summary(self):
        print("batch normalization?: {}".format(None if self.bmlayer_scale is None else self.bmlayer_scale[-1].shape))
        print("weight shapes: {}".format(self.weights.shape))

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

    def reset_parameters(self, scale=0.05):
        pass

    def forward(self, activations, straight_output=False):
        weights = self.weights
        new_activations = self.reshpae_in(activations) if isinstance(activations,list) else activations
        assert(new_activations.shape[1] == self.cum_taus[-1])
        output = fUpFunction.apply(self.maxL, self.taus, self.middle_taus, self.cum_middle_taus[-1], self.out_taus,
                                   weights, new_activations,
                                   self.moving_std, self.bm_cnt, self.bm_eps, self.training)
        #THIS IS VERY IMPORTANT!!
        self.bm_cnt+=1
        if straight_output:
            return output
        output = self.reshape_out(output)
        return output

#TODO: free cpu memory