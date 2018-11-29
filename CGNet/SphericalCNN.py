import sys
import os
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
REAL_PART=0
IMAG_PART=1

import numpy as np
import random
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
#import Complex_math as cm
import time

def Complex_bmm(w,f):
    wr, wi = w[:,:,:,REAL_PART], w[:,:,:,IMAG_PART]
    fr, fi = f[:,:,:,REAL_PART], f[:,:,:,IMAG_PART]
    real = torch.bmm(wr, fr) - torch.bmm(wi, fi)
    imag = torch.bmm(wr, fi) + torch.bmm(wi, fr)
    #since f will be of shape (batch_size, tau, 2l+1, 2), multiply from the left
    return torch.stack([real,imag], 3)
def Complex_mm(w,f):
    wr, wi = w[:,:,REAL_PART], w[:,:,IMAG_PART]
    fr, fi = f[:,:,REAL_PART], f[:,:,IMAG_PART]
    real = torch.bmm(wr, fr) - torch.bmm(wi, fi)
    imag = torch.bmm(wr, fi) + torch.bmm(wi, fr)
    #since f will be of shape (batch_size, tau, 2l+1, 2), multiply from the left
    return torch.stack([real,imag], 2)

from cudaCG import fCG

print("loaded SphericalCNN")

class UpdateFunction(nn.Module):
    def __init__(self, lmax, 
                 tau_pre, 
                 tau, 
                 precomputed_CG=None,
                 cudaFlag=False,
                 normalization=0,
                 use_relu=False,
                 layername="defaultname",
                 weight_scale=0.05):
        #Take it as a list of batch tensor (4D)
        super(UpdateFunction, self).__init__()
        self.lmax=lmax
        self.cudaFlag=cudaFlag
        numcols = self._calculate_number_of_cols(tau_pre)

        #print(tau_pre, numcols)
        #np.random.seed(0)
        #ws_np = [np.random.rand(tau[l], numcols[l], 2) * 0.05 for l in range(lmax+1)]
        #self.ws = nn.ParameterList([nn.Parameter(torch.tensor(ws_np[l],
        #            dtype=torch.float, requires_grad=True)) for l in range(lmax+1)])
        self.weight_scale = weight_scale
        wlength = np.sum(np.asarray(numcols) * tau)
        wbig = weight_scale * torch.rand(wlength,2,device=torch.device('cuda'),dtype=torch.float)
        self.ws = nn.ParameterList()
        offset = 0
        for l in range(lmax+1):
            self.ws.append(nn.Parameter(wbig[offset:(offset+numcols[l] * tau[l]),:].view(tau[l],numcols[l],2), requires_grad=True))
            offset += numcols[l] * tau[l]

        np.random.seed(0)
        self.normalization = normalization
        if normalization > 0:
            bm_np_pre = np.random.rand(np.sum(np.asarray(numcols)))
            bm_np = []
            st = 0
            for l in range(lmax+1):
                ed = st + numcols[l]
                bm_np.append(np.expand_dims(np.expand_dims(bm_np_pre[st:ed],1),2))
                st = ed
            self.bmlayer_scale = nn.ParameterList([nn.Parameter(torch.tensor(bm_np[l],
                                        dtype=torch.float), requires_grad=False) for l in range(lmax+1)])
        else:
            self.bmlayer_scale = None
        self.bmlayer_eps = 1e-5 * torch.ones(1, requires_grad=False)
        self.bmlayer_cnt = 1.
        if cudaFlag and normalization > 0:
            self.bmlayer_eps = self.bmlayer_eps.cuda()

        self.use_relu = use_relu

        self.cg = fCG.fCGModule(lmax, tau_pre)

        #self.reset_parameters()
        if cudaFlag:
            self.cuda()
        self.summary()

    def summary(self):
        print("cuda: {}".format(self.cudaFlag))
        print("normalization scheme: {}".format(self.normalization))
        print("batch normalization?: {}".format(None if self.bmlayer_scale is None else self.bmlayer_scale[-1].shape))
        print("weight shapes: {}".format([w.shape for w in self.ws]))

    def _calculate_number_of_cols(self, tau_pre):
        numls = [0 for l in range(self.lmax+1)]
        for l1 in range(self.lmax+1):
            for l2 in range(l1+1):
                for l in range(abs(l1-l2), min(l1+l2+1, self.lmax+1)):
                    numls[l] += tau_pre[l1] * tau_pre[l2]
        return numls
            
    def reset_parameters(self, scale=0.05):
        for l in range(self.lmax+1):
            self.ws[l].data.normal_(0,scale)

    def forward(self, fs):
        assert(isinstance(fs,list))
        batch_size = fs[0].shape[0]

        new_fs = self.cg(fs)

        for l in range(self.lmax+1):
            l_components = new_fs[l]
            #print(l_components.shape)
            if self.normalization > 0:
                if self.normalization == 1:
                    if self.training:

                        npv = l_components.cpu().detach().numpy().copy()
                        norm = np.linalg.norm(npv, ord=2, axis=3)
                        std = torch.tensor(np.std(norm, axis=(0,2))).cuda()
                        
                        #norm = l_components.clone().norm(2, -1)
                        #std = (norm - norm.mean(2, keepdim=True).mean(0, keepdim=True)).pow(2).mean(2).mean(0).pow(0.5)
                        
                        self.bmlayer_scale[l] *= self.bmlayer_cnt/(self.bmlayer_cnt+1.)
                        #print("mid", self.bmlayer_scale[l][0:10])
                        self.bmlayer_scale[l][:,0,0] += std / (self.bmlayer_cnt+1)
                        #print("std old", std.shape)

                        #if l==0:
                        #    print(norm[l][0:5])
                        #    print(std[0:5])
                        #    print("after old", self.bmlayer_scale[l][0:10].squeeze())
                l_components = l_components / torch.max(self.bmlayer_eps, self.bmlayer_scale[l])
                #if l==0:
                #    print("Normalized", l_components[0,0:5,:])

            new_fs[l] = Complex_bmm(self.ws[l].repeat(batch_size,1,1,1), l_components)
            #if l==0:
            #    print("Out", new_fs[l][0,0:5,:])
        #times.append(("matrix mult Done".format(l),time.time()-st))
        #print(times)
        self.bmlayer_cnt += 1
        return new_fs


class SphericalCNN(nn.Module):
    def __init__(self, lmax,
                 taus,
                 n_layers=2,
                 cudaFlag=True,
                 normalization=0,
                 skip_type=0,
                 use_relu=False,
                 num_channels_input=1):
        assert(cudaFlag)
        #the maximum l is lmax
        super(SphericalCNN, self).__init__()
        print("Creating network")
        self.lmax=lmax
        self.cudaFlag=cudaFlag
        # Define Update
        #taus = hidden_state_size if isinstance(hidden_state_size,list) else [[hidden_state_size]*(lmax+1)]*n_layers
        #No filter in even the first layer
        self.us = [UpdateFunction(lmax, 
                                        tau_pre=[num_channels_input]*(lmax+1) if i==0 else taus, 
                                        tau=taus,
                                        cudaFlag=cudaFlag,
                                        normalization=normalization,
                                        use_relu=use_relu,
                                        layername="layer{}".format(i)) for i in range(n_layers)]
        self.us = nn.ModuleList(self.us)

        self.normalize_input = False

        self.skip_connections_to_output = skip_type == 1
        if self.skip_connections_to_output:
            self.output_length = 2 * (num_channels_input+np.sum(np.asarray([self.us[i].ws[0].shape[0] for i in range(n_layers)])))
        else:
            self.output_length = 2 * self.us[-1].ws[0].shape[0]
        
        self.n_layers = n_layers
        if cudaFlag:
            self.cuda()
        print("..done..")
        
    def chop(self, f):
        if self.normalize_input:
            for channel in range(f.shape[1]):
                std = np.linalg.norm(f[:,channel,:,:],ord=2,axis=-1).std()
                f[:,channel,:,:] /= max(1e-5,std)
        return [f[:,:, l**2:(l+1)**2,:] for l in range(self.lmax+1)]

    def _preprocessing(self, f_0):


        #chopping and making single img into a batch of size 1
        #print("original input", [k.shape for k in f_0])
        if isinstance(f_0, np.ndarray):
            d = len(f_0.shape)
            if f_0.dtype == np.complex128:
                f_0 = np.stack([f_0.real, f_0.imag], axis=d)
            #already separated

            #IMPORTANT - what's passed must have channel unsqueezed!
            while (len(f_0.shape) < 4):
                f_0 = np.expand_dims(f_0, 0)
                
            f_0 = self.chop(f_0)
            #print("Chopping into ", [fi.shape for fi in f_0])
        else:
            d = len(f_0[0].shape)
            if f_0[0].dtype == np.complex128:
                f_0 = [np.stack([fi.real, fi.imag],axis=d) for fi in f_0]

            while (len(f_0[0].shape) < 4):
                f_0 = [np.expand_dims(fi, 0) for fi in f_0]
                #Expand the array to (batch, channel, (lmax+1)**2, 2)

        if self.cudaFlag:
            f = [torch.tensor(fi,requires_grad=False,device=torch.device('cuda'),dtype=torch.float) for fi in f_0]
        else:
            f = [torch.tensor(fi,requires_grad=False,dtype=torch.float) for fi in f_0]
        return f

    def forward_test_angle(self, f_0, output_type=0):
        #TODO actually no need to chop it anymore? it's already (batch, ltm, 2)
        #print("f_0", f_0[0][:,0:4])
        f = self._preprocessing(f_0)
        #print("f_0", f[0][0,:,:],f[1][0,:,:])
        fs = [f]
        batch_size = f[0].shape[0]
        for i in range(self.n_layers):
            fs.append(self.us[i](fs[-1]))
        #print(1/0)
        if output_type == 0:
            if self.skip_connections_to_output:
                embedding = torch.cat([f_i[0].squeeze(2).reshape(batch_size, -1) for f_i in fs],1)
            else:
                embedding = fs[-1][0].squeeze(2).reshape(batch_size, -1)
            return embedding
        else:
            f = f if output_type==1 else fs[-1]
            return [1j*f[l][:,:,:,1].numpy() + f[l][:,:,:,0].numpy() for l in range(self.lmax+1)]
    def forward(self, f_0):
        return self.forward_test_angle(f_0, output_type=0)
        

###BELOW IS TO MERGE TO OLD CODE

class SphericalResCNN(nn.Module):
    def __init__(self, lmax, 
                 taus,
                 lmax_step=2,
                 layer_step=2,
                 cudaFlag=True,
                 normalization=1,
                 skip_type=1,
                 num_channels_input=1):
        assert(cudaFlag)
        assert(normalization==1)
        assert(skip_type==1)
        #the maximum l is lmax
        super(SphericalResCNN, self).__init__()
        print("Creating network")
        self.lmax=lmax
        self.lmaxs = [l for l in range(lmax, lmax_step,-lmax_step)]
        self.lmaxs.reverse()
        print("lmax on different layers", self.lmaxs)
        self.nlayers = [layer_step for l in self.lmaxs]
        self.cudaFlag=cudaFlag
        # Define Update
        torch.manual_seed(1)
        self.taus = taus
        self.cum_taus = np.concatenate([[0], (self.taus * (1+2*np.arange(self.lmax+1))).cumsum()])

        self.us = nn.ModuleList([])
        for step in range(len(self.nlayers)):
            cur_lmax = self.lmaxs[step]
            tau_out = [self.taus[l] for l in range(cur_lmax + 1)]
            for layer in range(self.nlayers[step]):
                if layer == 0:
                    tau_in = [num_channels_input for l in range(cur_lmax + 1)]
                    if step > 0:
                        for l in range(self.lmaxs[step-1]+1):
                            tau_in[l] += self.taus[l]
                else:
                    tau_in = [self.taus[l] for l in range(cur_lmax + 1)]
                #print(tau_in, tau_out)
                u = UpdateFunction(cur_lmax, tau_pre = tau_in, tau=tau_out,
                                    normalization=normalization, cudaFlag=cudaFlag)
                self.us.append(u)


        self.normalize_input = True
        self.num_channels_input = num_channels_input

        self.skip_connections_to_output = skip_type == 1
        if self.skip_connections_to_output:
            self.output_length=2*(num_channels_input + self.taus[0] * layer_step * len(self.nlayers))
            #print(self.output_length)
        else:
            self.output_length = 2 * taus[0]
        
        if cudaFlag:
            self.cuda()
        print("..done..")
    
    def _preprocessing(self, f_0, to_torch=True):
        #chopping and making single img into a batch of size 1
        #print("original input", [k.shape for k in f_0])
        assert(isinstance(f_0, np.ndarray))
        d = len(f_0.shape)
        if f_0.dtype == np.complex128:
            f_0 = np.stack([f_0.real, f_0.imag], axis=d)
        #already separated
        if len(f_0.shape) == 3:
            f_0 = np.expand_dims(f_0, 0)
        if not to_torch:
            return f_0

        return torch.tensor(f_0.reshape(f_0.shape[0],-1,2),requires_grad=False,device=torch.device('cuda'),dtype=torch.float)

    def _chop(self, f, lmax, transform=lambda x: x):
        if self.normalize_input:
            for channel in range(self.num_channels_input):
                std = np.linalg.norm(f[:,channel,:,:],ord=2,axis=-1).std()
                f[:,channel,:,:] = f[:,channel,:,:]/ max(1e-4,std)
        return [transform(f[:,:, l**2:(l+1)**2,:]) for l in range(lmax+1)]

    def forward_test_angle(self, f_0, output_type=0):
        f_0 = self._preprocessing(f_0, False)
        transform = lambda x: torch.tensor(x.copy(),requires_grad=False,device=torch.device('cuda'),dtype=torch.float)
        fs = [self._chop(f_0, self.lmaxs[0],transform)]
        #fs = [self._preprocessing(f_0[:,:,0:(self.lmaxs[0]+1)**2].copy())]
        batch_size = f_0.shape[0]
        u_cnt = 0
        for step in range(len(self.nlayers)):
            cur_lmax = self.lmaxs[step]
            for layer_i in range(self.nlayers[step]):
                if layer_i == 0 and step > 0:
                    f_pre = self._chop(f_0, cur_lmax,transform)
                    for l in range(self.lmaxs[step-1]+1):
                        f_pre[l] = torch.cat([f_pre[l], fs[-1][l]], dim=1)
                    #coefs_pre = self._preprocessing(f_0[:,:,0:(cur_lmax+1)**2].copy())
                    #f_pre = []
                    #for l in range(cur_lmax+1):
                    #    temp = coefs_pre[:, self.num_channels_input*l**2:self.num_channels_input*(l+1)**2,:]
                    #    if l <= self.lmaxs[step-1]:
                    #        to_cat = fs[-1][:,self.cum_taus[l]:self.cum_taus[l+1],:]
                    #        temp = torch.cat([temp, to_cat],dim=1)
                    #    f_pre.append(temp)
                    #f_pre = torch.cat(f_pre, dim=1)
                else:
                    f_pre = fs[-1]
                fs.append(self.us[u_cnt](f_pre))
                u_cnt += 1

        if output_type == 0:
            if self.skip_connections_to_output:
                print(len(fs), )
                assert(len(fs) == 1+ len(self.nlayers) * self.nlayers[0])
                embedding = torch.cat([fs[i][0].view(batch_size,-1) for i in range(len(fs))], 1)
            else:
                embedding = fs[-1][0].view(batch_size,-1)
            return embedding

    def forward(self, f_0):
        return self.forward_test_angle(f_0, output_type=0)