import sys
import os
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
REAL_PART=0
IMAG_PART=1
import torch.nn as nn
import torch
import numpy as np

from cudaUpBatchNorm.fUpBatchNorm import fUpModule as UpdateModule

class SphericalCNN_fast(nn.Module):
    def __init__(self, lmax, 
                 taus,
                 n_layers=2,
                 cudaFlag=True,
                 normalization=1,
                 skip_type=0,
                 num_channels_input=1):
        assert(cudaFlag)
        assert(normalization==1)
        #the maximum l is lmax
        super(SphericalCNN_fast, self).__init__()
        print("Creating network")
        self.lmax=lmax
        self.cudaFlag=cudaFlag
        # Define Update
        #torch.manual_seed(1)
        self.taus = taus

        self.us = nn.ModuleList([])
        for layer_i in range(n_layers):
            u = UpdateModule(lmax, 
                taus=[num_channels_input]*(lmax+1) if layer_i==0 else taus, 
                out_taus=taus,
                batchnorm=normalization==1,
                layername="layer{}".format(layer_i))
            self.us.append(u)

        self.normalize_input = False
        self.num_channels_input = num_channels_input

        self.skip_connections_to_output = skip_type == 1
        if self.skip_connections_to_output:
            #print(taus[0], n_layers)
            self.output_length = 2 * (num_channels_input+taus[0]*(n_layers))
            #print(self.output_length)
        else:
            self.output_length = 2 * taus[0]
        
        self.n_layers = n_layers
        if cudaFlag:
            self.cuda()
        print("..done..")
    
    def _preprocessing(self, f_0):
        #chopping and making single img into a batch of size 1
        #print("original input", [k.shape for k in f_0])
        assert(isinstance(f_0, np.ndarray))
        d = len(f_0.shape)
        if f_0.dtype == np.complex128:
            f_0 = np.stack([f_0.real, f_0.imag], axis=d)
        #already separated
        if len(f_0.shape) == 3:
            f_0 = np.expand_dims(f_0, 0)
        batch_size=f_0.shape[0]
        f_0_new = np.concatenate([f_0[:,:,l**2:(l+1)**2,:].copy().reshape(batch_size,-1,2) for l in range(self.lmax+1)], axis=1)
        return torch.tensor(f_0_new, requires_grad=False,device=torch.device('cuda'),dtype=torch.float) 

    def forward_test_angle(self, f_0, output_type=0):
        f = self._preprocessing(f_0)
        fs = [f]
        batch_size = f.shape[0]
        for i in range(self.n_layers):
            fs.append(self.us[i](fs[-1], straight_output=True))
        if output_type == 0:
            if self.skip_connections_to_output:
                embedding = torch.cat([fs[0][:,0:self.num_channels_input,:].view(batch_size,-1)]+\
                                        [fi[:,0:self.taus[0],:].view(batch_size,-1) for fi in fs[1:]],1)
            else:
                embedding = fs[-1][:,0:self.taus[0],:].view(batch_size,-1)
            return embedding
        else:
            f = f if output_type==1 else fs[-1]
            return [1j*f[l][:,:,:,1].numpy() + f[l][:,:,:,0].numpy() for l in range(self.lmax+1)]

    def forward(self, f_0):
        return self.forward_test_angle(f_0, output_type=0)
        
class SphericalResCNN_fast(nn.Module):
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
        super(SphericalResCNN_fast, self).__init__()
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
                u = UpdateModule(cur_lmax, taus = tau_in, out_taus=tau_out,
                                    batchnorm=normalization==1,
                                    layername="layer{}_{}".format(step,layer))
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

        batch_size=f_0.shape[0]
        f_0_new = np.concatenate([f_0[:,:,l**2:(l+1)**2,:].copy().reshape(batch_size,-1,2) for l in range(self.lmax+1)], axis=1)
        if not to_torch:
            return f_0_new
        return torch.tensor(f_0_new, requires_grad=False,device=torch.device('cuda'),dtype=torch.float) 

    def _chop(self, f, lmax, transform=lambda x: x):
        if self.normalize_input:
            for channel in range(self.num_channels_input):
                std = np.linalg.norm(f[:,channel,:,:],ord=2,axis=-1).std()
                f[:,channel,:,:] = f[:,channel,:,:]/ max(1e-4,std)
        return [transform(f[:,:, l**2:(l+1)**2,:]) for l in range(lmax+1)]

    def forward_test_angle(self, f_0, output_type=0):
        fs = [self._preprocessing(f_0[:,:,0:(self.lmaxs[0]+1)**2].copy())]
        batch_size = f_0.shape[0]
        u_cnt = 0
        for step in range(len(self.nlayers)):
            cur_lmax = self.lmaxs[step]
            for layer_i in range(self.nlayers[step]):
                if layer_i == 0 and step > 0:
                    coefs_pre = self._preprocessing(f_0[:,:,0:(cur_lmax+1)**2].copy())
                    f_pre = []
                    for l in range(cur_lmax+1):
                        temp = coefs_pre[:, self.num_channels_input*l**2:self.num_channels_input*(l+1)**2,:]
                        if l <= self.lmaxs[step-1]:
                            to_cat = fs[-1][:,self.cum_taus[l]:self.cum_taus[l+1],:]
                            temp = torch.cat([temp, to_cat],dim=1)
                        f_pre.append(temp)
                    f_pre = torch.cat(f_pre, dim=1)
                else:
                    f_pre = fs[-1]
                fs.append(self.us[u_cnt](f_pre,straight_output=True))
                u_cnt += 1

        if output_type == 0:
            if self.skip_connections_to_output:
                embedding = torch.cat([fs[0][:,0:self.num_channels_input,:].view(batch_size,-1)]+\
                                        [fi[:,0:self.taus[0],:].view(batch_size,-1) for fi in fs[1:]],1)
            else:
                embedding = fs[-1][:,0:self.taus[0],:].view(batch_size,-1)
            return embedding

    def forward(self, f_0):
        return self.forward_test_angle(f_0, output_type=0)
        
      
class SphericalResCNN_fast2(nn.Module):
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
        #the maximum l is lmax
        super(SphericalResCNN_fast2, self).__init__()
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
                            tau_in[l] += self.taus[l] * (1 if l > 0 else self.nlayers[step-1])
                else:
                    tau_in = [self.taus[l] for l in range(cur_lmax + 1)]
                #print(tau_in, tau_out)
                u = UpdateModule(cur_lmax, taus = tau_in, out_taus=tau_out,
                                    batchnorm=normalization==1,
                                    layername="layer{}_{}".format(step,layer))
                self.us.append(u)


        self.normalize_input = True
        self.num_channels_input = num_channels_input

        self.output_length = 2 * (self.nlayers[-1] * taus[0])
        
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
        batch_size=f_0.shape[0]
        f_0_new = np.concatenate([f_0[:,:,l**2:(l+1)**2,:].copy().reshape(batch_size,-1,2) for l in range(self.lmax+1)], axis=1)
        if not to_torch:
            return f_0_new
        return torch.tensor(f_0_new, requires_grad=False,device=torch.device('cuda'),dtype=torch.float) 

    def _chop(self, f, lmax, transform=lambda x: x):
        if self.normalize_input:
            for channel in range(self.num_channels_input):
                std = np.linalg.norm(f[:,channel,:,:],ord=2,axis=-1).std()
                f[:,channel,:,:] = f[:,channel,:,:]/ max(1e-4,std)
        return [transform(f[:,:, l**2:(l+1)**2,:]) for l in range(lmax+1)]

    def forward_test_angle(self, f_0, output_type=0):
        fs = [self._preprocessing(f_0[:,:,0:(self.lmaxs[0]+1)**2].copy())]
        batch_size = f_0.shape[0]
        u_cnt = 0
        for step in range(len(self.nlayers)):
            cur_lmax = self.lmaxs[step]
            for layer_i in range(self.nlayers[step]):
                if layer_i == 0 and step > 0:
                    coefs_pre = self._preprocessing(f_0[:,:,0:(cur_lmax+1)**2].copy())
                    f_pre = []
                    for l in range(cur_lmax+1):
                        temp = coefs_pre[:, self.num_channels_input*l**2:self.num_channels_input*(l+1)**2,:]
                        if l == 0:
                            to_cat = [fs[-prev_layer][:,0:self.cum_taus[1],:] for prev_layer in range(self.nlayers[step-1])]
                            temp = torch.cat([temp] + to_cat,dim=1)
                        elif l <= self.lmaxs[step-1]:
                            to_cat = fs[-1][:,self.cum_taus[l]:self.cum_taus[l+1],:]
                            temp = torch.cat([temp, to_cat],dim=1)

                        f_pre.append(temp)
                    f_pre = torch.cat(f_pre, dim=1)
                else:
                    f_pre = fs[-1]
                fs.append(self.us[u_cnt](f_pre,straight_output=True))
                u_cnt += 1

        if output_type == 0:
            embedding = torch.cat([fs[-layer_i][:,0:self.taus[0],:].view(batch_size,-1) for layer_i in range(self.nlayers[-1])],1)
            return embedding

    def forward(self, f_0):
        return self.forward_test_angle(f_0, output_type=0)

