
import os
import sys
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, ".."))

import numpy as np
import random
#from lie_learn.representations.SO3 import wigner_d as wd
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import Complex_math as cm
import sys
import os
import ClebschGordan
REAL_PART,IMAG_PART=0,1
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
#Create the CG dicionary of matices, and making them Variables for pytorch
class ClebschGordanDict():
    def __init__(self,lmax,cudaFlag=False):
        print("Creating CGDictionary...L is {}".format(lmax))
        D = ClebschGordan.precompute(lmax)
        self.lmax = lmax
        self.Dict = {}
        for k in D.keys():
            v = torch.tensor(D[k], requires_grad=False).float()
            #self.Dict[k] = v.cuda() if cudaFlag else v
            self.Dict[k] = v
        del D
    
    def getCGmat(self, l1,l2,l):
        idx = l + (self.lmax+1) * (l2 + (self.lmax+1)*l1)
        return self.Dict.get(idx)
        
    def getEntry(self, l1, l2, l, m1, m2, m):
        CGmat = self.getCGmat(l1,l2,l)
        if CGmat is None:
            return None
        return CGmat[m+l, (m1+l1)*(2*l2+1)+(m2+l2)]

class UpdateFunction_old(nn.Module):#in the first use 10 filters (gs), after the first layer use F with F itself
    def __init__(self, lmax, 
                 tau_pre,
                 precomputed_CG=None,
                 cudaFlag=False):
        #NOOOOOOOOOOOOOOOBecause of how the cg module is cocded up, let's take the input as 
        #NOOOOOOOOOOOOOlist of list, where the first list is indexed by the image, and the second by l
        
        #Take it as a list of batch tensor (4D)
        super(UpdateFunction_old, self).__init__()
        self.lmax=lmax
        self.cudaFlag=cudaFlag

        self.CGDict = ClebschGordanDict(lmax, cudaFlag=cudaFlag) if precomputed_CG is None else precomputed_CG

        if cudaFlag:
            self.cuda()

    def cg(self, t, l1, l2, l):
        t_real, t_imag = t
        #t_real, t_imag = t[:,:,:,0], t[:,:,:,1]
        CG_mat_pre = self.CGDict.getCGmat(l1,l2,l)
        batch_CGmat_real = CG_mat_pre.repeat(t_real.shape[0],1,1)
        if self.cudaFlag:
            batch_CGmat_real = batch_CGmat_real.cuda()
        return torch.stack(list(cm.C_bmm(batch_CGmat_real, None, t_real, t_imag)), 3) 
            

    def forward(self, fs):
        batch_size = fs[0].shape[0]
        new_fs = [[] for i in range(self.lmax+1)]
        fs = [f.permute(0,2,1,3) for f in fs]
        for l1 in range(self.lmax+1):
            for l2 in range(l1 + 1):
                for l in range(abs(l1-l2), min(l1+l2, self.lmax)+1):
                    kp = cm.C_kron_prod(fs[l1][:,:,:,0], fs[l1][:,:,:,1],
                                        fs[l2][:,:,:,0], fs[l2][:,:,:,1])
                    new_fs[l].append(self.cg(kp, l1, l2, l).permute(0,2,1,3))

        for l in range(self.lmax+1):
            new_fs[l] = torch.cat(new_fs[l], 1)
        return new_fs


import torch
import numpy as np
import time
import torch.nn.functional as F
import fUpBatchNorm
reload(fUpBatchNorm)


GET_GRADIENT=True
BATCH_NORM=True
maxL, BATCH_SIZE, taus = 1, 1, [2,1]
#maxL, BATCH_SIZE, taus = 2, 4, [3,2,1]
#maxL, BATCH_SIZE, taus = 30, 50, [2 for i in range(30+1)]
#maxL, BATCH_SIZE, taus = 20, 50, [2 for i in range(20+1)]

def get_grad(v,out):
    if out[0].is_cuda:
        target = torch.zeros([1]).cuda()
    else:
        target = torch.zeros([1])
    tot = 0
    for lst in out:
        tot += torch.norm(lst)
    loss = F.mse_loss(tot, target)
    loss.backward()
    return [i.grad for i in v]

def routine1(v,m,verbose=maxL<5,getgradflag=GET_GRADIENT):
    #m.eval()
    #m.train()
    batch_size = v[0].shape[0]
    new_v = torch.cat([torch.tensor(vi.view(batch_size,-1, 2),requires_grad=True) for vi in v], dim=1).float()
    assert(new_v.is_cuda)
    starttime = time.time()
    vo = m(new_v)
    forward_time1 = time.time() - starttime
    grads = None
    backward_time1 = None
    if getgradflag:
        if BATCH_NORM:
            #Mess up the moving std to see if really using a clone
            m.moving_std.data.normal_(0,1.0)
        grads = get_grad(v,vo)
        #print("diu le??",v[0].grad)
        backward_time1 = time.time() - starttime - forward_time1
        #print("weight grads?", m.weights.grad)
    print("Forward time {}, Backward time {}".format(forward_time1, backward_time1))
    #print(m.weights.grad[0:5,:])
    return vo, (m.weights.grad, grads)

def routine(v, m, ws = None, bm_scale=None, verbose=maxL < 5, getgradflag=GET_GRADIENT):
    global maxL, BATCH_SIZE
    #training = False
    #training = True
    bmlayer_cnt = 1
    starttime = time.time()
    vo = m(v)
    #print("bm_pre", bm_scale)
    #print("old_pre",vo)
    if ws is not None:
        print("Applying weights")
        for l in range(maxL+1):
            l_components = vo[l]
            if bm_scale is not None:
                if m.training:
                    #print("pre old", bm_scale)
                    npv = l_components.cpu().detach().numpy().copy()
                    norm = np.linalg.norm(npv, ord=2, axis=3)
                    std = torch.tensor(np.std(norm, axis=(0,2))).cuda()
                    #print(std)
                    bm_scale[l] *= bmlayer_cnt/(float(bmlayer_cnt)+1)
                    #print("mid", bm_scale)
                    bm_scale[l][:,0,0] += std / (bmlayer_cnt+1)
                    #print("after old", bm_scale)
                l_components = l_components / torch.max(1e-5 * torch.ones(1, requires_grad=False).cuda(), bm_scale[l])
            #print("pre_{}".format(l),l_components)
            vo[l] = Complex_bmm(ws[l].repeat(BATCH_SIZE,1,1,1), l_components)
        #print("old_out", vo)
    #print("bm_post", bm_scale)
    forward_time1 = time.time() - starttime
    w_grads = None
    grads = None
    backward_time1 = None
    if getgradflag:
        grads = get_grad(v,vo)
        backward_time1 = time.time() - starttime - forward_time1
    if verbose:
        #print(vo)
        #print("grad", grads)
        if ws is not None and getgradflag:
            w_grads = torch.cat([torch.tensor(w.grad.view(-1, 2),requires_grad=True) for w in ws], dim=0)
    print("Forward time {}, Backward time {}".format(forward_time1, backward_time1))
    #print(grads)
    return vo, grads if ws is None else (w_grads, grads)

def wrap_listof_np(npv, dtype=torch.float, requires_grad=True):
    n = len(npv)
    return [torch.tensor(npv[i], dtype=dtype, device=torch.device('cuda'),requires_grad=requires_grad) for i in range(n)]


torch.manual_seed(1)
m1 = fUpBatchNorm.fUpModule(maxL, taus, out_taus=taus, batchnorm=BATCH_NORM)

if maxL < 10:
    m2 = UpdateFunction_old(maxL, taus, cudaFlag=True)

npv = [np.random.rand(BATCH_SIZE, taus[l], 2*l+1,2) for l in range(maxL+1)]
v1, v2 = wrap_listof_np(npv), wrap_listof_np(npv)

middle_taus = np.zeros(maxL+1, dtype=int)
for l1 in range(maxL+1):
    for l2 in range(l1+1):
        for l in range(l1-l2, min(maxL,l1+l2)+1):
            middle_taus[l] += taus[l1] * taus[l2]


ws1=None
torch.manual_seed(1)
wlength = np.sum(np.asarray([taus[l]*middle_taus[l] for l in range(maxL+1)]))
wbig = 0.05 * torch.rand(wlength,2,device=torch.device('cuda'),dtype=torch.float)
ws2 = []
offset = 0
for l in range(maxL+1):
    ws2.append(nn.Parameter(wbig[offset:(offset+taus[l]*middle_taus[l]),:].view(taus[l],middle_taus[l],2), requires_grad=True))
    offset +=taus[l]*middle_taus[l]
#print(ws2)
np.random.seed(0)
bm_np_pre = np.random.rand(np.sum(middle_taus))
#print(bm_np_pre)
bm_np = []
st = 0
for l in range(maxL+1):
    ed = st + middle_taus[l]
    bm_np.append(np.expand_dims(np.expand_dims(bm_np_pre[st:ed],1),2))
    st = ed
bm_scale1, bm_scale2 = None, wrap_listof_np(bm_np, requires_grad=False) if BATCH_NORM else None

m1.train()
m2.train()

vo1, grad1 = routine1(v1, m1)


if maxL < 10:
    vo2, grad2 = routine(v2, m2, ws2, bm_scale2)

    for i in range(len(vo2)):
        vdiff = vo2[i].cpu().detach().numpy() - vo1[i].detach().cpu().numpy() 
        print("{}: l={} max_vdiff={} < 1e-4".format(np.max(vdiff) < 1e-4, i, np.max(vdiff)))
        if GET_GRADIENT:
            gdiff = grad2[1][i].cpu().detach().numpy() - grad1[1][i].detach().cpu().numpy() 
            print("{}: grad_in l={} max_gdiff={} < 1e-4".format(np.max(gdiff) < 1e-4, i, np.max(gdiff)))

    if GET_GRADIENT:
            gdiff = grad2[0].cpu().detach().numpy() - grad1[0].detach().cpu().numpy() 
            print("{}: grad_w l={} max_gdiff={} < 1e-4".format(np.max(gdiff) < 1e-4, i, np.max(gdiff)))