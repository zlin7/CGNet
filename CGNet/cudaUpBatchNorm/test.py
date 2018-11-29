
import os
import sys
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, ".."))

import numpy as np
import random
import torch
from torch.nn.parameter import Parameter
import torch.nn as nn
import sys
import os

import torch
import numpy as np
import time
import torch.nn.functional as F
from cudaUpBatchNormreference import fUpBatchNormRef 
reload(fUpBatchNormRef)
import fUpBatchNorm
reload(fUpBatchNorm)


GET_GRADIENT=True
BATCH_NORM=True
ONLYONE=False
#ONLYONE=True
maxL, BATCH_SIZE, taus = 1, 1, [2,1]
#maxL, BATCH_SIZE, taus = 2, 4, [3,2,1]
#maxL, BATCH_SIZE, taus = 30, 50, [2 for i in range(30+1)]
#maxL, BATCH_SIZE, taus = 20, 50, [2 for i in range(20+1)]
#maxL, BATCH_SIZE, taus = 8, 50, [2 for i in range(8+1)]

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
    m.eval()
    m.train()
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

def wrap_listof_np(npv, dtype=torch.float, requires_grad=True):
    n = len(npv)
    return [torch.tensor(npv[i], dtype=dtype, device=torch.device('cuda'),requires_grad=requires_grad) for i in range(n)]



npv = [np.random.rand(BATCH_SIZE, taus[l], 2*l+1,2) for l in range(maxL+1)]
v1, v2 = wrap_listof_np(npv), wrap_listof_np(npv)

torch.manual_seed(0)
m2 = fUpBatchNorm.fUpModule(maxL, taus, out_taus=taus, batchnorm=BATCH_NORM)
#Testing m2
vo2, grad2 = routine1(v2, m2)
print(vo2,grad2)
if not ONLYONE:
    torch.manual_seed(0)
    m1 = fUpBatchNormRef.fUpModule(maxL, taus, out_taus=taus, batchnorm=BATCH_NORM)
    vo1, grad1 = routine1(v1, m1)

    if maxL<10:
        for i in range(len(vo2)):
            vdiff = vo2[i].cpu().detach().numpy() - vo1[i].detach().cpu().numpy() 
            print("{}: l={} max_vdiff={} < 1e-4".format(np.max(vdiff) < 1e-4, i, np.max(vdiff)))
            if GET_GRADIENT:
                gdiff = grad2[1][i].cpu().detach().numpy() - grad1[1][i].detach().cpu().numpy() 
                print("{}: grad_in l={} max_gdiff={} < 1e-4".format(np.max(gdiff) < 1e-4, i, np.max(gdiff)))

        if GET_GRADIENT:
            gdiff = grad2[0].cpu().detach().numpy() - grad1[0].detach().cpu().numpy() 
            print("{}: grad_w l={} max_gdiff={} < 1e-4".format(np.max(gdiff) < 1e-4, i, np.max(gdiff)))