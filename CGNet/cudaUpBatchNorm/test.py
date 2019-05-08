import os
import sys
CUR_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(CUR_DIR, ".."))

import cudaUpBatchNorm.fUpBatchNorm as fUp
import cudaCG.cuda.UpdateFunc as fUpPython

import torch.nn.functional as F
import numpy as np
from importlib import reload
reload(fUpPython)
reload(fUp)
import torch
maxL, BATCH_SIZE, taus = 2, 2, [2,1,3]

def get_grad(v,out):
    if out[0].is_cuda:
        target = torch.zeros([1], device=torch.device('cuda'))
    else:
        target = torch.zeros([1], device=torch.device('cpu'))
    tot = 0
    for lst in out:
        tot += torch.norm(lst)
    loss = F.mse_loss(tot, target)
    loss.backward()
    return [i.grad for i in v]

def _l2b(l):
    return torch.cat([vi.view(BATCH_SIZE, -1, 2) for vi in l], dim=1)

def routine1(v,m, merge_list=False, getgradflag=True):
    m.eval()
    m.train()
    batch_size = v[0].shape[0]
    if merge_list:
        new_v = torch.cat([vi.view(batch_size, -1, 2) for vi in v], dim=1)
        assert(new_v.is_cuda)
    else:
        new_v = v
    vo = m(new_v)
    grads = None
    if getgradflag:
        grads = get_grad(v,vo)
    #print(m.weights.grad[0:5,:])
    return vo, grads


def wrap_listof_np(npv, dtype=torch.float, requires_grad=True):
    n = len(npv)
    return [torch.tensor(npv[i], dtype=dtype, device=torch.device('cuda'),requires_grad=requires_grad) for i in range(n)]

if __name__ == "__main__":

    oCuda = fUp.fUpModule(maxL, taus, taus)
    oPython = fUpPython.UpdateFunction(maxL, taus, taus, cudaFlag=True)

    np.random.seed(0)
    npv = [np.random.rand(BATCH_SIZE, taus[l], 2*l+1,2) for l in range(maxL+1)]
    vCuda, vPython = wrap_listof_np(npv), wrap_listof_np(npv)

    #print("weights: \n\n", oCuda.weights,oPython.ws[0], oPython.ws[1])
    #print("bmscale: \n\n", oCuda.moving_std, oPython.bmlayer_scale[0], oPython.bmlayer_scale[1])

    voCuda, gradCuda = routine1(vCuda, oCuda, merge_list=True)
    #print("\n\nCUDA###=======================:\n", voCuda, gradCuda)

    voPython, gradPython = routine1(vPython, oPython)
    #print("\n\nPython###=======================:\n",voPython, gradPython)


    #print("\nweights: \n\n", oCuda.weights, "\n", oPython.ws[0],oPython.ws[1])
    #print("\nbmscale: \n\n", oCuda.moving_std, "\n", oPython.bmlayer_scale[0], oPython.bmlayer_scale[1])
    if maxL < 10:
        for i in range(len(voPython)):
            vdiff = voCuda[i].cpu().detach().numpy() - voPython[i].detach().cpu().numpy()
            print("{}: l={} max_vdiff={} < 1e-4".format(np.max(vdiff) < 1e-4, i, np.max(vdiff)))

            gdiff = gradCuda[i].cpu().detach().numpy() - gradPython[i].detach().cpu().numpy()
            print("{}: grad_in l={} max_gdiff={} < 1e-4".format(np.max(gdiff) < 1e-4, i, np.max(gdiff)))