import torch
import numpy as np
import torch.nn.functional as F
import fCG
reload(fCG)

def get_grad(v,w):
	if w[0].is_cuda:
		target = torch.zeros([1]).cuda()
	else:
		target = torch.zeros([1])
	tot = 0
	for lst in w:
	    tot += torch.norm(lst)
	loss = F.mse_loss(tot, target)
	loss.backward()
	return [i.grad for i in v]

maxL = 1
BATCH_SIZE=1
taus = [2,1]
m = fCG.fCGModule(maxL, taus)
npv = [np.random.rand(BATCH_SIZE, taus[l], 2*l+1,2) for l in range(maxL+1)]

device = torch.device('cuda')
#v1 = [torch.tensor(npv[i],dtype=torch.float,requires_grad=True).cuda() for i in range(maxL+1)]
v1 = [torch.tensor(npv[i],dtype=torch.float,device=device,requires_grad=True) for i in range(maxL+1)]
#print(v1)
w1 = m(v1)
print(w1[0])
print(get_grad(v1,w1))
