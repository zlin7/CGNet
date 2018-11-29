import torch
import numpy as np
import time
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
	#print([j.grad for j in w])
	return [i.grad for i in v]

maxL = 2
BATCH_SIZE=2
taus = [3,2,1]

#maxL = 30
#BATCH_SIZE=20
#taus = [2 for i in range(maxL+1)]

m = fCG.fCGModule(maxL, taus)
npv = [np.random.rand(BATCH_SIZE, taus[l], 2*l+1,2) for l in range(maxL+1)]

device = torch.device('cuda')
#v1 = [torch.tensor(npv[i],dtype=torch.float,requires_grad=True).cuda() for i in range(maxL+1)]
v1 = [torch.tensor(npv[i],dtype=torch.float,device=device,requires_grad=True) for i in range(maxL+1)]
#print(v1)
starttime=time.time()
w1 = m(v1)
if maxL < 5:
	print(w1)
g1 = get_grad(v1,w1)
print(time.time() - starttime)
#print(g1)



print()
print()
print()


import sys
import os
sys.path.append("/home/zhen/Desktop/gitRes/sphericalcnn/sphereProj/abandon/FastCG/v1/python/")
import CGfunctions as cg

v2 = [torch.tensor(npv[i],dtype=torch.float,requires_grad=True) for i in range(maxL+1)]
starttime=time.time()
w2=cg.BatchCGproduct(BATCH_SIZE,[taus,taus],maxl=maxL)(*(v2+v2))
g2 = get_grad(v2,w2)
if maxL < 5:
	print(w2)
print(time.time() - starttime)



print("Comparing gradients", g1)
print("Comparing gradients", g2)

#print(get_grad(v1,w1))
#print(get_grad(v2,w2))