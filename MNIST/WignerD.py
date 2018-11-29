import numpy as np
from math import factorial, sqrt, cos, sin

fact = lambda x: factorial(int(x))
def choose(n, k):
    return fact(n)/fact(k)/fact(n-k)
def dmat_entry(j,m_,m,beta):
    #real valued. implemented according to wikipedia
    partA = sqrt(fact(j+m_)*fact(j-m_)*fact(j+m)*fact(j-m))
    partB = 0.
    for s in range(max(int(m-m_),0),min(int(j+m),int(j-m_))+1):
        temp = (-1.)**s / (fact(j+m-s)*fact(s)*fact(m_-m+s)*fact(j-m_-s))
        partB += temp * cos(beta/2)**(2*j+m-m_-2*s) * (sin(beta/2))**(m_-m+2*s)
    return partA * partB
def dm(theta,l):
    ret=np.zeros((2*l+1,2*l+1))
    for m in range(-l,l+1):
        for n in range(-l,l+1):
            ret[m+l,n+l]=dmat_entry(l,m,n,theta)
    return ret
def Dmat_entry(l,m,n,alpha,beta,gamma):
    return np.exp(-1j*m*alpha) * dmat_entry(l,m,n,beta) * np.exp(-1j*n*gamma)
def Dm(angles, l=1):
    ret = np.zeros((2*l+1,2*l+1), dtype=np.complex)
    for m in range(-l,l+1):
        for n in range(-l,l+1):
            ret[m+l,n+l] = Dmat_entry(l,m,n,angles[0],angles[1],angles[2])
            #print(ret[m+l,n+l])
    return ret


def _Dm_hardcode(angles, l=1):
    alpha, beta, gamma = angles
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    exp = np.exp
    i = 1j
    if l == 0:
        return np.ones((1,1),dtype=np.complex)
    
    if l == 1:
        D = np.zeros((3,3),dtype=np.complex)
        D[2,2] = (1+cos(beta))/2*exp(-i*(alpha+gamma))
        D[2,1] = -1/sqrt(2)*sin(beta)*exp(-i*alpha)
        D[2,0] = (1-cos(beta))/2*exp(-i*(alpha-gamma))
        
        D[1,2] = 1/sqrt(2)*sin(beta)*exp(-i*gamma)
        D[1,1] = cos(beta)
        D[1,0] = -1/sqrt(2)*sin(beta)*exp(i*gamma)
        
        D[0,2] = (1-cos(beta))/2*exp(i*(alpha-gamma))
        D[0,1] = 1/sqrt(2)*sin(beta)*exp(i*alpha)
        D[0,0] = (1+cos(beta))/2*exp(i*(alpha+gamma))
        return D
    if l == 2:
        ei = lambda x: exp(1j * x)
        D = np.zeros((5,5),dtype=np.complex)
        D[4,4] = ((1+cos(beta))/2)**2*exp(-2*i*(alpha+gamma))
        D[4,3] = -(1+cos(beta))/2*sin(beta)*exp(-i*(2*alpha+gamma))
        D[4,2] = sqrt(3./8)*sin(beta)**2*exp(-i*2*alpha)
        D[4,1] = -(1-cos(beta))/2*sin(beta)*exp(i*(-2*alpha+gamma))
        D[4,0] = ((1-cos(beta))/2)**2*exp(2*i*(-alpha+gamma))
        
        D[3,4] = (1+cos(beta))/2*sin(beta)*exp(-i*(alpha+2*gamma))
        D[3,3] = (cos(beta)**2-(1-cos(beta))/2)*exp(-i*(alpha+gamma))
        D[3,2] = -sqrt(3./8)*sin(2*beta)*exp(-i*alpha)
        D[3,1] = ((1+cos(beta))/2-cos(beta)**2)*exp(i*(-alpha+gamma))
        D[3,0] = -((1-cos(beta))/2)*sin(beta)*exp(i*(-alpha+2*gamma))
    
        D[2,4] = sqrt(3./8)*sin(beta)**2*exp(-i*2*gamma)
        D[2,3] = sqrt(3./8)*sin(2*beta)*exp(-i*gamma)
        D[2,2] = (3*cos(beta)**2-1.)/2
        D[2,1] = -sqrt(3./8)*sin(2*beta)*exp(i*gamma)
        D[2,0] = sqrt(3./8)*sin(beta)**2*exp(i*2*gamma)
        
        D[1,4] = (1-cos(beta))/2*sin(beta)*exp(i*(alpha-2*gamma))
        D[1,3] = ((1+cos(beta))/2-cos(beta)**2)*exp(i*(alpha-gamma))
        D[1,2] = sqrt(3./8)*sin(beta)**2*exp(i*alpha)
        D[1,1] = (cos(beta)**2-(1-cos(beta))/2)*exp(i*(alpha+gamma))
        D[1,0] = -(1+cos(beta))/2*sin(beta)*exp(i*(alpha+2*gamma))
        
        D[0,4] = ((1-cos(beta))/2)**2*exp(2*i*(alpha-gamma))
        D[0,3] = ((1-cos(beta))/2)*sin(beta)*exp(i*(2*alpha-gamma))
        D[0,2] = sqrt(3./8)*sin(beta)**2*exp(i*2*alpha)
        D[0,1] = (1+cos(beta))/2*sin(beta)*exp(i*(2*alpha+gamma))
        D[0,0] = ((1+cos(beta))/2)**2*exp(2*i*(alpha+gamma))
        
        return D