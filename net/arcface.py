import torch
import torch.nn as nn
from itertools import combinations
class Arcface(nn.Module):
    def __init__(self):
        super(Arcface, self).__init__()
    def forward(self,x,target):
        s=5
        ret_x=[]
        ret_target=[]
        #x:(B,S,W,h)
        B,S,W,h=x.shape
        #label:(B,S,W)
        x=x/x.norm()
        for b in range(B):
            for comb in combinations(range(S),2):
                x0,x1=x[b,comb[0]],x[b,comb[1]]
                t0,t1=target[b,comb[0]],target[b,comb[1]]
                ret_x+=[(x0@x1.T)*s]
                ret_target+=[(t0.view(-1,1)==t1.view(1,-1)).int().argmax(-1)]
        return torch.cat(ret_x,dim=0),torch.cat(ret_target,dim=0)