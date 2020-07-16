import torch
import torch.nn as nn
from itertools import combinations


def impose(x, m1=1., m2=0., m3=0.):
    # TODO implement m1
    return x * torch.tensor(m2).cos() - (1 - x ** 2).sqrt() * torch.tensor(m2).sin() - m3


class Arcface(nn.Module):
    def __init__(self):
        super(Arcface, self).__init__()

    def forward(self, x, target):
        s = 5
        ret_x = []
        ret_target = []
        # x:(B,S,W,h)
        B, S, W, h = x.shape
        # label:(B,S,W)
        x = x / ((x ** 2).sum(-1).sqrt().view(B, S, W, 1))
        for b in range(B):
            for comb in combinations(range(S), 2):
                x0, x1 = x[b, comb[0]], x[b, comb[1]]
                t0, t1 = target[b, comb[0]], target[b, comb[1]]
                ret_target += [(t0.view(-1, 1) == t1.view(1, -1)).int().argmax(-1)]
                _x = (x0 @ x1.T) * s
                mask = (t0.view(-1, 1) == t1.view(1, -1))
                # _x[mask]=impose(_x[mask])
                ret_x += [_x]
        return torch.cat(ret_x, dim=0), torch.cat(ret_target, dim=0)
