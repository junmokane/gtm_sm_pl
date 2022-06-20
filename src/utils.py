import torch
from torch import nn

class Preprocess_img(nn.Module):
    def forward(self, x):
        return x * 2 -1


class Deprocess_img(nn.Module):
    def forward(self, x):
        return (x + 1) / 2


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.contiguous().view(N, -1)


class Exponent(nn.Module):
    def forward(self, x):
        return torch.exp(x)

class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=3, H=8, W=8):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)