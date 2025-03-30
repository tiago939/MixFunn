import torch
import torch.nn as nn
import mixfunn as mf

class MixFunn(nn.Module):
    
    def __init__(self):
        super(MixFunn, self).__init__()

        self.layers = nn.Sequential(
            mf.Quad(2, 4),
            mf.MixFun(4, 4, second_order_function=True),
            mf.Quad(4, 1)
        )

    def forward(self, x):
        x = self.layers(x)

        return x