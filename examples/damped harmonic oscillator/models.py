import torch
import torch.nn as nn
import mixfunn as mf

class MixFunn(nn.Module):
    
    def __init__(self):
        super(MixFunn, self).__init__()

        self.layers = nn.Sequential(
            mf.Mixfun(1, 1, second_order_function=True),
        )

    def forward(self, x):
        x = self.layers(x)

        return x