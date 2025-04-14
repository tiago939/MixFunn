import torch
import torch.nn as nn
import mixfunn as mf


class MixFunn(nn.Module):
    
    def __init__(self):
        super(MixFunn, self).__init__()

        self.layer = nn.Sequential(
            mf.Mixfun(2, 1, second_order_input=True, normalization_function=True)
        )

    def forward(self, x):

        x = self.layer(x)

        return x