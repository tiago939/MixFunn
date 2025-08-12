import torch
import torch.nn as nn
import mixfun as mf

class MixFunn(nn.Module):
    
    def __init__(self, normalization_function, normalization_neuron, p_drop, second_order_input, second_order_function):
        super(MixFunn, self).__init__()

        self.layers = nn.Sequential(
            mf.Mixfun(1, 1, normalization_function=normalization_function, normalization_neuron=normalization_neuron, p_drop=p_drop, second_order_input=second_order_input, second_order_function=second_order_function),
        )

    def forward(self, x):
        x = self.layers(x)

        return x