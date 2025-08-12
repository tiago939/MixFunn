import torch
import torch.nn as nn
import mixfun as mf

class PINN(nn.Module):
    
    def __init__(self):
        super(PINN, self).__init__()
        
        L = 1
        N = 16

        layers = []
        layers.append(nn.Linear(3, N))
        layers.append(nn.Tanh())
        for _ in range(L):
            layers.append(nn.Linear(N, N))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(N, 1))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)

        return x

class MixFunn(nn.Module):
    
    def __init__(self):
        super(MixFunn, self).__init__()

        layers = []

        self.layers = nn.Sequential(
            mf.Mixfun(3, 1, normalization_function=False, normalization_neuron=False, p_drop=False, second_order_input=True, second_order_function=False),
            #mf.Mixfun(8, 1, normalization_function=False, normalization_neuron=False, p_drop=False, second_order_input=False, second_order_function=False),
            )

    def forward(self, x):

        x = self.layers(x)

        return x


class Hybrid(nn.Module):
    
    def __init__(self):
        super(Hybrid, self).__init__()

        L = 2
        N = 4

        layers = []
        layers.append(mf.Quad(3, N, second_order=True))
        layers.append(nn.Tanh())
        for _ in range(L):
            layers.append(nn.Linear(N, N))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(N, 1))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        x = self.layers(x)

        return x