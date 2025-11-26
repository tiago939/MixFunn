import torch
import torch.nn as nn
import itertools
import random
import numpy as np

class Sin(torch.nn.Module):
    def forward(self, x):
        f = torch.sin(x)
        return f

class Cos(torch.nn.Module):
    def forward(self, x):
        f = torch.cos(x)
        return f

class Exp(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(x)
        return f

class ExpN(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(-x)
        return f

class ExpAbs(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(-0.01*abs(x))
        return f

class ExpAbsP(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(0.01*abs(x))
        return f

class Sqrt(torch.nn.Module):
    #this is an approximation of the square root function in order to avoid numerical instability
    def __init__(self):
        super(Sqrt, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        f = (0.01 + self.relu(x))**0.5
        return f

class Log(torch.nn.Module):
    #this is an approximation of the log function in order to avoid numerical instability
    def __init__(self):
        super(Log, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        f = torch.log(0.1 + self.relu(x))
        return f

class Id(torch.nn.Module):
    def forward(self, x):
        f = x
        return f

class Tanh(torch.nn.Module):
    def forward(self, x):
        f = torch.tanh(x)
        return f

functions = [Sin(), Cos(), ExpAbs(), ExpAbsP(), Sqrt(), Log(), Id()]
L = len(functions)


class Quad(nn.Module):
    def __init__(self, n_in, n_out, second_order=True):
        super(Quad, self).__init__()
        
        self.second_order = second_order

        #first order neurons
        if not second_order:
            self.linear = nn.Linear(n_in, n_out)

        #second order neurons
        else:
            L = int(n_in*(n_in-1)/2)
            self.linear = nn.Linear(L + n_in, n_out)
            self.ids = torch.triu_indices(n_in, n_in, 1)

    def forward(self, x):

        # NICOLAS: notice that you run x = self.linear(x) and return it
        # independently if second_order or not

        if self.second_order:
            x2 = x[:, :, None] @ x[:, None, :]
            x2 = x2[:,self.ids[0], self.ids[1]]
            x = torch.cat((x, x2), axis=1)

        x = self.linear(x)
        return x


class Mixfun(nn.Module):
    def __init__(self, n_in, n_out, normalization_function=False, normalization_neuron=False, p_drop=False, second_order_input=False, second_order_function=False, temperature=1.0):
        super(Mixfun, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.p_drop = p_drop

        if p_drop:
            self.dropout = nn.Dropout(p=p_drop)
        self.second_order_function = second_order_function
        self.temperature = temperature

        #first order projection
        self.project1 = Quad(n_in, L*n_out, second_order=second_order_input)

        self.l = int(L*(L+1)/2) * second_order_function # 0 otherwise
        if second_order_function:
            #second order projection
            self.project2_1 = Quad(n_in, L*n_out, second_order=second_order_function)
            self.project2_2 = Quad(n_in, L*n_out, second_order=second_order_function)
            self.ids = torch.triu_indices(L, L, 0)

        #neuron output
        self.normalization_function = normalization_function #forces each neuron to choose a single function
        self.normalization_neuron = normalization_neuron #forces each neuron to have a different function from the others

        # NICOLAS: notice that the changes between second_order_function
        # or not is the second argument of torch.ones/randn.
        # If second_order_function, we add self.l, which is only defined when
        # second_order_function is True. Then, we can go branchless
        # by defining "self.l = int(L*(L+1)/2) * second_order_function".


        # the following cases are TT, FT, TF and FF. Notice that TF and FT
        # are the same case: self.p = nn.Parameter(torch.ones(n_out, torch_L))
        # thus, the second case can be written with elifs and an "or"

        if normalization_function and normalization_neuron:
            self.p1 = nn.Parameter(torch.ones(n_out, L + self.l))
            self.p2 = nn.Parameter(torch.ones(n_out, L + self.l))

        elif normalization_function or normalization_neuron:
            self.p = nn.Parameter(torch.ones(n_out, L + self.l))

        else:
            self.p = nn.Parameter(torch.randn(n_out, L + self.l))


        if normalization_function or normalization_neuron:
            self.amplitude = nn.Parameter(torch.randn(n_out))


    # __methods are hidden to the user
    def __project_and_stack(self, x, projection):
        y = projection(x).reshape((x.shape[0], self.n_out, L))

        # enumerate is more pythonic
        y = [fun(y[:, :, i]) for i, fun in enumerate(functions)]

        y = torch.stack(y, dim=1).reshape((x.shape[0], self.n_out, L))
        return y


    def forward(self, x):

        # NICOLAS: notice the same structure being called over and over
        # we can abstract it, just fixing the correct projection

        if self.second_order_function:
            #first order functions
            x1 = self.__project_and_stack(x, self.project1)

            #second order functions
            x2_1 = self.__project_and_stack(x, self.project2_1)
            x2_2 = self.__project_and_stack(x, self.project2_2)
            x2 = x2_1[:, :, :, None] @ x2_2[:, :, None, :]
            x2 = x2[:, :, self.ids[0], self.ids[1]]

            #concatenate first order and second order functions
            x = torch.cat((x1,x2), axis=2)

        else:
            #first order functions
            x = self.__project_and_stack(x, self.project1)

        if self.p_drop and self.training:
            x = self.dropout(x)

        # NICOLAS: here, notice that we have the condition TT, TF, FT and FF.
        # if we have skipped TT, then it means either one (or both) is false.
        # Then, we don't need to check BOTH each time, we can check only one

        if self.normalization_function and self.normalization_neuron:
            Z = torch.sum(torch.exp(-self.p1/self.temperature), axis=1)
            p1 = torch.exp(-self.p1/self.temperature)/(1e-8 + Z.reshape((self.n_out, 1)))

            Z = torch.sum(torch.exp(-self.p2/self.temperature), axis=0)

            # once again, as we defined self.l = 0 if not second_order_function
            # we can remove the branching here
            p2 = torch.exp(-self.p2/self.temperature)/(1e-8 + Z.reshape((1, L + self.l)))

            x = self.amplitude * torch.sum(p2 * p1 * x, axis=2)

        elif self.normalization_function:
            Z = torch.sum(torch.exp(-self.p/self.temperature), axis=1)
            p = torch.exp(-self.p/self.temperature)/(1e-8 + Z.reshape((self.n_out, 1)))
            x = self.amplitude * torch.sum(p * x, axis=2)

        elif self.normalization_neuron:
            Z = torch.sum(torch.exp(-self.p/self.temperature), axis=0)
            p = torch.exp(-self.p/self.temperature)/(1e-8 + Z.reshape((1, L + self.l)))
            x = self.amplitude * torch.sum(p * x, axis=2)

        else:
            x = torch.sum(self.p * x, axis=2)

        return x
