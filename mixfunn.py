import torch
import torch.nn as nn
import torch.nn.functional as F


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
        f = torch.exp(-0.01 * abs(x))
        return f


class ExpAbsP(torch.nn.Module):
    def forward(self, x):
        f = torch.exp(0.01 * abs(x))
        return f


class Sqrt(torch.nn.Module):
    # this is an approximation of the square root
    # function in order to avoid numerical instability
    def __init__(self):
        super(Sqrt, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        f = (0.01 + self.relu(x)) ** 0.5
        return f


class Log(torch.nn.Module):
    # this is an approximation of the log
    # function in order to avoid numerical instability
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

        # first order neurons
        if not second_order:
            self.linear = nn.Linear(n_in, n_out)

        # second order neurons
        else:
            L = int(n_in * (n_in - 1) / 2)
            self.linear = nn.Linear(L + n_in, n_out)
            self.ids = torch.triu_indices(n_in, n_in, 1)

    def forward(self, x):

        # NICOLAS: notice that you run x = self.linear(x) and return it
        # independently if second_order or not

        if self.second_order:
            x2 = x[:, :, None] @ x[:, None, :]
            x2 = x2[:, self.ids[0], self.ids[1]]
            x = torch.cat((x, x2), axis=1)

        x = self.linear(x)
        return x


class Mixfun(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        normalization_function=False,
        normalization_neuron=False,
        p_drop=False,
        second_order_input=False,
        second_order_function=False,
        temperature=1.0,
    ):
        super(Mixfun, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.p_drop = p_drop

        if p_drop:
            self.dropout = nn.Dropout(p=p_drop)
        self.second_order_function = second_order_function
        self.temperature = temperature

        # first order projection
        self.project1 = Quad(n_in, L * n_out, second_order=second_order_input)

        # NICOLAS: notice that the changes between second_order_function
        # or not is the second argument of torch.ones/randn.
        # If second_order_function, we add self.l, which is only defined when
        # second_order_function is True. Then, we can go branchless
        # by defining self.l = int(L*(L+1)/2) * second_order_function

        self.length = int(L * (L + 1) / 2) * int(second_order_function)
        self.F = L + self.length  # avoid recomputation

        if second_order_function:
            # second order projection
            self.project21 = Quad(
                n_in, L * n_out, second_order=second_order_function
            )
            self.project22 = Quad(
                n_in, L * n_out, second_order=second_order_function
            )
            self.ids = torch.triu_indices(L, L, 0)

        # neuron output
        # forces each neuron to choose a single function
        self.normalization_function = normalization_function

        # forces each neuron to have a different function from the others
        self.normalization_neuron = normalization_neuron

        # the following cases are TT, FT, TF and FF. Notice that TF and FT
        # are the same case: self.p = nn.Parameter(torch.ones(n_out, torch_L))
        # thus, the second case can be written with elifs and an "or"

        if not (normalization_function or normalization_neuron):
            self.p_raw = nn.Parameter(torch.randn(n_out, self.F))

        if normalization_function:
            self.p_fun = nn.Parameter(torch.ones(n_out, self.F))

        else:
            self.p_fun = None

        if normalization_neuron:
            self.p_neuron = nn.Parameter(torch.ones(n_out, self.F))

        else:
            self.p_neuron = None

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
            # first order functions
            x1 = self.__project_and_stack(x, self.project1)

            # second order functions
            x2_1 = self.__project_and_stack(x, self.project21)
            x2_2 = self.__project_and_stack(x, self.project22)
            x2 = x2_1[:, :, :, None] @ x2_2[:, :, None, :]
            x2 = x2[:, :, self.ids[0], self.ids[1]]

            # concatenate first order and second order functions
            x = torch.cat((x1, x2), axis=2)

        else:
            # first order functions
            x = self.__project_and_stack(x, self.project1)

        if self.p_drop and self.training:
            x = self.dropout(x)

        # NICOLAS: we rewrite this using inner softmax function
        # from pytorch

        if not (self.normalization_function and self.normalization_neuron):
            return torch.sum(self.p_raw * x, axis=2)

        if self.normalization_function:
            p_fun = F.softmax(-self.p_fun / self.temperature, dim=1)

        else:
            p_fun = 1.0

        if self.normalization_neuron:
            p_neuron = F.softmax(-self.p_neuron / self.temperature, dim=0)

        else:
            p_neuron = 1.0

        x = self.amplitude * torch.sum(p_neuron * p_fun * x, axis=2)
        return x
