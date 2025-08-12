import torch
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import models

device = 'cuda'

#random seed
manualSeed = 1
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True

model = models.MixFunn(normalization_function = False,
                       normalization_neuron = False,
                       p_drop = False,
                       second_order_input = True,
                       second_order_function = False
                       ).to(device)
                       
optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9,0.9))

N = 256
tmin = 0.0
tmax = 40.0
t = np.linspace(tmin, tmax, N)

x0 = 1.0 #initial position
v0 = 0.0 #initial velocity
k = 1.0 #elastic spring constant
b = 0.1 #damping coefficient
m = 1.0 #mass
F0 = 1.0 #force amplitude
omega = 0.9 #force frequency


T = torch.tensor(t).reshape((N, 1)).float().to(device)
t2 = np.linspace(tmin, 2*tmax, N)
T2 = torch.tensor(t2).reshape((N, 1)).float().to(device)

def damped_oscillator(t, y):
    x1, x2 = y
    dx1dt = x2
    dx2dt = (F0 * np.cos(omega * t) - b * x2 - k * x1) / m
    return [dx1dt, dx2dt]

t_values = np.linspace(tmin, 2*tmax, N)

initial_conditions = [x0, v0]

solution = solve_ivp(damped_oscillator, [tmin, 2*tmax], initial_conditions, t_eval=t_values, method='RK45')

target = solution.y[0]

loss_list = []
loss_train = []
loss_test = []
loss_low = np.inf
for epoch in range(10000):

    model.train()
    T = tmax*torch.rand((N,1), device=device)
    T.requires_grad_(True)
    F = F0*torch.cos(omega*T).reshape((N, 1))
    inputs = torch.cat((T, F.detach()), axis=1)

    x = model(T)
    
    dx_dt = torch.autograd.grad(x.sum(), T, retain_graph=True, create_graph=True)[0]
    dx2_dt2 = torch.autograd.grad(dx_dt.sum(), T, retain_graph=True, create_graph=True)[0]

    residual = m*dx2_dt2 + k*x + b*dx_dt - F0*torch.cos(omega*T)
    
    loss_residual = torch.mean((0-residual)**2)

    T0 = torch.zeros((1,1), device=device)
    T0.requires_grad_(True)
    F = F0*torch.cos(omega*T0).reshape((1, 1))
    inputs = torch.cat((T0, F.detach()), axis=1)
    x = model(T0)
    dx_dt = torch.autograd.grad(x.sum(), T0, retain_graph=True, create_graph=True)[0]

    loss_ic = (x0 - x)**2 + (v0 - dx_dt)**2

    loss = loss_residual + loss_ic
    print(epoch, loss.item())
    loss_list.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    F = F0*torch.cos(omega*T2).reshape((N, 1))
    inputs = torch.cat((T2, F), axis=1)
    x = model(T2).detach().cpu().numpy().flatten()

    loss_train.append(np.mean( (x[0:N//2] - target[0:N//2])**2))
    loss_test.append(np.mean( (x[N//2:N] - target[N//2:N])**2))
    
    if loss.item() < loss_low:
        torch.save({'model': model.state_dict()}, 'best_model_%i.pt' % manualSeed)

    if epoch % 1 == 0 and loss < loss_low:
        loss_low = 1*loss.item()
        plt.clf()
        plt.plot(t2, target, label='solution')
        plt.plot(t2, x, label='NN')
        plt.axvline(x = tmax, color = 'black')
        plt.text(tmax/4, 1.3, 'Training domain')
        plt.text(1.2*tmax, 1.3, 'Generalization domain')
        plt.ylim([-6, 6])
        plt.xlabel('time t')
        plt.ylabel('position x(t)')
        plt.legend()
        plt.pause(0.0001)






