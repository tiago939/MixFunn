import torch
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import models
import json

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

model = models.MixFunn().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9,0.9))

batch_size = 256
tmin = 0.0
tmax = 40.0

x0 = 1.0 #initial position
v0 = 0.0 #initial velocity
k = 1.0 #elastic spring constant
c = 0.1 #damping coefficent
m = 1.0 #mass

t = np.linspace(tmin, 2 * tmax, batch_size)
target = x0 * np.cos(t*((k/m)**0.5)) * np.exp(-c*t/(2*m))

loss_min = np.inf
for epoch in range(10000):

    model.train()
    T = tmax*torch.rand((batch_size, 1), device=device)
    T.requires_grad_(True)
    x = model(T)
    
    dx_dt = torch.autograd.grad(x.sum(), T, retain_graph=True, create_graph=True)[0]
    dx2_dt2 = torch.autograd.grad(dx_dt.sum(), T, retain_graph=True, create_graph=True)[0]

    residual = m*dx2_dt2 + k*x + c*dx_dt
    
    loss_residual = torch.mean((0-residual)**2)

    T0 = torch.zeros((1,1), device=device)
    T0.requires_grad_(True)
    x = model(T0)
    dx_dt = torch.autograd.grad(x.sum(), T0, retain_graph=True, create_graph=True)[0]

    loss_ic = (x0 - x)**2 + (v0 - dx_dt)**2

    loss = loss_residual + loss_ic

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()

    T = torch.linspace(tmin, 2 * tmax, batch_size, device=device).reshape((batch_size, 1))
    T.requires_grad_(True)
    x = model(T)
    X = x.detach().cpu().numpy().flatten()
    dx_dt = torch.autograd.grad(x.sum(), T, retain_graph=True, create_graph=True)[0]
    dx2_dt2 = torch.autograd.grad(dx_dt.sum(), T, retain_graph=True, create_graph=True)[0]
    residual = m*dx2_dt2 + k*x + c*dx_dt
    loss_residual = torch.mean((0-residual)**2)

    T0 = torch.zeros((1,1), device=device)
    T0.requires_grad_(True)
    x = model(T0)
    dx_dt = torch.autograd.grad(x.sum(), T0, retain_graph=True, create_graph=True)[0]
    loss_ic = (x0 - x)**2 + (v0 - dx_dt)**2

    loss = loss_residual + loss_ic

    #only plot for lowest total loss
    if loss < loss_min:
        loss_min = loss.item()
    
        plt.clf()
        plt.plot(t, target, label='solution')
        plt.plot(t, X, label='model')
        plt.axvline(x = tmax, color = 'black')
        plt.text(tmax/4, 1.3, 'Training domain')
        plt.text(1.2*tmax, 1.3, 'Generalization domain')
        plt.ylim([-1.2, 1.2])
        plt.xlabel('time t')
        plt.ylabel('position x(t)')
        plt.legend()
        plt.pause(0.0001)

plt.show()






