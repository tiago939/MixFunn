import torch
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
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

model = models.MixFunn().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9,0.9))

N = 256
n = 2
E = (n*np.pi)**1
loss_low = np.inf

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
for epoch in range(10000):
    model.train()
    x = torch.rand((N,1), device=device, requires_grad=True)
    energy = E*torch.ones((N, 1), device=device)
        
    inputs = torch.cat((x, energy), axis=1)
    psi    = model(inputs)
    
    norm   = torch.sum(abs(psi)**2)
    loss_norm = torch.mean((N-norm)**2)
    
    dpsi_dx = torch.autograd.grad(psi.sum(), x, retain_graph=True, create_graph=True)[0]
    dpsi2_dx2 = torch.autograd.grad(dpsi_dx.sum(), x, retain_graph=True, create_graph=True)[0]

    residual = dpsi2_dx2 + (energy**2)*psi
    loss_residual = torch.mean((0-residual)**2)

    X0      = torch.zeros((N,1), device=device)
    inputs  = torch.cat((X0, energy), axis=1)
    psi     = model(inputs)
    loss_bc = torch.mean((0-psi)**2)

    X1      = torch.ones((N,1), device=device)
    inputs  = torch.cat((X1, energy), axis=1)
    psi     = model(inputs)
    loss_bc += torch.mean((0-psi)**2)

    loss  = loss_residual + loss_bc + loss_norm
    print(epoch, loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if loss < loss_low:
        loss_low = loss.item()
        x = torch.linspace(0, 1.0, N, device=device,requires_grad=True).reshape((N, 1))
        model.eval()
        inputs = torch.cat((x, energy), axis=1)
        psi2    = model(inputs)

        energy = (1*np.pi)*torch.ones((N, 1), device=device)
        inputs = torch.cat((x, energy), axis=1)
        psi1    = model(inputs)

        energy = (3*np.pi)*torch.ones((N, 1), device=device)
        inputs = torch.cat((x, energy), axis=1)
        psi3    = model(inputs)

        x = np.linspace(0.0, 1.0, N)
        target2 = (2**0.5)*np.sin(2*np.pi*x)
        target2 = abs(target2)**2

        target1 = (2**0.5)*np.sin(1*np.pi*x)
        target1 = abs(target1)**2

        target3 = (2**0.5)*np.sin(3*np.pi*x)
        target3 = abs(target3)**2
        
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax1.plot(x, (abs(psi1)**2).detach().cpu().numpy().flatten())
        ax1.plot(x, target1)

        ax2.plot(x, (abs(psi2)**2).detach().cpu().numpy().flatten())
        ax2.plot(x, target2)

        ax3.plot(x, (abs(psi3)**2).detach().cpu().numpy().flatten())
        ax3.plot(x, target3)
        plt.pause(0.0001)

plt.show()



    








