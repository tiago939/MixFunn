import torch
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import models

device = 'cuda'
N = 128 
energies = np.linspace(0.0, 4.5*np.pi, 20)

losses = []
for E in energies:
    print('Energy candidate: ', E)

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

    loss_min = np.inf
    for epoch in range(500):
        model.train()
        x = torch.rand((N,1), device=device, requires_grad=True)
        energy = E*torch.ones((N, 1), device=device)
            
        inputs = torch.cat((x, energy), axis=1)
        psi    = model(inputs)
        
        norm   = torch.sum(abs(psi)**2)
        loss_norm = (N-norm)**2
        
        dpsi_dx = torch.autograd.grad(psi.sum(), x, retain_graph=True, create_graph=True)[0]
        dpsi2_dx2 = torch.autograd.grad(dpsi_dx.sum(), x, retain_graph=True, create_graph=True)[0]

        residual = dpsi2_dx2 + (energy**2)*psi
        loss_residual = torch.sum((0-residual)**2)

        X0      = torch.zeros((N,1), device=device)
        inputs  = torch.cat((X0, energy), axis=1)
        psi     = model(inputs)

        loss_bc = torch.sum((0-psi)**2)

        X1      = torch.ones((N,1), device=device)
        inputs  = torch.cat((X1, energy), axis=1)
        psi     = model(inputs)
        loss_bc += torch.sum((0-psi)**2)

        loss  = loss_residual + loss_bc + loss_norm

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #----------------------------------------------------------------------------
        model.eval()
        x = torch.linspace(0, 1, N, device=device).reshape((N, 1))
        x.requires_grad_(True)
        energy = E*torch.ones((N, 1), device=device)
            
        inputs = torch.cat((x, energy), axis=1)
        psi    = model(inputs)
        
        norm   = torch.sum(abs(psi)**2)
        loss_norm = (N-norm)**2
        
        dpsi_dx = torch.autograd.grad(psi.sum(), x, retain_graph=True, create_graph=True)[0]
        dpsi2_dx2 = torch.autograd.grad(dpsi_dx.sum(), x, retain_graph=True, create_graph=True)[0]

        residual = dpsi2_dx2 + (energy**2)*psi
        loss_residual = torch.sum((0-residual)**2)

        X0      = torch.zeros((N,1), device=device)
        inputs  = torch.cat((X0, energy), axis=1)
        psi     = model(inputs)

        loss_bc = torch.sum((0-psi)**2)

        X1      = torch.ones((N,1), device=device)
        inputs  = torch.cat((X1, energy), axis=1)
        psi     = model(inputs)
        loss_bc += torch.sum((0-psi)**2)

        loss  = loss_residual + loss_bc + loss_norm

        if loss < loss_min:
            loss_min = loss.item()
    
    losses.append(loss_min)

plt.plot(energies, losses)
plt.xlabel('Energy')
plt.ylabel('Loss')
plt.grid()
plt.show()
    



    








