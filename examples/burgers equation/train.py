import torch
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
import models
import json
from scipy.integrate import solve_ivp

device = 'cuda'
batch_size = 1024

#Numerical solution (for error evaluation)
##########################################################################
x = np.linspace(-1.0, 1.0, 100)
dx = x[1] - x[0]
u0 = -np.sin(np.pi * x)

def pde(t, u):
    dudx = np.zeros_like(u)
    d2udx2 = np.zeros_like(u)
    
    dudx[1:-1] = (u[2:] - u[:-2]) / (2 * dx)
    d2udx2[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    
    dudt = -u * dudx + 0.01 * d2udx2
    
    u[0] = 0
    u[-1] = 0

    return dudt

sol = solve_ivp(pde, t_span=(0.0, 2.0), y0=u0, method='RK45', t_eval=np.linspace(0.0, 2.0, 100))

solution = sol.y.T
time_points = sol.t

x, t = np.meshgrid(x, time_points)
# plt.figure(figsize=(8, 6))
# #plt.contourf(t, x, solution, levels=100, cmap='viridis')
# plt.scatter(t.flatten(), x.flatten(), c=solution.flatten())
# plt.colorbar(label='u(x, t)')
# plt.xlabel('t')
# plt.ylabel('x')
# plt.title('Solution of the PDE')
# plt.show()
# sys.exit()

solution_train = solution[t <= 1.0]
t_train = t[t <= 1.0]
x_train = x[t <= 1.0]
solution_test = solution[t > 1.0]
t_test = t[t > 1.0]
x_test = x[t > 1.0]
X = torch.tensor(x.reshape((x.shape[0]*x.shape[1],1))).float().to(device)
T = torch.tensor(t.reshape((t.shape[0]*t.shape[1],1))).float().to(device)
y = - torch.sin(np.pi*X)
inputs_test = torch.cat((X,T,y),axis=1)
X2 = X.cpu().numpy().flatten()
T2 = T.cpu().numpy().flatten()

#################################################################################

N = 40
x2 = np.linspace(-1, 1, N)
t2 = np.linspace(0, 1, N)
x2, t2 = np.meshgrid(x2, t2)
x2 = torch.tensor(x2.reshape((N * N, 1)), device=device).float()
t2 = torch.tensor(t2.reshape((N * N, 1)), device=device).float()
y = -torch.sin(np.pi*x2)
inputs2 = torch.cat((x2,t2,y),axis=1)
x2 = x2.cpu().numpy().flatten()
t2 = t2.cpu().numpy().flatten()

#random seed
manualSeed = 3
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True

model = models.MixFunn().to(device)

number_of_parameters=0
for p in list(model.parameters()):
    nn=1
    for s in list(p.size()):
        nn = nn*s
    number_of_parameters += nn

optimizer = optim.Adam(model.parameters(), lr=0.1, betas=(0.9,0.9))

loss_low = np.inf
error_train = np.inf
error_test = np.inf
for epoch in range(10000):

    model.train()
    
    x_input = 2*torch.rand((batch_size, 1), device=device) - 1.0
    t_input = 1*torch.rand((batch_size, 1), device=device)
    y = -torch.sin(np.pi*x_input)
    t_input.requires_grad_(True)
    x_input.requires_grad_(True)

    inputs = torch.cat((x_input,t_input,y),axis=1)
    u = model(inputs)

    ## get residual
    du_dx = torch.autograd.grad(u.sum(), x_input, create_graph=True, retain_graph=True)[0]
    du_dt = torch.autograd.grad(u.sum(), t_input, create_graph=True, retain_graph=True)[0]
    du2_dx2 = torch.autograd.grad((du_dx).sum(), x_input, create_graph=True, retain_graph=True)[0]
    residual = du_dt + u*du_dx - 0.01*du2_dx2

    ## initial and boundary conditions u(x,t)
    #u(x,0)
    t0 = 0.0*torch.rand((batch_size,1),device=device)
    x0 = 2.0*torch.rand((batch_size,1),device=device) - 1.0
    y = -torch.sin(np.pi*x0)
    x0.requires_grad_(True)
    inputs = torch.cat((x0,t0,y),axis=1)
    u = model(inputs)
    du_dx = torch.autograd.grad(u.sum(), x0, create_graph=True, retain_graph=True)[0]
    du2_dx2 = torch.autograd.grad((du_dx).sum(), x0, create_graph=True, retain_graph=True)[0]
    bc_loss = torch.mean((-torch.sin(np.pi*x0)-u)**2)
    bcd_loss = torch.mean((-np.pi*torch.cos(np.pi*x0)-du_dx)**2)
    bcd_loss2 = torch.mean((np.pi*np.pi*torch.sin(np.pi*x0)-du2_dx2)**2)

    #u(t,1)
    t0 = torch.rand((batch_size,1),device=device)
    x0 = torch.ones((batch_size,1),device=device)
    y = -torch.sin(np.pi*x0)
    inputs = torch.cat((x0,t0,y),axis=1)
    u = model(inputs)
    bc_loss2 = torch.mean((0.0-u)**2)

    #u(t,-1)
    t0 = torch.rand((batch_size,1),device=device)
    x0 = -torch.ones((batch_size,1),device=device)
    y = -torch.sin(np.pi*x0)
    inputs = torch.cat((x0,t0,y),axis=1)
    u = model(inputs)
    bc_loss3 = torch.mean((0.0-u)**2)

    loss = torch.mean((0-residual)**2) + bc_loss + bc_loss2 + bc_loss3 + bcd_loss + bcd_loss2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ###### Evaluate model

    model.eval()
    N = 40
    x2 = np.linspace(-1, 1, N)
    t2 = np.linspace(0, 1, N)
    x2, t2 = np.meshgrid(x2, t2)
    x_input = torch.tensor(x2.reshape((N * N, 1)), device=device).float()
    t_input = torch.tensor(t2.reshape((N * N, 1)), device=device).float()
    x_input.requires_grad_(True)
    t_input.requires_grad_(True)
    y = -torch.sin(np.pi*x_input)
    inputs = torch.cat((x_input,t_input,y),axis=1)
    u = model(inputs)

    ## get residual
    du_dx = torch.autograd.grad(u.sum(), x_input, create_graph=True, retain_graph=True)[0]
    du_dt = torch.autograd.grad(u.sum(), t_input, create_graph=True, retain_graph=True)[0]
    du2_dx2 = torch.autograd.grad((du_dx).sum(), x_input, create_graph=True, retain_graph=True)[0]
    residual = du_dt + u*du_dx - 0.01*du2_dx2

    ## initial and boundary conditions u(x,t)
    #u(x,0)
    t0 = 0.0*torch.rand((batch_size,1),device=device)
    x0 = 2.0*torch.rand((batch_size,1),device=device) - 1.0
    y = -torch.sin(np.pi*x0)
    x0.requires_grad_(True)
    inputs = torch.cat((x0,t0,y),axis=1)
    u = model(inputs)
    du_dx = torch.autograd.grad(u.sum(), x0, create_graph=True, retain_graph=True)[0]
    du2_dx2 = torch.autograd.grad((du_dx).sum(), x0, create_graph=True, retain_graph=True)[0]
    bc_loss = torch.mean((-torch.sin(np.pi*x0)-u)**2)
    bcd_loss = torch.mean((-np.pi*torch.cos(np.pi*x0)-du_dx)**2)
    bcd_loss2 = torch.mean((np.pi*np.pi*torch.sin(np.pi*x0)-du2_dx2)**2)

    #u(t,1)
    t0 = torch.rand((batch_size,1),device=device)
    x0 = torch.ones((batch_size,1),device=device)
    y = -torch.sin(np.pi*x0)
    inputs = torch.cat((x0,t0,y),axis=1)
    u = model(inputs)
    bc_loss2 = torch.mean((0.0-u)**2)

    #u(t,-1)
    t0 = torch.rand((batch_size,1),device=device)
    x0 = -torch.ones((batch_size,1),device=device)
    y = -torch.sin(np.pi*x0)
    inputs = torch.cat((x0,t0,y),axis=1)
    u = model(inputs)
    bc_loss3 = torch.mean((0.0-u)**2)

    loss = torch.mean((0-residual)**2) + bc_loss + bc_loss2 + bc_loss3 + bcd_loss + bcd_loss2

    print(epoch, loss.item(), loss_low)

    if loss < loss_low:

        loss_low = loss.item()
        model.eval()
        u = model(inputs_test).detach().cpu().numpy().flatten()
        u_train = u[t.flatten() <= 1.0]
        u_test = u[t.flatten() > 1.0]
        error = abs(u - solution.flatten())
        error_train = np.mean(abs(u_train - solution_train))
        error_test = np.mean(abs(u_test - solution_test))
        
        plt.clf()
        plt.scatter(T2, X2, c=u, marker='s', s=175)
        plt.xlabel('t')
        plt.ylabel('x')
        plt.colorbar()
        plt.pause(0.00001)




    








