import torch
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb}')


#RK4:
def h1_error(sol,sols,grid_fin,grids,len_grids):
    sol_len=len(sol)
    errs=torch.zeros(len(len_grids)-1,dtype=torch.float64)
    interp_sol=torch.zeros((len_grids[-1],len(len_grids)-1)+tuple(list(sol[0].shape)),dtype=torch.float64)
    for i in range(len(len_grids)-1):
        cumsum=torch.cumsum(len_grids,0)
        cumsum=torch.cat((torch.tensor([0]),cumsum))
        grid=grids[cumsum[i]:cumsum[i+1]]
        sol_coarse=sols[cumsum[i]:cumsum[i+1]]
        interp_sol[0,i,::]=sol[0]
        k=0
        for j in range(sol_len-1):
            t=grid_fin[j+1]
            t0=grid[k]
            y0=sol_coarse[k]
            t1=grid[k+1]
            y1=sol_coarse[k+1]
            if t<=t0 or t>t1:
                k+=1
                t=grid_fin[j+1]
                t0=grid[k]
                y0=sol_coarse[k]
                t1=grid[k+1]
                y1=sol_coarse[k+1]

            if t == t0:
                interp_sol[j+1,i,::]=y0
            elif t == t1:
                interp_sol[j+1,i,::]=y1
            else:
                slope = (t - t0) / (t1 - t0)
                interp_sol[j+1,i,::]=y0 + slope * (y1 - y0)
        dy_fine=sol[1:,::]-sol[:-1,::]
        dt_fine=grid_fin[1:]-grid_fin[:-1]
        dy_coarse=interp_sol[1:,i,::]-interp_sol[:-1,i,::]
        for j in range(sol_len-1):
            dy_fine[j,::]=dy_fine[j,::]/torch.sqrt(dt_fine[j])
            dy_coarse[j,::]=dy_coarse[j,::]/torch.sqrt(dt_fine[j])
        errs[i]=torch.linalg.norm(dy_fine-dy_coarse)
    return errs
#forward pass:

#sol=torch.load("results/ref_sol_forward_RK4.pt",weights_only=False)
sols_ada=torch.load("results/sols_forward_RK4_ada.pt",weights_only=False)
sols_uni=torch.load("results/sols_forward_RK4_uni.pt",weights_only=False)

len_grids_ada=torch.load("results/len_grids_forward_RK4_ada.pt",weights_only=False)
len_grids_uni=torch.load("results/len_grids_forward_RK4_uni.pt",weights_only=False)

estis_ada=torch.sqrt(torch.load("results/estis_forward_RK4_ada.pt",weights_only=False))
estis_uni=torch.sqrt(torch.load("results/estis_forward_RK4_uni.pt",weights_only=False))

grids_ada=torch.load("results/grids_forward_RK4_ada.pt",weights_only=False)[:-len_grids_ada[-1]]
grids_uni=torch.load("results/grids_forward_RK4_uni.pt",weights_only=False)[:-len_grids_uni[-1]]

len_grids_ada=len_grids_ada[:-1]
len_grids_uni=len_grids_uni[:-1]

#calc H1 norm:
sol=sols_ada[range(-len_grids_ada[-1],0),::]
sol_len=len(sol)
grid_fin=grids_ada[range(-len_grids_ada[-1],0)]
errs_ada=h1_error(sol,sols_ada,grid_fin,grids_ada,len_grids_ada)

#interpolate adaptive sol to uniform grid:
sol_uni=torch.zeros_like(sols_uni[range(-len_grids_uni[-1],0),::])
sol_len=len(sol_uni)
grid_fin_uni=grids_uni[range(-len_grids_uni[-1],0)]

k=0
for i in range(sol_len):
    t=grid_fin_uni[i]
    
    while grid_fin[k]<=t:
        if grid_fin[k]==t:
            sol_uni[i]=sol[k]
            break
        elif grid_fin[k]<t and grid_fin[k+1]>t:
            slope=(t-grid_fin[k])/(grid_fin[k+1]-grid_fin[k])
            sol_uni[i]=sol[k]+slope*(sol[k+1]-sol[k])  
            break  
        k+=1
sol=sol_uni
errs_uni=h1_error(sol,sols_uni,grid_fin_uni,grids_uni,len_grids_uni)

plt.loglog(len_grids_ada.detach().numpy(), estis_ada.detach().numpy(), "+-", label=r"adaptive (estimator)")
plt.loglog(len_grids_uni.detach().numpy(), estis_uni.detach().numpy(), "x-", label=r"uniform (estimator)")
plt.loglog(len_grids_ada[:-1].detach().numpy(), errs_ada.detach().numpy(), "o-", label=r"adaptive (error)")
plt.loglog(len_grids_uni[:-1].detach().numpy(), errs_uni.detach().numpy(), "v-", label=r"uniform (error)")

# Reference line: 1/N
N = len_grids_uni.detach().numpy()
plt.loglog(N, 10*1/N, "--", label=r"${\# \mathcal{T}}^{-1}$")

plt.xlabel(r"$\# \mathcal{T}$")


plt.legend()
plt.grid(True)
plt.show()


#plt.plot(grid_fin_uni,sol[:,0])
#plt.show()
#backward pass:

#sol=torch.load("results/ref_sol_backward_RK4.pt")
sols_ada=torch.load("results/sols_backward_RK4_ada.pt",weights_only=False)
sols_uni=torch.load("results/sols_backward_RK4_uni.pt",weights_only=False)

len_grids_ada=torch.load("results/len_grids_backward_RK4_ada.pt",weights_only=False)
len_grids_uni=torch.load("results/len_grids_backward_RK4_uni.pt",weights_only=False)

estis_ada=torch.sqrt(torch.load("results/estis_backward_RK4_ada.pt",weights_only=False))
estis_uni=torch.sqrt(torch.load("results/estis_backward_RK4_uni.pt",weights_only=False))

grids_ada=torch.load("results/grids_backward_RK4_ada.pt",weights_only=False)[:-len_grids_ada[-1]]
grids_uni=torch.load("results/grids_backward_RK4_uni.pt",weights_only=False)[:-len_grids_uni[-1]]

len_grids_ada=len_grids_ada[:-1]
len_grids_uni=len_grids_uni[:-1]

#calc H1 norm:
sol=sols_ada[range(-len_grids_ada[-1],0),::]
print(sol.shape)
sol_len=len(sol)
grid_fin=grids_ada[range(-len_grids_ada[-1],0)]
errs_ada=h1_error(sol,sols_ada,grid_fin,grids_ada,len_grids_ada)

#interpolate adaptive sol to uniform grid:
sol_uni=torch.zeros_like(sols_uni[range(-len_grids_uni[-1],0),::])
sol_len=len(sol_uni)
grid_fin_uni=grids_uni[range(-len_grids_uni[-1],0)]

k=0
for i in range(sol_len):
    t=grid_fin_uni[i]
    
    while grid_fin[k]<=t:
        if grid_fin[k]==t:
            sol_uni[i]=sol[k]
            break
        elif grid_fin[k]<t and grid_fin[k+1]>t:
            slope=(t-grid_fin[k])/(grid_fin[k+1]-grid_fin[k])
            sol_uni[i]=sol[k]+slope*(sol[k+1]-sol[k])  
            break  
        k+=1
sol=sol_uni
errs_uni=h1_error(sol,sols_uni,grid_fin_uni,grids_uni,len_grids_uni)

plt.loglog(len_grids_ada.detach().numpy(), estis_ada.detach().numpy(), "+-", label=r"adaptive (estimator)")
plt.loglog(len_grids_uni.detach().numpy(), estis_uni.detach().numpy(), "x-", label=r"uniform (estimator)")
plt.loglog(len_grids_ada[:-1].detach().numpy(), errs_ada.detach().numpy(), "o-", label=r"adaptive (error)")
plt.loglog(len_grids_uni[:-1].detach().numpy(), errs_uni.detach().numpy(), "v-", label=r"uniform (error)")

# Reference line: 1/N
N = len_grids_uni.detach().numpy()
plt.loglog(N, 1000*1/N, "--", label=r"${\# \mathcal{T}}^{-1}$")

plt.xlabel(r"$\# \mathcal{T}$")


plt.legend()
plt.grid(True)
plt.show()

#plt.plot(grid_fin_uni,sol[:,0])
#plt.show()