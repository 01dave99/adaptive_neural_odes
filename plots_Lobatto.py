import torch
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb}')

#Lobatto:
def h1_error(sol,sols,grid_fin,grids,len_grids,method="Lobatto"):
    if method=="Lobatto":
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
            for j in range((sol_len-1)//2):
                if method=="Lobatto":
                    thalf=grid_fin[2*j+1]
                    t=grid_fin[2*j+2]
                    t0=grid[2*k]
                    y0=sol_coarse[2*k]
                    thalf_c=grid[2*k+1]
                    yhalf=sol_coarse[2*k+1]
                    t1=grid[2*k+2]
                    y1=sol_coarse[2*k+2]                
                if  not (thalf>t0 and t<=t1):
                    k+=1
                    thalf=grid_fin[2*j+1]
                    t=grid_fin[2*j+2]
                    t0=grid[2*k]
                    y0=sol_coarse[2*k]
                    thalf_c=grid[2*k+1]
                    yhalf=sol_coarse[2*k+1]
                    t1=grid[2*k+2]
                    y1=sol_coarse[2*k+2]    
                def interpolant(t):
                    return y0*(t-thalf_c)*(t-t1)/((t0-thalf_c)*(t0-t1))+yhalf*(t-t0)*(t-t1)/((thalf_c-t0)*(thalf_c-t1))+y1*(t-t0)*(t-thalf_c)/((t1-t0)*(t1-thalf_c))
                interp_sol[2*j+1,i,::]=interpolant(thalf)
                interp_sol[2*j+2,i,::]=interpolant(t)


            dy_fine=sol[1:,::]-sol[:-1,::]
            dt_fine=grid_fin[1:]-grid_fin[:-1]
            dy_coarse=interp_sol[1:,i,::]-interp_sol[:-1,i,::]
            for j in range(sol_len-1):
                dy_fine[j,::]=dy_fine[j,::]/torch.sqrt(dt_fine[j])
                dy_coarse[j,::]=dy_coarse[j,::]/torch.sqrt(dt_fine[j])
            errs[i]=torch.linalg.norm(dy_fine-dy_coarse)
        return errs
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

#sol=torch.load("results/ref_sol_latent_forward_Lobatto_full_full.pt")
sols_ada=torch.load("results/sols_latent_forward_Lobatto_ada.pt",weights_only=False)
sols_uni=torch.load("results/sols_latent_forward_Lobatto_uni.pt",weights_only=False)

len_grids_ada=torch.load("results/len_grids_latent_forward_Lobatto_ada.pt",weights_only=False)
len_grids_uni=torch.load("results/len_grids_latent_forward_Lobatto_uni.pt",weights_only=False)

estis_ada=torch.sqrt(torch.load("results/estis_latent_forward_Lobatto_ada.pt",weights_only=False))
estis_uni=torch.sqrt(torch.load("results/estis_latent_forward_Lobatto_uni.pt",weights_only=False))

grids_ada=torch.load("results/grids_latent_forward_Lobatto_ada.pt",weights_only=False)[:-len_grids_ada[-1]]
grids_uni=torch.load("results/grids_latent_forward_Lobatto_uni.pt",weights_only=False)[:-len_grids_uni[-1]]

len_grids_ada=len_grids_ada[:-1]
len_grids_uni=len_grids_uni[:-1]

#calc H1 norm:
#extend grids:
grids_ada_new=torch.zeros(sols_ada.shape[0],dtype=torch.float64)
grids_uni_new=torch.zeros(sols_uni.shape[0],dtype=torch.float64)
len_grids_ada_new=len_grids_ada*2-1
len_grids_uni_new=len_grids_uni*2-1
for i in range(len(len_grids_ada)):
    cumsum=torch.cumsum(len_grids_ada,0)
    cumsum=torch.cat((torch.tensor([0]),cumsum))
    grid=grids_ada[cumsum[i]:cumsum[i+1]]
    new_vs=(grid.clone().detach()[:-1]+grid.clone().detach()[1:])/2.
    new_grid,idxs=torch.sort(torch.cat((grid,new_vs)))
    cumsum_new=torch.cumsum(len_grids_ada_new,0)
    cumsum_new=torch.cat((torch.tensor([0]),cumsum_new))
    grids_ada_new[cumsum_new[i]:cumsum_new[i+1]]=new_grid
for i in range(len(len_grids_uni)):
    cumsum=torch.cumsum(len_grids_uni,0)
    cumsum=torch.cat((torch.tensor([0]),cumsum))
    grid=grids_uni[cumsum[i]:cumsum[i+1]]
    new_vs=(grid.clone().detach()[:-1]+grid.clone().detach()[1:])/2.
    new_grid,idxs=torch.sort(torch.cat((grid,new_vs)))
    cumsum_new=torch.cumsum(len_grids_uni_new,0)
    cumsum_new=torch.cat((torch.tensor([0]),cumsum_new))
    grids_uni_new[cumsum_new[i]:cumsum_new[i+1]]=new_grid

len_grids_ada=len_grids_ada_new
len_grids_uni=len_grids_uni_new
grids_ada=grids_ada_new
grids_uni=grids_uni_new

sol=sols_ada[range(-len_grids_ada[-1],0),::]
sol_len=len(sol)
grid_fin=grids_ada[range(-len_grids_ada[-1],0)]
errs_ada=h1_error(sol,sols_ada,grid_fin,grids_ada,len_grids_ada)

#interpolate adaptive sol to uniform grid:
sol_uni=torch.zeros_like(sols_uni[range(-len_grids_uni[-1],0),::])
sol_len=len(sol_uni)
grid_fin_uni=grids_uni[range(-len_grids_uni[-1],0)]

k=0
sol_uni[0,::]=sol[0,::]
for j in range((sol_len-1)):

    t=grid_fin_uni[j+1]
    t0=grid_fin[2*k]
    y0=sol[2*k]
    thalf_c=grid_fin[2*k+1]
    yhalf=sol[2*k+1]
    t1=grid_fin[2*k+2]
    y1=sol[2*k+2] 
    while not (t>t0 and t<=t1):
        k+=1
        t0=grid_fin[2*k]
        y0=sol[2*k]
        thalf_c=grid_fin[2*k+1]
        yhalf=sol[2*k+1]
        t1=grid_fin[2*k+2]
        y1=sol[2*k+2]                   
    def interpolant(t):
        return y0*(t-thalf_c)*(t-t1)/((t0-thalf_c)*(t0-t1))+yhalf*(t-t0)*(t-t1)/((thalf_c-t0)*(thalf_c-t1))+y1*(t-t0)*(t-thalf_c)/((t1-t0)*(t1-thalf_c))
    sol_uni[j+1,::]=interpolant(t)
sol=sol_uni
errs_uni=h1_error(sol,sols_uni,grid_fin_uni,grids_uni,len_grids_uni)

plt.loglog(len_grids_ada.detach().numpy()/2., estis_ada.detach().numpy(), "+-", label=r"adaptive (estimator)")
plt.loglog(len_grids_uni.detach().numpy()/2., estis_uni.detach().numpy(), "x-", label=r"uniform (estimator)")
plt.loglog(len_grids_ada[:-1].detach().numpy()/2., errs_ada.detach().numpy(), "o-", label=r"adaptive (error)")
plt.loglog(len_grids_uni[:-1].detach().numpy()/2., errs_uni.detach().numpy(), "v-", label=r"uniform (error)")

# Reference line: 1/N
N = len_grids_uni.detach().numpy()/2.
plt.loglog(N, 1000*1/N**2, "--", label=r"${\# \mathcal{T}}^{-2}$")

plt.xlabel(r"$\# \mathcal{T}$")


plt.legend()
plt.grid(True)
plt.savefig("latent_Lobatto_forward_real.eps")
plt.clf()
#backward pass:

#sol=torch.load("results/ref_sol_latent_backward_Lobatto_full_full.pt")
sols_ada=torch.load("results/sols_latent_backward_Lobatto_ada.pt",weights_only=False)
sols_uni=torch.load("results/sols_latent_backward_Lobatto_uni.pt",weights_only=False)

len_grids_ada=torch.load("results/len_grids_latent_backward_Lobatto_ada.pt",weights_only=False)
len_grids_uni=torch.load("results/len_grids_latent_backward_Lobatto_uni.pt",weights_only=False)

estis_ada=torch.sqrt(torch.load("results/estis_latent_backward_Lobatto_ada.pt",weights_only=False))
estis_uni=torch.sqrt(torch.load("results/estis_latent_backward_Lobatto_uni.pt",weights_only=False))

grids_ada=torch.load("results/grids_latent_backward_Lobatto_ada.pt",weights_only=False)[:-len_grids_ada[-1]]

grids_uni=torch.load("results/grids_latent_backward_Lobatto_uni.pt",weights_only=False)[:-len_grids_uni[-1]]

len_grids_ada=len_grids_ada[:-1]
len_grids_uni=len_grids_uni[:-1]


#calc H1 norm:
#extend grids:
grids_ada_new=torch.zeros(sols_ada.shape[0],dtype=torch.float64)
grids_uni_new=torch.zeros(sols_uni.shape[0],dtype=torch.float64)
len_grids_ada_new=len_grids_ada*2-1
len_grids_uni_new=len_grids_uni*2-1
for i in range(len(len_grids_ada)):
    cumsum=torch.cumsum(len_grids_ada,0)
    cumsum=torch.cat((torch.tensor([0]),cumsum))
    grid=grids_ada[cumsum[i]:cumsum[i+1]]
    new_vs=(grid.clone().detach()[:-1]+grid.clone().detach()[1:])/2.
    new_grid,idxs=torch.sort(torch.cat((grid,new_vs)))
    cumsum_new=torch.cumsum(len_grids_ada_new,0)
    cumsum_new=torch.cat((torch.tensor([0]),cumsum_new))
    grids_ada_new[cumsum_new[i]:cumsum_new[i+1]]=new_grid
for i in range(len(len_grids_uni)):
    cumsum=torch.cumsum(len_grids_uni,0)
    cumsum=torch.cat((torch.tensor([0]),cumsum))
    grid=grids_uni[cumsum[i]:cumsum[i+1]]
    new_vs=(grid.clone().detach()[:-1]+grid.clone().detach()[1:])/2.
    new_grid,idxs=torch.sort(torch.cat((grid,new_vs)))
    cumsum_new=torch.cumsum(len_grids_uni_new,0)
    cumsum_new=torch.cat((torch.tensor([0]),cumsum_new))
    grids_uni_new[cumsum_new[i]:cumsum_new[i+1]]=new_grid
len_grids_ada=len_grids_ada_new
len_grids_uni=len_grids_uni_new
grids_ada=grids_ada_new
grids_uni=grids_uni_new

sol=sols_ada[range(-len_grids_ada[-1],0),::]
sol_len=len(sol)
grid_fin=grids_ada[range(-len_grids_ada[-1],0)]
errs_ada=h1_error(sol,sols_ada,grid_fin,grids_ada,len_grids_ada)

#interpolate adaptive sol to uniform grid:
sol_uni=torch.zeros_like(sols_uni[range(-len_grids_uni[-1],0),::])
sol_len=len(sol_uni)
grid_fin_uni=grids_uni[range(-len_grids_uni[-1],0)]

k=0
sol_uni[0,::]=sol[0,::]
for j in range((sol_len-1)):

    t=grid_fin_uni[j+1]
    t0=grid_fin[2*k]
    y0=sol[2*k]
    thalf_c=grid_fin[2*k+1]
    yhalf=sol[2*k+1]
    t1=grid_fin[2*k+2]
    y1=sol[2*k+2] 
    while not (t>t0 and t<=t1):
        k+=1
        t0=grid_fin[2*k]
        y0=sol[2*k]
        thalf_c=grid_fin[2*k+1]
        yhalf=sol[2*k+1]
        t1=grid_fin[2*k+2]
        y1=sol[2*k+2]                   
    def interpolant(t):
        return y0*(t-thalf_c)*(t-t1)/((t0-thalf_c)*(t0-t1))+yhalf*(t-t0)*(t-t1)/((thalf_c-t0)*(thalf_c-t1))+y1*(t-t0)*(t-thalf_c)/((t1-t0)*(t1-thalf_c))
    sol_uni[j+1,::]=interpolant(t)
sol=sol_uni
errs_uni=h1_error(sol,sols_uni,grid_fin_uni,grids_uni,len_grids_uni)

plt.loglog(len_grids_ada.detach().numpy()/2., estis_ada.detach().numpy(), "+-", label=r"adaptive (estimator)")
plt.loglog(len_grids_uni.detach().numpy()/2., estis_uni.detach().numpy(), "x-", label=r"uniform (estimator)")
plt.loglog(len_grids_ada[:-1].detach().numpy()/2., errs_ada.detach().numpy(), "o-", label=r"adaptive (error)")
plt.loglog(len_grids_uni[:-1].detach().numpy()/2., errs_uni.detach().numpy(), "v-", label=r"uniform (error)")

# Reference line: 1/N
N = len_grids_uni.detach().numpy()/2.
plt.loglog(N, 100*1/N**2, "--", label=r"${\# \mathcal{T}}^{-2}$")

plt.xlabel(r"$\# \mathcal{T}$")


plt.legend()
plt.grid(True)
plt.savefig("latent_Lobatto_backward_real.eps")