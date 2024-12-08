import torch
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint2
import numpy as np
import matplotlib.pyplot as plt

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

method="Lobatto"

class Func(torch.nn.Module):
    def __init__(self):
        super(Func, self).__init__()
        # Define parameters initialized to 0
        self.param1 = torch.nn.Parameter(torch.tensor(5,dtype=torch.float64))
        self.param2 = torch.nn.Parameter(torch.tensor(-5,dtype=torch.float64))

    def forward(self, t, y):
        f1 = self.param1* y[0]
        f2 = self.param2 * y[1]
        f = torch.stack([f1, f2])
        return f

func=Func()

adjoint_params=func.parameters()
adjoint_params=tuple(adjoint_params)
true_y0 = torch.tensor([0.1, 1.],dtype=torch.float64)
yend=true_y0
yend.requires_grad=True
loss1 = torch.mean(torch.abs(yend))
grad_y=torch.autograd.grad(loss1,yend)[0]
aug_state = [torch.zeros((), dtype=yend.dtype, device=yend.device), yend, grad_y] # y, vjp_y
aug_state.extend([torch.zeros_like(param) for param in adjoint_params])
aug_state=tuple(aug_state)

def augmented_dynamics(t, y_aug):
    # Dynamics of the original system augmented with
    # the adjoint wrt y, and an integrator wrt t and args.
    y = y_aug[1]
    adj_y = y_aug[2]
    # ignore gradients wrt time and parameters

    with torch.enable_grad():
        t_ = t.detach()
        t = t.requires_grad_(True)
        y = y.detach().requires_grad_(True)

        # If using an adaptive solver we don't want to waste time resolving dL/dt unless we need it (which
        # doesn't necessarily even exist if there is piecewise structure in time), so turning off gradients
        # wrt t here means we won't compute that if we don't need it.
        func_eval = func(t_, y)

        # Workaround for PyTorch bug #39784
        _t = torch.as_strided(t, (), ())  # noqa
        _y = torch.as_strided(y, (), ())  # noqa
        _params = tuple(torch.as_strided(param, (), ()) for param in adjoint_params)  # noqa

        vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
            func_eval, (t, y) + adjoint_params, -adj_y,
            allow_unused=True, retain_graph=True,create_graph=True
        )

    # autograd.grad returns None if no gradient, set to zero.
    vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
    vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
    vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                    for param, vjp_param in zip(adjoint_params, vjp_params)]

    return (vjp_t, func_eval, vjp_y, *vjp_params)

max_n=10000



#sol_ref=odeint2(func,true_y0,torch.linspace(0,10,1000,dtype=torch.float64))
#sol1=odeint2(func,true_y0,torch.linspace(0,1,1000,dtype=torch.float64),method="adaptiveLobatto",options={'step_size':0.01,'theta':0.7,'conv_ana':True,'file_id':"forward_Lobatto_ada_single","max_nodes":max_n})
print("Finished")
#sol2=odeint2(func,true_y0,torch.linspace(0,1,1000,dtype=torch.float64),method="adaptiveLobatto",options={'step_size':0.01,'theta':1.0,'conv_ana':True,'file_id':"forward_Lobatto_uni_single","max_nodes":max_n})
print("Finished")

#solend=sol1[-1]
#print(solend)

#sol_ref2=odeint2(func,solend,torch.linspace(10,0,10000,dtype=torch.float64),method="rk4",options={"step_size":0.001})
#odeint2(augmented_dynamics,aug_state,torch.linspace(1,0,1000,dtype=torch.float64),method="adaptiveLobatto",options={'step_size':0.01,'theta':0.7,'conv_ana':True,'faster_adj_solve':False,"adjoint_params":adjoint_params,"file_id":"backward_Lobatto_ada_single",'max_nodes':max_n,'original_func':func, 'shapes' :[y0_.shape for y0_ in aug_state]})
#odeint2(augmented_dynamics,aug_state,torch.linspace(1,0,1000,dtype=torch.float64),method="adaptiveLobatto",options={'step_size':0.01,'theta':1.,'conv_ana':True,'faster_adj_solve':False,"adjoint_params":adjoint_params,"file_id":"backward_Lobatto_uni_single",'max_nodes':max_n,'original_func':func, 'shapes' :[y0_.shape for y0_ in aug_state]})

#forward pass:

#sol=torch.load("results/ref_sol_forward_Lobatto.pt")
sols_ada=torch.load("results/sols_forward_Lobatto_ada_single.pt")
sols_uni=torch.load("results/sols_forward_Lobatto_uni_single.pt")

len_grids_ada=torch.load("results/len_grids_forward_Lobatto_ada_single.pt")
len_grids_uni=torch.load("results/len_grids_forward_Lobatto_uni_single.pt")

estis_ada=torch.sqrt(torch.load("results/estis_forward_Lobatto_ada_single.pt"))
estis_uni=torch.sqrt(torch.load("results/estis_forward_Lobatto_uni_single.pt"))

grids_ada=torch.load("results/grids_forward_Lobatto_ada_single.pt")[:-len_grids_ada[-1]]
grids_uni=torch.load("results/grids_forward_Lobatto_uni_single.pt")[:-len_grids_uni[-1]]

len_grids_ada=len_grids_ada[:-1]
len_grids_uni=len_grids_uni[:-1]

#calc H1 norm:
if method=="Lobatto":
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

if method=="Lobatto":
    k=0
    for j in range((sol_len-1)//2):
        
        
        thalf=grid_fin_uni[2*j+1]
        t=grid_fin_uni[2*j+2]
        t0=grid_fin[2*k]
        y0=sol[2*k]
        thalf_c=grid_fin[2*k+1]
        yhalf=sol[2*k+1]
        t1=grid_fin[2*k+2]
        y1=sol[2*k+2] 
        while not (thalf>t0 and t<=t1):
            k+=1
            thalf=grid_fin_uni[2*j+1]
            t=grid_fin_uni[2*j+2]
            t0=grid_fin[2*k]
            y0=sol[2*k]
            thalf_c=grid_fin[2*k+1]
            yhalf=sol[2*k+1]
            t1=grid_fin[2*k+2]
            y1=sol[2*k+2]                   
        def interpolant(t):
            return y0*(t-thalf_c)*(t-t1)/((t0-thalf_c)*(t0-t1))+yhalf*(t-t0)*(t-t1)/((thalf_c-t0)*(thalf_c-t1))+y1*(t-t0)*(t-thalf_c)/((t1-t0)*(t1-thalf_c))
        sol_uni[2*j+1,::]=interpolant(thalf)
        sol_uni[2*j+2,::]=interpolant(t)

else:
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


plt.loglog(len_grids_ada.detach().numpy(),estis_ada.detach().numpy(),"+-")
plt.loglog(len_grids_uni.detach().numpy(),estis_uni.detach().numpy(),"x-")
plt.loglog(len_grids_ada[:-1].detach().numpy(),errs_ada.detach().numpy(),"o-")
plt.loglog(len_grids_uni[:-1].detach().numpy(),errs_uni.detach().numpy(),"v-")
plt.show()

#backward pass:

#sol=torch.load("results/ref_sol_backward_Lobatto.pt")
sols_ada=torch.load("results/sols_backward_Lobatto_ada_single.pt")
sols_uni=torch.load("results/sols_backward_Lobatto_uni_single.pt")

len_grids_ada=torch.load("results/len_grids_backward_Lobatto_ada_single.pt")
len_grids_uni=torch.load("results/len_grids_backward_Lobatto_uni_single.pt")

estis_ada=torch.sqrt(torch.load("results/estis_backward_Lobatto_ada_single.pt"))
estis_uni=torch.sqrt(torch.load("results/estis_backward_Lobatto_uni_single.pt"))

grids_ada=torch.load("results/grids_backward_Lobatto_ada_single.pt")[:-len_grids_ada[-1]]
grids_uni=torch.load("results/grids_backward_Lobatto_uni_single.pt")[:-len_grids_uni[-1]]

len_grids_ada=len_grids_ada[:-1]
len_grids_uni=len_grids_uni[:-1]

#calc H1 norm:
if method=="Lobatto":
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

if method=="Lobatto":
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

else:
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


plt.loglog(len_grids_ada.detach().numpy(),estis_ada.detach().numpy(),"+-")
plt.loglog(len_grids_uni.detach().numpy(),estis_uni.detach().numpy(),"x-")
plt.loglog(len_grids_ada[:-1].detach().numpy(),errs_ada.detach().numpy(),"o-")
plt.loglog(len_grids_uni[:-1].detach().numpy(),errs_uni.detach().numpy(),"v-")
plt.show()

