import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint2
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
latent_dim = 4
nhidden = 20

z0 = torch.tensor([ 1.8088, -1.4681, -1.5551,  0.0151])
max_n=50000
class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out
    
func=LatentODEfunc(latent_dim,nhidden)

state=torch.load("ckpt.pth",weights_only=False)['func_state_dict']
func.load_state_dict(state)

sol1=odeint2(func,z0,torch.tensor([0,1],dtype=torch.float64),method="adaptiveLobatto",options={'step_size':0.1,'theta':0.7,'conv_ana':True,'file_id':"latent_forward_Lobatto_ada",'max_nodes':max_n})
odeint2(func,z0,torch.tensor([0,1],dtype=torch.float64),method="adaptiveLobatto",options={'step_size':0.1,'theta':1.,'conv_ana':True,'file_id':"latent_forward_Lobatto_uni",'max_nodes':max_n})

print("Forward pass finished.")
#backward-pass:

#adjoint:
adjoint_params=func.parameters()
adjoint_params=tuple(adjoint_params)
yend=sol1[-1].detach().clone()
yend.requires_grad=True
loss1 = torch.mean(torch.abs(yend))
grad_y=torch.autograd.grad(loss1,yend)[0]
aug_state = [torch.zeros((), dtype=yend.dtype, device=yend.device), yend.detach().requires_grad_(False), grad_y.requires_grad_(False)] # y, vjp_y
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
            allow_unused=True, retain_graph=True
        )

    # autograd.grad returns None if no gradient, set to zero.
    vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
    vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
    vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                    for param, vjp_param in zip(adjoint_params, vjp_params)]

    return (vjp_t, func_eval, vjp_y, *vjp_params)

#reference solution:



odeint2(augmented_dynamics,aug_state,torch.tensor([1,0],dtype=torch.float64),method="adaptiveLobatto",options={'step_size':0.1,'theta':0.7,'conv_ana':True,'faster_adj_solve':True,"adjoint_params":adjoint_params,"file_id":"latent_backward_Lobatto_ada",'max_nodes':max_n,'original_func':func, 'shapes' :[y0_.shape for y0_ in aug_state]})
odeint2(augmented_dynamics,aug_state,torch.tensor([1,0],dtype=torch.float64),method="adaptiveLobatto",options={'step_size':0.1,'theta':1.,'conv_ana':True,'faster_adj_solve':True,"adjoint_params":adjoint_params,"file_id":"latent_backward_Lobatto_uni",'max_nodes':max_n,'original_func':func, 'shapes' :[y0_.shape for y0_ in aug_state]})

print("Backward pass finished.")
