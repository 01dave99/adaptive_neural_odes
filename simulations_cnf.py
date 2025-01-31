import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint2
import numpy as np
import matplotlib.pyplot as plt

class _TupleFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        f = self.base_func(t, _flat_to_shape(y, (), self.shapes))
        return torch.cat([f_.reshape(-1) for f_ in f])

def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tuple(tensor_list)

torch.set_default_dtype(torch.float64)
hidden_dim=32
width=64
max_n=10000


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()

class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]


class CNF(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hyper_net(t)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)



func = CNF(in_out_dim=2, hidden_dim=hidden_dim, width=width)

state=torch.load("ckpt_cnf.pth",weights_only=False)['func_state_dict']
func.load_state_dict(state)

z0=(torch.tensor([[-0.5502,  0.1435]]),torch.tensor([[0.]]))

sol11,sol12=odeint2(func,z0,torch.tensor([0,10],dtype=torch.float64),method="adaptiveLobatto",options={'step_size':0.1,'theta':0.7,'conv_ana':True,'file_id':"cnf_forward_Lobatto_ada",'max_nodes':max_n,'efficient':True})
odeint2(func,z0,torch.tensor([0,10],dtype=torch.float64),method="adaptiveLobatto",options={'step_size':0.1,'theta':1.,'conv_ana':True,'file_id':"cnf_forward_Lobatto_uni",'max_nodes':max_n,'efficient':True})

print("Forward pass finished.")
#backward-pass:

#adjoint:
adjoint_params=func.parameters()
adjoint_params=tuple(adjoint_params)

yend=(sol11[-1].detach().clone(),sol12[-1].detach().clone())


# Normalise to tensor (non-tupled) input
shapes = [y0_.shape for y0_ in yend]
yend = torch.cat([y0_.reshape(-1) for y0_ in yend])
func = _TupleFunc(func, shapes)

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



odeint2(augmented_dynamics,aug_state,torch.tensor([10,0],dtype=torch.float64),method="adaptiveLobatto",options={'step_size':0.1,'theta':0.7,'conv_ana':True,'faster_adj_solve':True,"adjoint_params":adjoint_params,"file_id":"cnf_backward_Lobatto_ada",'max_nodes':max_n,'original_func':func, 'shapes' :[y0_.shape for y0_ in aug_state],'efficient':True})
odeint2(augmented_dynamics,aug_state,torch.tensor([10,0],dtype=torch.float64),method="adaptiveLobatto",options={'step_size':0.1,'theta':1.,'conv_ana':True,'faster_adj_solve':True,"adjoint_params":adjoint_params,"file_id":"cnf_backward_Lobatto_uni",'max_nodes':max_n,'original_func':func, 'shapes' :[y0_.shape for y0_ in aug_state],'efficient':True})

print("Backward pass finished.")
