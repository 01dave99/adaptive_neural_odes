import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint2
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
state=torch.load("odefunc.pth",weights_only=True)
max_n=10000

def norm(dim):
    return torch.nn.GroupNorm(min(32, dim), dim)


class ConcatConv2d(torch.nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = torch.nn.ConvTranspose2d if transpose else torch.nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)

class ODEfunc(torch.nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out

downsampling_layers = [
            torch.nn.Conv2d(1, 64, 3, 1),
            norm(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 64, 4, 2, 1),
        ]
downsampling_layers = torch.nn.Sequential(*downsampling_layers)

func=ODEfunc(64)
func.load_state_dict(state)
transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

true_y0=downsampling_layers(datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test).__getitem__(0)[0].unsqueeze(0).to(torch.float64))

print(true_y0.shape)
sol1=odeint2(func,true_y0,torch.tensor([0,1],dtype=torch.float64),method="adaptiveEuler",options={'step_size':0.01,'theta':0.7,'conv_ana':True,'file_id':"mnist_forward_Euler_ada",'max_nodes':max_n,'efficient':True})
odeint2(func,true_y0,torch.tensor([0,1],dtype=torch.float64),method="adaptiveEuler",options={'step_size':0.01,'theta':1.,'conv_ana':True,'file_id':"mnist_forward_Euler_uni",'max_nodes':max_n,'efficient':True})

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



odeint2(augmented_dynamics,aug_state,torch.tensor([1,0],dtype=torch.float64),method="adaptiveEuler",options={'step_size':0.01,'theta':0.7,'conv_ana':True,'faster_adj_solve':True,"adjoint_params":adjoint_params,"file_id":"mnist_backward_Euler_ada",'max_nodes':max_n,'original_func':func, 'shapes' :[y0_.shape for y0_ in aug_state],'efficient':True})
odeint2(augmented_dynamics,aug_state,torch.tensor([1,0],dtype=torch.float64),method="adaptiveEuler",options={'step_size':0.01,'theta':1.,'conv_ana':True,'faster_adj_solve':True,"adjoint_params":adjoint_params,"file_id":"mnist_backward_Euler_uni",'max_nodes':max_n,'original_func':func, 'shapes' :[y0_.shape for y0_ in aug_state],'efficient':True})

print("Backward pass finished.")
