import torch
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint2
import numpy as np
import matplotlib.pyplot as plt

#Eulerank-Nicholson:

#training:
device = torch.device('cuda:' if torch.cuda.is_available() else 'cpu')
data_size=1000
batch_time=10
batch_size=10
niters=100
test_freq=20
max_n=50000

#true right-hand-side
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]],dtype=torch.float64)
true_y0 = torch.tensor([[2., 0.]],dtype=torch.float64).to(device)
t = torch.linspace(0., 25., data_size,dtype=torch.float64).to(device)
class Lambda(torch.nn.Module):
    def forward(self, t, y):
        return torch.mm(y**3,true_A)
    
#nn righ-hand side
class ODEFunc(torch.nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 50,dtype=torch.float64),
            torch.nn.Tanh(),
            torch.nn.Linear(50, 2,dtype=torch.float64),
        )

        for m in self.net.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=0.1)
                torch.nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)

#generate training-data
with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, atol=1e-2, method='adaptiveEuler',options={'step_size':0.01,'theta':0.7,'max_nodes':max_n,'efficient':True})
    print("Training data generated!")

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

#training:
ii = 0

func = ODEFunc().to(device)

optimizer = torch.optim.RMSprop(func.parameters(), lr=1e-3)

for itr in range(1, niters + 1):
    optimizer.zero_grad()
    batch_y0, batch_t, batch_y = get_batch()
    pred_y = odeint(func, batch_y0, batch_t , method='adaptiveEuler',atol=1e-2,faster_adj_solve=True,options={'step_size':0.01,'theta':0.7,'max_nodes':max_n,'efficient':True}).to(device)
    loss = torch.mean(torch.abs(pred_y - batch_y))
    loss.backward()
    optimizer.step()


    if itr % test_freq == 0:
        with torch.no_grad():
            pred_y = odeint(func, true_y0, t,atol=1e-2, method='adaptiveEuler',options={'step_size':0.01,'theta':0.7,'max_nodes':max_n,'efficient':True})
            loss = torch.mean(torch.abs(pred_y - true_y))
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
            #visualize(true_y, pred_y, func, ii)
            ii += 1

print("Training concluded!")

#convergence analysis:

#forward-pass:

#reference-solution:
#sol=odeint2(func,true_y0,torch.linspace(0,25,1000,dtype=torch.float64),method="dopri8")
#torch.save(sol,"results/ref_sol_forward_Euler.pt")

sol1=odeint2(func,true_y0,torch.linspace(0,25,1000,dtype=torch.float64),method="adaptiveEuler",options={'step_size':0.01,'theta':0.7,'conv_ana':True,'file_id':"forward_Euler_ada",'max_nodes':max_n,'efficient':True})
odeint2(func,true_y0,torch.linspace(0,25,1000,dtype=torch.float64),method="adaptiveEuler",options={'step_size':0.01,'theta':1.,'conv_ana':True,'file_id':"forward_Euler_uni",'max_nodes':max_n,'efficient':True})

print("Forward pass finished.")
#backward-pass:

#adjoint:

adjoint_params=func.parameters()
adjoint_params=tuple(adjoint_params)
yend=sol1[-1].detach().clone()
yend.requires_grad=True
loss1 = torch.mean(torch.abs(yend - true_y[-1]))
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
#reference solution:
#sol=odeint2(augmented_dynamics,aug_state,torch.linspace(25,0,1000,dtype=torch.float64),method="dopri8",atol=1e-12)

#flatten sol:
#sol = torch.cat([sol_.reshape((1000,-1)) for sol_ in sol],dim=1)
#torch.save(sol,"results/ref_sol_backward_Euler.pt")

odeint2(augmented_dynamics,aug_state,torch.linspace(25,0,1000,dtype=torch.float64),method="adaptiveEuler",options={'step_size':0.01,'theta':0.7,'conv_ana':True,'faster_adj_solve':True,"adjoint_params":adjoint_params,"file_id":"backward_Euler_ada",'max_nodes':max_n,'original_func':func, 'shapes' :[y0_.shape for y0_ in aug_state],'efficient':True})
odeint2(augmented_dynamics,aug_state,torch.linspace(25,0,1000,dtype=torch.float64),method="adaptiveEuler",options={'step_size':0.01,'theta':1.,'conv_ana':True,'faster_adj_solve':True,"adjoint_params":adjoint_params,"file_id":"backward_Euler_uni",'max_nodes':max_n,'original_func':func, 'shapes' :[y0_.shape for y0_ in aug_state],'efficient':True})

print("Backward pass finished.")

