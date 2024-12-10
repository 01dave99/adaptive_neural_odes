import torch
from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as odeint2
import numpy as np
import matplotlib.pyplot as plt

method="adaptiveLobatto"
device = torch.device('cuda:' if torch.cuda.is_available() else 'cpu')
data_size=1000
batch_time=10
batch_size=100
niters=4000
test_freq=20
max_n=50000
loss_tensor=torch.zeros(niters//test_freq,dtype=torch.float64)

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
    true_y = odeint(Lambda(), true_y0, t, atol=1e-2, method=method,options={'step_size':0.1,'theta':0.7,'max_nodes':max_n})
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
    pred_y = odeint(func, batch_y0, batch_t , method=method,atol=1e-2,faster_adj_solve=True,options={'step_size':0.1,'theta':0.7}).to(device)
    loss = torch.mean(torch.abs(pred_y - batch_y))
    loss.backward()
    optimizer.step()


    if itr % test_freq == 0:
        with torch.no_grad():
            pred_y = odeint(func, true_y0, t,atol=1e-2, method=method,options={'step_size':0.1,'theta':0.7})
            loss = torch.mean(torch.abs(pred_y - true_y))
            print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
            #visualize(true_y, pred_y, func, ii)
            loss_tensor[ii]=loss
            ii += 1

print("Training concluded!")

torch.save(loss_tensor,"results/loss_Lobatto.pt")