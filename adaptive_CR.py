import numpy as np
import scipy as sci
import matplotlib.pyplot as plt


def ode_solve_adaptive(y_0,F,dtF,dyF,t_0,t_1,theta,tol,ninit):
    toterr=np.inf
    t0_or=t_0
    t1_or=t_1
    F_or=F
    dtF_or=dtF
    dyF_or=dyF
    if t_0>t_1:
        F= lambda t,y: -F_or(t_1-t+t_0,y)
        dtF= lambda t,y: dtF_or(t_1-t+t_0,y)
        dyF= lambda t,y: -dyF_or(t_1-t+t_0,y)
        t_1=t_0
        t_0=t1_or
    mesh=np.linspace(t_0,t_1,ninit)
    while(toterr>tol):
        sol=np.zeros((len(y_0),len(mesh)))
        sol[:,0]=y_0
        estis=np.zeros(len(mesh)-1)
        for i in range(len(mesh)-1):
            sol[:,i+1]=eval_CR(sol[:,i],mesh[i],mesh[i+1],F)
            estis[i]=eval_esti(dtF,dyF,mesh[i],mesh[i+1],sol[:,i],sol[:,i+1])
        toterr=np.sum(estis)
        if toterr>tol:
            mesh=refine_msh(mesh,estis,theta)
            print(len(mesh))
    if t0_or>t1_or:
        mesh=t0_or-mesh+t1_or
    return [sol,mesh]

def eval_esti(dtF,dyF,t_0,t_1,y_0,y_1):
    esti=sci.integrate.quad(lambda t: np.linalg.norm((dtF(t,y_0+(y_1-y_0)/(t_1-t_0)*(t-t_0))+np.matmul(dyF(t,y_0+(y_1-y_0)/(t_1-t_0)*(t-t_0)),(y_1-y_0)/(t_1-t_0))))**2,t_0,t_1)
    return ((t_1-t_0)**2)*abs(esti[0])

def refine_msh(mesh,estis,theta):
    sorted_idx=np.argsort(estis)[::-1]
    sorted_estis=estis[sorted_idx]
    cumsum=np.cumsum(sorted_estis)
    idx=np.argmax(cumsum>=theta*cumsum[-1])
    marked=sorted_idx[range(idx+1)]
    new_vs=(mesh[marked]+mesh[marked+1])/2
    new_mesh=np.sort(np.concatenate((mesh,new_vs)))
    return new_mesh

def eval_CR(y_0,t_0,t_1,F):
    def optim_f(y):
        return y-y_0-(t_1-t_0)*(F(t_1,y)+F(t_0,y_0))/2
    root=sci.optimize.fsolve(optim_f,y_0)
    return root


#test:
def F(t,y):
    return np.array([y[1],5*(1-y[0]**2)*y[1]-y[0]],ndmin=1)
def dtF(t,y):
    return np.zeros(2)
def dyF(t,y):
    return np.array([[0,1],[(-2)*5*y[0]*y[1]-1, (-5)*(y[0]**2-1)]])

[sol,mesh]=ode_solve_adaptive(np.array([1,1]),F,dtF,dyF,0,10,0.1,10**(-5),20)
plt.plot(mesh,sol[0,:])
plt.plot(mesh,sol[1,:])
plt.show()