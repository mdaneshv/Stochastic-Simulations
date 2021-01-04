import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import scipy.misc
import scipy.special
from matplotlib.lines import Line2D 
import seaborn as sns
from cycler import cycler
plt.style.use('seaborn')

    
def incremental_sampling(b, N, P0, d, num_sim):
    """
    b -- birth_rate
    N -- max population 
    P0 -- initial population 
    num_sim -- number of simulations
    d -- degree of population growth. 
    """    
    s = np.zeros((num_sim, N))  # Sojurn times
    X1 = np.zeros((num_sim, N)) # Population matrix
    inc = np.zeros((num_sim,N))
    X1[:,0] = P0
    for j in range(num_sim):
        for i in range(N-1):
            U = np.random.uniform(0,1)
            h = - np.log(U)/(b*X1[j,i]**d)  # Time incerements
            inc[j,i] = h
            s[j,i+1] = s[j,i] + h
            X1[j,i+1] = X1[j,i] + 1 
        return [s, X1, inc]
    
    
def increment_sampling_plots(b, N, P0, deg, num_sim):
    fig, ax = plt.subplots(2,3,figsize=(12,8))
    marker_style = dict(linestyle=':',marker='o', markersize=4) 
    col = ['cornflowerblue', 'orchid', 'orange']
    for i, d in enumerate(deg):
        [s, X1, inc] = incremental_sampling(b, N, P0, d, num_sim)
        ax[0,i].plot(s[0,:], X1[0,:], color=col[i], **marker_style)
        sns.distplot(inc, ax=ax[1,i], bins=np.linspace(0,0.1,100), kde=False)
        ax[0,i].set_title(r'$\nu_n \propto n^{%1.1f}$'%d)
        ax[1,i].set_title(r'distribution of increments: $\nu_n\propto'\
                          ' n^{%1.1f}$'%d, fontsize=10)
        ax[0,0].set_ylabel('population size',fontsize=12)
        ax[0,1].set_xlabel('time',fontsize=12)
        ax[1,i].set_xlim([0,0.03])
        plt.suptitle('Incremental sampling with different rates:\n'\
                     ' explosion occurs in a finite time when the power'\
                     ' of n exceeds 1', y=1.05)  
        plt.tight_layout(h_pad=8,w_pad=2)
        

def tau_leaping(b, tau, P0,
                num_steps, num_sim):
    """
    b -- birth_rate
    num_steps -- number of steps
    P0 -- initial population 
    num_sim -- number of simulations
    d -- degree of population growth. 
    """
    X2 = np.zeros((num_sim, num_steps))  # Population matrix
    X2[:,0] = P0
    for j in range(num_sim):
        for i in range(num_steps-1):
            r = np.random.poisson(lam = b*X2[j,i]*tau)
            X2[j,i+1] = X2[j,i] + r
        X2_aver = np.mean(X2, axis=0)
    return [X2, X2_aver]  


def exact_mean(b, P0, t):
    return P0*np.exp(b*t)
       
    
def tau_leaping_plots(b, tau, P0, num_steps, num_sim):
    fig, ax = plt.subplots(1, figsize=(10,4))
    col = ['cornflowerblue', 'orchid']
    [X2, X2_aver] = tau_leaping(b, tau, P0, num_steps, num_sim)
    t = np.linspace(0, tau*num_steps, num_steps)
    y = exact_mean(b, P0, t)
    ax.plot(t, X2_aver, color=col[0], label='tau_leaping')
    ax.plot(t, y, color=col[1], label='exact mean')
    ax.set_title('Comparison between tau-leaping simulations'\
                 ' and deterministic method for the linear growth', y=1.03) 
    ax.text(2,100, r"$\nu_n = 0.5\,n$", style = 'italic', fontsize=12)
    ax.legend(['mean from tau-leaping',' mean from deterministic method'])
    ax.set_xlabel('time', fontsize=12)
    ax.set_ylabel('population mean', fontsize=12)  
    plt.tight_layout()
    

def tau_leaping_hist(b, tau, P0,
                     num_steps, num_sim):
    
    fig, ax = plt.subplots()
    [X2, X2_aver] = tau_leaping(b, tau, P0, num_steps, num_sim)
    sns.distplot(X2[:,num_steps-1], label='tau_leaping', color='orchid')
    ax.legend(['population size: one realization'])
    ax.text(600,0.004, r"$\nu_n=0.5\,n$", style = 'italic' , size=12)
    ax.text(600,0.0035, r"$\tau=0.01,\,t =10$", style = 'italic' , size=12)
    ax.set_xlabel('population size',fontsize=12)
    ax.set_ylabel('KDE',fontsize=12)
    plt.title(r'Kernel density estimation for population size at time'\
                 r' $t=10$', va='center', ha='center',y=1.03) 
       

def trajectories(b, tau, P0,
                 num_steps, num_sim):
    
    fig, ax = plt.subplots(figsize=(6,5))
    col = ['cornflowerblue','orchid','orange','green']
    [X2, X2_aver] = tau_leaping(b, tau, P0, num_steps, num_sim)
    t = np.linspace(0, tau*num_steps, num_steps)
    for i in range(num_sim):
        ax.plot(t,X2[i,:], color=col[i])
        at = AnchoredText(r'$\nu_n=0.5\,n,\,\tau=0.01$',
                          loc='upper left', frameon=True)
        ax.add_artist(at)
        ax.set_xlabel('time',fontsize=12)
        ax.set_ylabel('population size',fontsize=12)
        plt.title(r'Different trajectories for population size'
                  , va='center', ha='center',y=1.03) 

        
# Plots for incremental sampling        
b = 0.5
N = 2000
P0 = 1
deg = [1, 1.5, 2]
num_sim = 1
increment_sampling_plots(b, N, P0, deg, num_sim)


# Plots for trajectories
b = 0.5
tau = 0.01
P0 = 1 
num_steps = 1000
num_sim = 4
trajectories(b, tau, P0, num_steps, num_sim)


# Histograms for tau_leaping method
b = 0.5
tau = 0.01
P0 = 1 
num_steps = 1000
num_sim = 1000
tau_leaping_hist(b, tau, P0, num_steps, num_sim)    


# Plots for tau_leaping method
b = 0.5
tau = 0.01
P0 = 1 
num_steps = 1000
num_sim = 1000
tau_leaping_plots(b, tau, P0, num_steps, num_sim)    
