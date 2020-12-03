#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import quad
import seaborn as sns
from numba import jit
import time
plt.style.use('seaborn')


@jit
def Fokker_Planck_Equation(x, N, u, v, s):
    """ 
    Returns a function for stationary distribution of Moran process.
    """
    return np.power(x, N*v - 1)*np.power((1 - x), (N*u - 1))*np.exp(N*s*x)

@jit
def Fokker_Planck_Mean(x, N, u, v, s):
    """
    Returns the mean of Moran process at stationary using Fokker-Planck.
    """
    return x*np.power(x, N*v - 1)*np.power((1 - x), (N*u - 1))*np.exp(N*s*x)

@jit
def Fixation_Prob_Moran(N, w_a, w_A):
    """
    Returns an approximation for fixation probability for Moran process.
    """
    return ((w_A/w_a) - 1)/(np.power(w_A/w_a, N) - 1)

@jit
def Fixation_Prob_Fisher(N, s):
    """
    Returns an approximation for fixation probability for Wright-Fisher process.
    """
    return (1 - np.exp(-s))/(1 - np.exp(-N*s))


@jit
def Moran_Process(N, w_a, w_A, u, v,
                  num_a, num_A, max_steps,
                  dt, num_sim):
    """
    N -- population size,
    w_a, w_A -- weights,
    u, v -- mutation rates,
    dt -- time step.
    """
    step = 0
    a_counts = []
    while step < max_steps:
        prob_birth = ((w_a * num_a)*(1 - u) / (w_a * num_a + w_A * num_A) 
                      + (num_A * v) / (w_a * num_a + w_A * num_A)) * num_A / N
        prob_death = ((num_A)*(1 - v) / (w_a * num_a + w_A * num_A) 
                      + (w_a * num_a)*u / (w_a * num_a + w_A * num_A)) * num_a / N
        r = np.random.rand(1)
        if r <= prob_birth*dt:
            num_a += 1
            num_A -= 1
        elif r <= (prob_birth + prob_death)*dt:
            num_a -= 1
            num_A += 1
        step += 1    
        a_counts.append(num_a)
    return a_counts[-1]


# Plot the Fokker Planck distribution
population_sizes = [10, 100, 1000]
mutation_rates = [0.2, 0.02, 0.002]
s_list = [0.2, 0.02, 0.002]
w_A = 1
max_steps = 500000
dt = 1
num_sims = 1000
frequency = []
Fokker_Planck = []

for i in range(len(population_sizes)):
    x = np.linspace(0, 1, population_sizes[i])
    P = Fokker_Planck_Equation(x, population_sizes[i], mutation_rates[i],
                               mutation_rates[i], s_list[i])
    print("Population Size: ", population_sizes[i])
    simulations = [Moran_Process(population_sizes[i], 1+s_list[i], w_A, 
                                 mutation_rates[i], mutation_rates[i], 
                                 population_sizes[i]//2, 
                                 population_sizes[i]-(population_sizes[i]//2),
                                 max_steps, dt, j) for j in range(num_sims)]
    Fokker_Planck.append(P)
    a_counts = np.asarray(simulations)
    frequency.append(a_counts)
    

fig, ax = plt.subplots(1,len(population_sizes), figsize=(20,5))
fig.suptitle('Stationary Distribution For Different Population Sizes',
             fontsize=18, y=1.12)
fig.text(0.5, -0.02, 'Frequency of a', ha='center', fontsize=16)
fig.text(-0.02, 0.5, 'Density', va='center', rotation='vertical', fontsize=16)
for i in range(len(population_sizes)):
    x = np.linspace(0, 1, population_sizes[i])
    data = frequency[i]/population_sizes[i]
    FP = Fokker_Planck[i]
    ax[i].plot(x, 2*FP, label='1', color='r')
    sns.kdeplot(data, ax=ax[i], label='2', color='b')
    ax[i].axvline(np.mean(data), color='teal', linestyle='dashed',
                  linewidth=2, alpha=0.75)
    ax[i].legend(['Fokker Planck solution', 'Simulations'], fontsize=14)
    ax[i].set_title('N = {0}'.format(population_sizes[i]), fontsize=14)
    ax[i].set_xlim([-0.01,1.01])
    plt.tight_layout()
    
    
# Plot the fixation probabilities    
population_sizes = [10, 100, 1000, 10000]
mutation_rate = 0
s_list = [0.2, 0.02, 0.002, 0.0002]
w_A = 1
max_steps = 500000
dt = 1
num_sims = 1000
fix_prob_sim = []
fix_prob_Moran = []
fix_prob_wf = []

for i in range(len(population_sizes)):
    print("Population Size: ", population_sizes[i])
    f_M = Fixation_Prob_Moran(population_sizes[i], 1+s_list[i], w_A)
    f_WF = Fixation_Prob_Fisher(population_sizes[i], s_list[i])
    simulations = [Moran_Process(population_sizes[i], 1+s_list[i], w_A,
                                 mutation_rate, mutation_rate, 
                                 1, population_sizes[i] - 1, max_steps,
                                 dt, j) for j in range(num_sims)]
    fix_prob = simulations.count(population_sizes[i])/num_sims
    fix_prob_sim.append(fix_prob)
    fix_prob_Moran.append(f_M)
    fix_prob_wf.append(f_WF)

plt.scatter(population_sizes, fix_prob_sim, color='b', label='simulation')
plt.scatter(population_sizes, fix_prob_Moran, color='r', label='Moran-Analytical')
plt.scatter(population_sizes, fix_prob_wf, color='orange', label='WF-Analytical')
plt.xscale('Log')
plt.xlabel('population size')
plt.ylabel('fixation probability at N')
plt.ylim([-0.01,0.3])
plt.legend()
plt.show()

