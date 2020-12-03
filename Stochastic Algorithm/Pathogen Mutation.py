#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import random
from numba import jit
import time
from math import comb
from termcolor import colored
plt.style.use('seaborn')


# In[ ]:


def hamming_distance(list1, list2):
    
    dist = 0
    for n in range(len(list1)):
        if list1[n] != list2[n]:
            dist += 1
    return dist


# In[ ]:


def linear_mutation_model(population_size, h_thr, z,
                          str_leng, max_time):
    """
    It simulates the linear mutation model.
    h_thr -- Threshhold for immiunity system
    z -- Number of people exposed by each infected individual
    str_leng -- The length of the string representing the pathogene
    """
    
    indiv = {}   # Memory repertoire for individuals
    for i in range(population_size):
        indiv[i] = [0, []]
    
    for time in range(max_time):
        if time == 0:
            bits = [0 for i in range(str_leng)]   # Bitstrings for pathogens
            r0 = random.randint(0, population_size-1)
            indiv[r0] = [1, [bits]]
            
            prob = np.zeros(max_time)    # Fraction infected at time t
            prob[0] = 1 / population_size
        else: 
            bits = bits.copy()
            bits[time-1] = 1    # Linear mutation in time
            
            infected = [i for i in range(population_size) if indiv[i][0]==1]   # Infected individuals
            exposed = set()   
            for i in infected:
                excluded = list(np.r_[0:i, i+1:population_size])
                exposed = exposed.union(random.sample(excluded, z))   # Exposed indivuals
                  
            # Add the strain to individual's memory    
            for i in exposed:
                if indiv[i][1] == []:
                    indiv[i][0] = 1
                    indiv[i][1].append(bits)   
                else:
                    h_min = str_leng + 1
                    for genes in indiv[i][1]:
                        d = hamming_distance(genes, bits)
                        h_min = min(h_min, d)
                    if h_min > h_thr:
                        indiv[i][0] = 1
                        indiv[i][1].append(bits) 
                                    
            for i in infected:
                indiv[i][0] = 0   
     
            prob[time] = len([i for i in range(population_size) if indiv[i][0]==1])/population_size  
            
    return prob  


def linear_mutation_plots(population_size, hvec, zvec,
                          str_leng, max_time):
    
    fig, ax = plt.subplots(len(zvec), len(hvec), figsize=(15,10))
    t = np.linspace(0, max_time, max_time)
   
    for i, z in enumerate(zvec):
        for j, h_thr in enumerate(hvec):
            
            col = ['black','green','orchid','orange','cornflowerblue','red']
            prob = linear_mutation_model(population_size, h_thr,
                                         z, str_leng, max_time)
            
            ax[i,j].plot(t, prob, color=col[i], linewidth=1)
            ax[i,j].set_title(r'$h_{thr}$=%1.1f' %h_thr)
            ax[-1,j].set_xlabel('time', fontsize=14)
            ax[i,0].set_ylabel('z=%1.1f' %z, fontsize=14, fontweight='bold')
            fig.text(-0.02, 0.5, 'fraction infected ($p_t$)', va='center',
                     rotation='vertical', fontsize=18)
            plt.suptitle('Different behavior of $p_t$ with different parameters',                         y=1.05, color='purple', fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.savefig('p1.png')


# In[ ]:


def fraction_infected(tau, z, T):
    
    """
    It returns a difference equation for the linear model.
    tau -- time steps an individual remains recoverd before infected 
    z -- Number of people exposed by one infected individual
    T -- max time
    """
    
    p = np.zeros(T)  # Fraction infected
    p[tau] = 10**-4
    
    for t in range(tau, T-1): 
        p[t+1] = (1 - np.exp(-z*p[t]))*(1 - sum([p[t-i] for i in range(tau)]))
        
    return p


def plots(tauvec, zvec, T):
    
    fig, ax = plt.subplots(len(zvec), len(tauvec), figsize=(15,10))
    t = np.linspace(0, T, T)
   
    for i, z in enumerate(zvec):
        for j, tau in enumerate(tauvec):
            
            col = ['black','green','orchid','orange','cornflowerblue','red']
            p = fraction_infected(tau, z, T)
            
            ax[i,j].plot(t, p, color=col[i], linewidth=1)
            ax[i,j].set_title(r'$\tau$={}'.format(tau))
            ax[-1,j].set_xlabel('time', fontsize=14)
            ax[i,0].set_ylabel('z=%1.1f' %z, fontsize=14, fontweight='bold')
            fig.text(-0.02, 0.5, 'fraction infected ($p_t$)', va='center',
                     rotation='vertical', fontsize=18)
            plt.suptitle('Different behavior of $p_t$ with different parameters',                         y=1.05, color='purple', fontsize=18, fontweight='bold')
            plt.tight_layout()
            plt.savefig('p5.png')
            


# In[ ]:


"""Solve and plot"""

hvec = [4, 5, 6]
zvec = [1, 2, 3, 4] 
population_size = 10**4
max_time = 300
str_leng = 300
mutation_rate = 1

linear_mutation_plots(population_size, hvec, zvec,
                      str_leng, max_time)



T  = 400
tauvec = [4, 5, 6]
zvec = [1, 2, 3, 4] 

plots(tauvec, zvec, T)

