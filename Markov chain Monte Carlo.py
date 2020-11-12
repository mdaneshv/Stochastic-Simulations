#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import t,norm,uniform
from matplotlib.offsetbox import AnchoredText
plt.style.use('ggplot')


def Markov_Monte_Carlo(x0, delta, num_samples):
    """
    Returns samples from t-distribution
    using Metropolis-Hastings algorithim
    and an uniform distribution.
    """
    X = np.zeros(num_samples)  # Samples
    X[0] = x0
    for i in range(1,num_samples):
        r = np.random.uniform(-delta,delta)
        y = X[i-1] + r
        acc_ratio = min(np.log(t.pdf(y,2))-np.log(t.pdf(X[i-1],2)), 0)  # Acceptance ratio
        u = np.random.uniform(0,1)
        if np.log(u) <= acc_ratio:
            X[i] = y
        else:
            X[i] = X[i-1] 
    return X


def MCMC_Plots(x0, delta_vec, num_samples):
    half = len(delta_vec)//2
    fig1 = plt.figure(figsize=(16,6), constrained_layout=True)
    gs1 = fig1.add_gridspec(2,3)
    ax = {}
    for i, delta in enumerate(delta_vec[:half+1]):  
        ax[i] = fig1.add_subplot(gs1[0, i]) 
        samples = Markov_Monte_Carlo(x0, delta, num_samples)
        x = np.linspace(-8,8,num_samples)
        z = t.pdf(x,2)
        ax[i].plot(x, z, label='true density', color='cornflowerblue')
        sns.kdeplot(samples, label='MCMC simulation', color='orchid', ax=ax[i])
        ax[i].legend(fontsize=8)
        at = AnchoredText('$\delta$={}'.format(delta), loc='center left', frameon=True)
        ax[i].add_artist(at)
        ax[0].set_ylabel('Density')
        ax[i].set_xlim([-8,8])
    for i, delta in enumerate(delta_vec[half+1:]):  
        ax[i] = fig1.add_subplot(gs1[1, i]) 
        samples = Markov_Monte_Carlo(x0, delta, num_samples)
        x = np.linspace(-8,8,num_samples)
        z = t.pdf(x,2)
        ax[i].plot(x, z, label='true density', color='cornflowerblue')
        sns.kdeplot(samples, label='MCMC simulation', color='orchid', ax=ax[i])
        ax[i].legend(fontsize=8)
        at = AnchoredText('$\delta$={}'.format(delta), loc='center left', frameon=True)
        ax[i].add_artist(at)
        ax[0].set_ylabel('Density') 
        ax[i].set_xlim([-8,8])
        
        
def Quotient(n, mu, sigma, c):
    return t.pdf(n,2)/(c*norm.pdf(n, mu, sigma))


def Supremum(mu, sigma):
    x = np.linspace(-10,10,100)
    q = norm.pdf(x, mu, sigma) / t.pdf(x,2)
    c = max(q)
    return c 


def Rejection_Sampling1(mu, sigma, num_sim):
    """
    Returns samples of a t-distribution using
    rejection sampling method and a fixed 
    prob of acceptance. Candidate density is
    a normal distribution.
    """
    c = Supremum(mu, sigma)
    sample1 = []
    for i in range(num_sim):  
        u = np.random.uniform(0,1)    
        n = np.random.normal(mu, sigma)    
        if u <= Quotient(n, mu, sigma, c):
            sample1.append(n) 
    return [c, sample1]


def Rejection_Sampling2(mu, sigma, c0, num_sim):
    """
    An alternative method. Prob of acceptance
    is updated during the simulations.
    """
    
    c_list = []
    sample2 = []
    c = c0   
    for i in range(num_sim):   
        u = np.random.uniform(0,1)    
        n = np.random.normal(mu, sigma)     
        if u <= Quotient(n, mu, sigma, c):  
            sample2.append(n) 
            c = max(c, norm.pdf(n, mu, sigma)/t.pdf(n,2))  
            c_list.append(c)       
    return [c_list, sample2]  


def Rejection_Comparison_Plots(mu, var_vec, c0, num_sim):
    
     """It compares both methods for rejection sampling."""
        
    fig, ax = plt.subplots(len(var_vec),2, figsize=(12,10))
    x = np.linspace(-8,8,num_sim)
    for i, v in enumerate(var_vec):
        [c, sample1] = Rejection_Sampling1(mu, np.sqrt(v), num_sim)
        [c_list, sample2] = Rejection_Sampling2(mu, np.sqrt(v), c0, num_sim)
        ax[i,0].plot(x, t.pdf(x,2), label='true density', color='black')
        sns.kdeplot(sample1, label='first approach', color='cornflowerblue', ax=ax[i,0])
        sns.kdeplot(sample2, label='alternative method', color='orchid', ax=ax[i,0])
        ax[0,0].set_title('KDE plots')
        ax[i,0].legend(fontsize=10)
        ax[-1,0].set_xlabel('x', fontsize=14)
        ax[i,0].set_xlim([-8,8])
        ax[i,0].text(-6,0.2,'Candidate density:\n$\mu=0,\;\sigma^2$={}'.format(v))
        ax[i,1].axhline(y=c, label='4', color='r', linestyle='--', alpha=0.6)
        ax[i,1].plot(np.arange(len(c_list)), c_list, label='5', color='b', alpha=0.6)
        ax[0,1].set_title('Plots for $c$', fontsize=12)
        ax[i,1].legend(['$c$ in the first approach','$c$ in the second approach'], fontsize=10)
        ax[-1,1].set_xlabel('steps', fontsize=14)
        ax[i,1].set_xlim(0,50)
        plt.suptitle('Rejection sampling method with fixed $c$ and its alternative', y=1.05)
        plt.tight_layout()
        
        
def MCMC_Rejection_Plots(x0, delta, mu, v, s:list):
    
    """It compares Monte-Carlo Narkov chain with rejection sampling method."""
    
    fig, ax = plt.subplots(len(s), figsize=(10,10))
    for i, num_samples in enumerate(s):
        first_mathod = Markov_Monte_Carlo(x0, delta, num_samples)
        [c, second_method] = Rejection_Sampling1(mu, np.sqrt(v), num_samples)
        x = np.linspace(-8,8,num_samples) 
        ax[i].plot(x, t.pdf(x,2), label='1', color='black')
        sns.kdeplot(first_mathod, label='1', color='cornflowerblue', ax=ax[i])
        sns.kdeplot(second_method, label='2', color='orchid', ax=ax[i])
        ax[i].legend(['true density','MCMC','rejection sampling'], fontsize=10)
        ax[i].text(-6,0.25,'number of simulations={}'.format(num_samples), fontsize=12)
        ax[i].set_ylabel('KDE', fontsize=14)
        ax[i].set_xlim([-8,8])
        plt.suptitle('Comparison between MCMC and rejection sampling,\n for MCMC:'                    r' $\delta=1$ and for rejection sampling: $\mu=0,\,\sigma^2=3$'                     ' and $c\approx1.25$',y=1.05)
        plt.tight_layout()  
      
    
# An Example
x0 = 0
delta = 1  
mu = 0   
v = 3
s = [5000, 10000, 50000]

MCMC_Rejection_Plots(x0, delta, mu, v, s)        
