#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import matrix_power
from termcolor import colored
import timeit
plt.style.use('seaborn')

def transition_matrix(N, mu_a, mu_A, 
                      f, g):
     '''
    N -- population size
    mu_a, mu_A are mutation rates
    and f, g are fitnesses for 
    allele a and allele A respectively.
    '''
    P = np.zeros((N+1,N+1))   
    for j in range(1,N):
        sa = f*j/(f*j + g*(N-j))   # Prob of selecting an allele a
        sA = 1 - sa   # Prob of selecting an allele A
        P[j][j+1] = (sa*(1 - mu_a) + (sA*mu_A))*(N - j)/N   # Prob of birth for allele a
        P[j][j-1] = (sa*mu_a + sA*(1 - mu_A))*j/N   # Prob of death for allele a
        P[j][j] = 1 - P[j][j+1] - P[j][j-1]
    P[0][1] = mu_A    
    P[N][N-1] = mu_a  
    P[0][0] = 1 - P[0][1]
    P[N][N] = 1 - P[N][N-1]
    return P


def fixation_prob(N, f, g, x0, k):
    '''
    Returns the probability of 
    fixation at state N, starting at x0.
    '''
    P = transition_matrix(N, 0, 0, f, g)
    estimate_prob = matrix_power(P,k)[x0][N] 
    return estimate_prob


def time_to_fixation(N, f, g, x0):
    '''
    Returns mean time to 
    fixation at either state 0 or N,
    starting at x0.
    '''
    P = transition_matrix(N, 0, 0, f, g)
    Q = P[1:N,1:N]
    J = np.ones(N-1)
    I = np.identity(N-1)
    A = np.dot(np.linalg.inv(I-Q),J)
    mean_time = A[x0]
    return mean_time


def fixation_plots(pop_vec, pows, fit_vec, x0):
    fig, ax = plt.subplots(len(pop_vec), len(fit_vec), figsize=(12,8)) 
    col = ['cornflowerblue','orange','orchid']
    for i, (N,k) in enumerate(zip(pop_vec, pows)):
        for j, (f,g) in enumerate(fit_vec):
            mean_time = time_to_fixation(N, f, g, x0)
            trans_prob = []
            for iteration in range(k):
                p = fixation_prob(N, f, g, x0, iteration)
                trans_prob.append(p)
            ax[i,j].plot(np.linspace(1,k,k), trans_prob, color=col[i])
            ax[0,j].set_title('(f,g)={}'.format((f,g)), fontsize=12) 
            ax[-1,j].set_xlabel('steps(k)', fontsize=12)
            ax[i,0].set_ylabel('N={}'.format(N), fontsize=14)   
            ax[i,j].legend(['$Prob[X_k={}|x_0=1]$'.format(N)], fontsize=10)
            if i==0:
                ax[i,j].text(200,0.1,'Mean time to fixtaion\n= %1.1f' %mean_time, fontsize=9)
            elif i==1:
                ax[i,j].text(400,0.07,'Mean time to fixtaion\n= %1.1f' %mean_time, fontsize=9)
            else:
                ax[i,j].text(1000,0.03,'Mean time to fixtaion\n=  %1.1f' %mean_time, fontsize=9)
            ax[i,j].set_ylim([0,1/N + 0.01])
            ax[i,j].set_yticks(np.linspace(0,1/N,5))     
            plt.suptitle('First Plot: $Prob[X_{k}=N | x_{0}=1]$ versus'\
                         ' $k\,(steps)$ for different values of $N$',y=1.05) 
            plt.tight_layout()
            
            
def stationary_distribution(N, mu_a, mu_A,
                            f, g, k):
    P = transition_matrix(N, mu_a, mu_A, f, g)
    steady_state_approx = matrix_power(P, k)[0,:]
    return steady_state_approx  


def stationary_plots(N, mut_vec, pows, fit_vec):
    fig, ax = plt.subplots(len(mut_vec), len(fit_vec), figsize=(12,8)) 
    samples = 1000
    mutations = zip(mut_vec, pows)
    for i,((mu_a, mu_A), k) in enumerate(mutations):
        for j, (f,g) in enumerate(fit_vec):     
            P = transition_matrix(N, mu_a, mu_A, f, g)
            steady_state_approx = stationary_distribution(N, mu_a, mu_A,
                                                          f, g, k)
            state_frequency = steady_state_approx * samples
            mean_state = np.dot(state_frequency, np.arange(N+1))/samples
            ax[i,j].bar(np.arange(N+1), state_frequency)
            ax[i,j].axvline(x=mean_state, ymin=0, ymax=0.95, linestyle='dashed', color ='green', label='Mean')
            ax[0,j].set_title('(f,g)={}'.format((f,g)), fontsize=12)
            ax[-1,j].set_xlabel('states', fontsize=12)
            ax[i,0].set_ylabel('$\mu_a$=$\mu_A$={}'.format(mu_a), fontsize=12)  
            ax[i,j].legend()
            ax[i,j].set_ylim([0,max(state_frequency)+30])  
            plt.tight_layout()          
            plt.suptitle('Second Plot: Frequncy of states at stationary for N=20'\
                         ' in 1000 samples' ,y=1.05)  
            


start = timeit.default_timer()

# Plot the fixation probabilities
pop_vec = [5, 10, 20]
pows = [500, 1000, 2000]
fit_vec = [(1,1), (1,1.1), (1,1.2)]
x0 = 1
fixation_plots(pop_vec, pows, fit_vec, x0)


# Plot the stationary distributions
N = 20            
mut_vec = [(0.005,0.005), (0.05,0.05), (0.5,0.5)]
pows = [50000, 20000, 5000]
fit_vec = [(1,1), (1,1.1), (1,1.2)]
stationary_plots(N, mut_vec, pows, fit_vec)

# timing        
end = timeit.default_timer()
print('run time = %1.2f sec.'%(end-start))
