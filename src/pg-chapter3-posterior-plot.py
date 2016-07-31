# script for chapter 3 on MCMC sampling. 

import numpy as np 
import matplotlib.pyplot as plt 
import pymc as pm 
import scipy.stats as stats 


N = 20

lambda1_true = 2
lambda2_true = 5

sim_data = np.concatenate(stats.poisson.rvs(lambda1_true, size=(N, 1)), stats.poisson.rvs(lambda2_true, size=(N,1)), axis=1)

x = np.linspace(0.01, 6, 120)
y = np.linspace(0.01, 6, 120)

likelihood_x = np.array([stats.poisson.pmf(data[:,0], i) for i in x]).prod(axis=1)
likelihood_y = np.array([stats.poisson.pmf(data[:,1], j) for j in y]).prod(axis=1)

L = np.dot(likelihood_x[:,np.newaxis], likelihood_y[np.newaxis,:])

uniform_x = stats.uniform.pdf(x, loc=0, scale=5)
uniform_y = stats.uniform.pdf(y, loc=0, scale=5)

Prior = np.dot(uniform_x[:, np.newaxis], uniform_y[np.newaxis, :])
im = plt.imshow(M, interpolation='none', origin='lower', cmap=jet, extent=(0, 6, 0, 6))
plt.xlim(0, 6)
plt.ylim(0, 6)

plt.contour(L*Prior)
plt.show()



exp_A = stats.exponential.pdf(x, loc=0, scale=3)
exp_B = stats.exponential.pdf(y, loc=0, scale=10)


Prior_Exp = np.dot(exp_A[:, np.newaxis], uniform_y[np.newaxis, :])


