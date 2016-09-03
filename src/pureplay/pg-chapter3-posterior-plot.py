# script for chapter 3 on MCMC sampling. 

import numpy as np 
import matplotlib.pyplot as plt 
import pymc as pm 
import scipy.stats as stats 

from mpl_toolkits.mplot3d import Axes3D
jet = plt.cm.jet
fig = plt.figure()

N = 2

lambda1_true = 2
lambda2_true = 5

sim_data = np.concatenate((stats.poisson.rvs(lambda1_true, size=(N, 1)), stats.poisson.rvs(lambda2_true, size=(N,1))), axis=1)

x = np.linspace(0.01, 6, 120)
y = np.linspace(0.01, 6, 120)
X, Y = np.meshgrid(x, y)

likelihood_x = np.array([stats.poisson.pmf(sim_data[:,0], i) for i in x]).prod(axis=1)
likelihood_y = np.array([stats.poisson.pmf(sim_data[:,1], j) for j in y]).prod(axis=1)

print likelihood_x.shape
print likelihood_y.shape

L = np.dot(likelihood_x[:,np.newaxis], likelihood_y[np.newaxis,:])

uniform_x = stats.uniform.pdf(x, loc=0, scale=5)
uniform_y = stats.uniform.pdf(y, loc=0, scale=5)

Prior = np.dot(uniform_x[:, np.newaxis], uniform_y[np.newaxis, :])
# im = plt.imshow(Prior, interpolation='none', origin='lower',
#                 cmap=jet, vmax=1, vmin=-.15, extent=(0, 6, 0, 6))
plt.xlim(0, 6)
plt.ylim(0, 6)

# ax = fig.add_subplot(122, projection='3d')

# ax.plot_surface(X, Y, Prior, cmap=plt.cm.jet, vmax=1, vmin=-.15)
# plt.show()



plt.subplot(222)
exp_A = stats.expon.pdf(x, loc=0, scale=3)
exp_B = stats.expon.pdf(y, loc=0, scale=10)


Prior_Exp = np.dot(exp_A[:, np.newaxis], uniform_y[np.newaxis, :])
plt.contour(x, y, Prior_Exp)

im = plt.imshow(Prior_Exp, interpolation='none', origin='lower', cmap=jet, extent=(0, 6, 0, 6))
plt.scatter(lambda2_true, lambda1_true, c="k", s=50, edgecolor="none") 

plt.subplot(223)
Posterior = Prior_Exp * L

plt.contour(x, y, Posterior)
im2 = plt.imshow(Posterior, interpolation='none', origin='lower', cmap=jet, extent=(0, 6, 0, 6))

plt.scatter(lambda2_true, lambda1_true, c='k', s=50, edgecolor='none')
# show plot
plt.xlim(0, 5)
plt.xlim(0, 5)

plt.show()
