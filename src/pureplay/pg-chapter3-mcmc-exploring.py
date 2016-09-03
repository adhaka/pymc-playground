#  script for chapter 3 on mcmc exploring the mcmc landscape.

import numpy as np 
import pymc as pm 
import scipy.stats as stats
from matplotlib import pyplot as plt


# script begins here 
# load data 

LW =1
data = np.loadtxt("../data/mixture_data.csv", delimiter=',')
print data.shape

p = pm.Uniform('p', 0, 1)

assignment = pm.Categorical('assignment', [p, 1-p], size=data.shape[0])


taus = 1. / pm.Uniform("taus", 0, 100, size=2) ** 2
# tau1 = 1. / pm.Uniform("std1", 0, 100) ** 2
# tau2 = 1. / pm.Uniform("std2", 0, 100) ** 2
# print taus.shape

centers = pm.Normal("centers", [120, 190], [0.01, 0.01], size=2)

@pm.deterministic
def center_i(assignment=assignment, centers=centers):
	return centers[assignment]

@pm.deterministic
def tau_i(assignment=assignment, taus=taus):
	return taus[assignment]


observations = pm.Normal("obs", center_i, tau_i, value=data, observed=True)

model = pm.Model([p, assignment, observations, taus, centers])
mcmc = pm.MCMC(model)
mcmc.sample(60000)

plt.subplot(311)
colors = ["#348ABD", "#A60628"]
center_trace = mcmc.trace("centers")[:]
# plt.plot(center_trace[:,0], label="center1 trace", c=colors[0], lw=LW)
# plt.plot(center_trace[:,1], label="center2 trace", c=colors[1], lw=LW)

std_trace = mcmc.trace("taus")[:]


for i in range(2):
	plt.subplot(312)
	plt.hist(center_trace[:, i], color=colors[i], bins=30, histtype="stepfilled")
	plt.subplot(313)
	plt.hist(std_trace[:, i], color=colors[i], bins=30, histtype='stepfilled')

plt.show()


norm_pdf = stats.norm.pdf 
p_trace = mcmc.trace("p")[:]

x = 175

v = p_trace * norm_pdf(x, loc=center_trace[:,0], scale=std_trace[:, 0]) > 
	(1 - p_trace)*norm_pdf(x, loc=center_trace[:,1], scale=std_trace[:, 1]) 

print "Probability of point belonging to cluster1:", v.mean()

# plt.show()





