# script based on chapter 2 of pyMC

import numpy as np 
import scipy.stats as stats
import pymc as pm 
import matplotlib.pyplot as plt


def logistic(x, beta, alpha):
	p_new =  1. / (1. + np.exp(alpha + np.dot(beta, x)))
	return p_new


normal = stats.norm
x = np.linspace(-8, 7, 150)

challenger_data = np.genfromtxt('../data/challenger-data.csv', skip_header=1, usecols=[1, 2], missing_values="NA", delimiter=',')

challenger_data = challenger_data[~np.isnan(challenger_data[:,1])]
print challenger_data


temperature = challenger_data[:,0]
failure = challenger_data[:,1]

# first random variable - a gaussian 
beta = pm.Normal("beta", 0, 0.001, value=0)

alpha = pm.Normal("alpha", 0, 0.001, value=0)


@pm.deterministic
def prob(t=temperature, alpha=alpha, beta=beta):
	prob = 1.0 / (1 + np.exp(alpha + beta*temperature))
	return prob

observed = pm.Bernoulli("bernoulli_obs", prob, observed=True, value=failure)

model = pm.Model([observed, beta, alpha])
map_inference = pm.MAP(model)
map_inference.fit()

mcmc = pm.MCMC(model)
mcmc.sample(140000, 60000, 2)

alpha_samples = mcmc.trace('alpha')[:, np.newaxis]
beta_samples = mcmc.trace('beta')[:, np.newaxis]

print alpha_samples.shape
print alpha_samples.mean()

plt.subplot(211)
plt.hist(beta_samples, histtype='stepfilled', color='red', alpha=0.85, normed=True)
plt.xlim(0,1)
plt.legend()

plt.subplot(212)
plt.hist(alpha_samples, histtype='stepfilled', color='blue', alpha=0.71, normed=True)
plt.xlim((-35,0))
# plt.show()

t1 = np.linspace(temperature.min()-5, temperature.max()+5, 100)[:,np.newaxis]
p_new = logistic(t1.T, beta_samples, alpha_samples)

mean_prob_t1 = p_new.mean(axis=0)

plt.plot(t1, mean_prob_t1)
plt.plot(t1, p_new[0, :], ls="--", label="realization from posterior")
plt.plot(t1, p_new[-2, :], ls="--", label="realization from posterior")
plt.xlim(temperature.min(), temperature.max())
plt.ylim(-0.1, 1.2)
plt.ylabel('Probability')
plt.xlabel('Temperature')
plt.show()

