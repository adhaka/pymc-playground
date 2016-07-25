import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import pymc as pm
from collections import Counter 


count_data = np.loadtxt("../data/data1.csv")
n_count = len(count_data)
print n_count
avg_count = np.mean(count_data, axis=0)

alpha = 1. / avg_count
lambda_1 = pm.Exponential("lambda_1", alpha)
lambda_2 = pm.Exponential("lambda_2", alpha)

tau = pm.DiscreteUniform("tau", lower=0, upper=n_count)

@pm.deterministic
def comp_lambda(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
	out = np.zeros(n_count,)
	out2 = np.zeros(n_count,)
	out[:tau] = lambda_1
	out[tau:] = lambda_2
	# out2 = lambda_1 / lambda_2

	return out


observation = pm.Poisson("sim_obs", comp_lambda, value=count_data, observed=True)
dgm = pm.Model([observation, lambda_1, lambda_2, tau])
mcmc_model = pm.MCMC(dgm)

mcmc_model.sample(40000, 10000, 1)
lambda1_samples = mcmc_model.trace('lambda_1')[:]
lambda2_samples = mcmc_model.trace('lambda_2')[:]
tau_samples = mcmc_model.trace('tau')[:]

# plt.hist(lambda1_samples, histtype='stepfilled', bins=30, alpha=0.85,
#          label="posterior of $\lambda_1$", color='red', normed=True)
# plt.hist(lambda2_samples, histtype='stepfilled', bins=30, alpha=0.85,
#          label="posterior of $\lambda_1$", color="blue", normed=True)

plt.hist(tau_samples,  alpha=1,
         label= "posterior of $\tau$",
         color="#467821", rwidth=2.)

# plt.xticks(np.arange(avg_count))

# plt.show()
print Counter(tau_samples)
expected_messages = np.zeros(n_count,)
N_samples = tau_samples.shape[0]

for day in xrange(n_count):
	idx = day < tau_samples
	expected_messages[day] = (lambda1_samples[idx].sum() + lambda2_samples[~idx].sum()) / N_samples



lambda1_posterior_mean = lambda1_samples.mean()
lambda2_posterior_mean = lambda2_samples.mean()
ratio_posterior = [0] * len(lambda1_samples)

# for i in xrange(len(ratio_posterior)):
# 	ratio_posterior[i] = lambda1_samples[i] / lambda2_samples[i]

ratio_posterior = [ x/y for x,y in zip(lambda1_samples, lambda2_samples)]
ratio_posterior = np.asarray(ratio_posterior)


print "lambda 1 posterior mean:", lambda1_posterior_mean
print "lambda 2 posterior mean:", lambda2_posterior_mean
print "lamda1.mean/lambda2.mean:", lambda1_posterior_mean / lambda2_posterior_mean
print "lambda1/lambda2 mean:", ratio_posterior.mean()

#  if we know that tau is not 45, it can be only some thing less than 45; mean of lambda1, when tau is not 45.

idx2 = np.asarray(tau_samples < 45)
lambda1_posterior_mean_conditonal = lambda1_samples[idx2].mean()

print "lambda 1 posterior mean conditioned when tau < 45:", lambda1_posterior_mean_conditonal


