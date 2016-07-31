#  script based on chapter 2 of pyMc.

import numpy as np 
import scipy.stats as stats 
import pymc as pm
import matplotlib.pyplot as plt

binomial = stats.binom


# 
bin_params = [(10, 0.4), (10, 0.85)]
colors= ['red', 'blue']

for i in xrange(len(bin_params)):
	N, p = bin_params[i]
	x = np.arange(N+1)
	plt.bar(x, binomial.pmf(x, N, p), alpha=0.6, color=colors[i])

plt.xlim(0,11)
plt.xlabel('$k$')
plt.show()


N = 100 
p = pm.Uniform('p', 0, 1)

cheating_studs = pm.Bernoulli("truths", p, size=N)
# sim_observations = pm.Bernoulli("sim_observations", )

@pm.deterministic
def p_skewed(p=p):
	p_combined = p*0.5 + 0.25
	return p_combined


true_cheats = pm.Binomial("num_cheaters", 100, p_skewed, observed=True, value=35)

# first argument-observations variable; second argument-posterior, third argument-prior variable.
dgm = pm.Model([true_cheats, p_skewed, p])

mcmc = pm.MCMC(dgm)
mcmc.sample(30000, 5000)
p_trace = mcmc.trace('truths')[:]

print p_trace.mean()
