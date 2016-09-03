# script to calculate the chances of probability of a player scoring a goal- in american football using MCMC posterior
#  simple model given p and N, calculate the posterior of the true "chance", as p is the observed value.


import numpy as np
import scipy.stats as stats
import pymc as pm 
import matplotlib.pyplot as plt


football_data = np.genfromtxt("../data/football-averages.csv", delimiter=',', skip_header=1, usecols=[1,2], missing_values="NA")
player_names = np.genfromtxt("../data/football-averages.csv", delimiter=',', skip_header=1, usecols=[0], missing_values="NA")
n_players = football_data.shape[0]

# print football_data


def calculate_posterior_player(success, N, n_samples=20000):
	success_chance= pm.Uniform('success_chance', 0, 1)
	success = success
	print N
	print success
	observations =pm.Binomial('obs', p=success_chance, n=N, value=success, observed=True)
	# dgm = pm.Model([observations, p])
	mcmc= pm.MCMC([success_chance, observations])
	mcmc.sample(n_samples)
	print mcmc.trace('success_chance')[:]
	return mcmc.trace('success_chance')[:]


success_posterior= []
success_posterior_mean=[]
# rand_int = np.random.randint(5, size=football_data.shape[0])
for i in range(football_data.shape[0]):
	score = int(football_data[i,0]*football_data[i,1] / 100)
	posterior_values = calculate_posterior_player(score, football_data[i,1], n_samples=50000)
	success_posterior.append(posterior_values)
	success_posterior_mean.append(np.mean(posterior_values))

print success_posterior_mean


for i in range(1, n_players-3, 4) :
	plot_number = 200 + 20 + i%4
	plt.subplot(221)
	plt.hist(success_posterior[i], bins=20, histtype='stepfilled', alpha=0.85)
	plt.xlim(0,1)
	plt.subplot(222)
	plt.hist(success_posterior[i+1], bins=20, histtype='stepfilled', alpha=0.85)
	plt.xlim(0,1)
	plt.hist(success_posterior[i+2], bins=20, histtype='stepfilled', alpha=0.85)
	plt.xlim(0,1)
	plt.hist(success_posterior[i+3], bins=20, histtype='stepfilled', alpha=0.85)
	plt.xlim(0,1)




plt.show()


