# bayesian analysis 
# change point model for field hockey matches.


import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt 
import pymc as pm 
from collections import Counter


def  calculateWinnerLoser(hdata):
	score_max = np.max(hdata, axis=1)
	score_min = np.min(hdata, axis=1)
	sum_goals = np.sum(hdata, axis=1)
	diff_goals = score_max - score_min
	score_winner_sum = np.sum(score_max)
	score_loser_sum = np.sum(score_min)
	return score_winner_sum, score_loser_sum, sum_goals, diff_goals


count_data_2009 = np.genfromtxt("../data/hockey-data/2009ct.csv", delimiter=',', skip_header=0)
count_data_2010 = np.genfromtxt("../data/hockey-data/2010ct.csv", delimiter=',', skip_header=0)
count_data_2011 = np.genfromtxt("../data/hockey-data/2011ct.csv", delimiter=',', skip_header=0)
count_data_2012 = np.genfromtxt("../data/hockey-data/2012ct.csv", delimiter=',', skip_header=0)
count_data_2014 = np.genfromtxt("../data/hockey-data/2014ct.csv", delimiter=',', skip_header=0)
count_data_2016 = np.genfromtxt("../data/hockey-data/2016ct.csv", delimiter=',', skip_header=0)

# print count_data_2009
winner_sum=[]
loser_sum =[]
w9, l9, s9, d9 = calculateWinnerLoser(count_data_2009)
w10, l10, s10, d10 = calculateWinnerLoser(count_data_2010)
w11, l11, s11, d11 = calculateWinnerLoser(count_data_2011)
w12, l12, s12, d12 = calculateWinnerLoser(count_data_2012)
w14, l14, s14, d14 = calculateWinnerLoser(count_data_2014)
w16, l16, s16, d16 = calculateWinnerLoser(count_data_2016)

#  we have four different kind of features- difference between goals, sum of goals of each match.
w = [w9, w10, w11, w12, w14, w16]
l = [l9, l10, l11, l12, l14, l16]


#concatenate the sums/diffs for all the matches in all the series throughout the years.
 
total_sum = np.concatenate((s9, s10, s11, s12, s14, s16), axis=0)
total_diff = np.concatenate((d9, d10, d11, d12, d14, d16), axis=0)

n_count = len(w)
diff_list = [int(x-y) for x,y in zip(w,l)]
diff_list = map(int, diff_list)
sum_list = [x+y for x,y in zip(w,l)]

# diff_list = np.asarray([41., 47., 36., 38., 48., 31., 37., 23., 22., 20., 25., 24., 26.])
# n_count = len(diff_list)

n_count = len(total_sum)
n_count2 = len(total_diff)
# diff_list = diff_list.T 
# print diff_list.shape

# print w, l, diff_list, sum_list
print diff_list
alpha = sum(diff_list) / n_count
alpha2 = sum(total_sum) / n_count
alpha3 = sum(total_diff)/ n_count2

lambda_1 = pm.Exponential("lambda_1", 1. / alpha3)
lambda_2 = pm.Exponential("lambda_2", 1. / alpha3)

tau = pm.DiscreteUniform("tau", lower=0, upper=n_count2)


@pm.deterministic
def compute_lamda(tau=tau, lambda_1=lambda_1, lambda_2=lambda_2):
	out = np.zeros(n_count,)
	out[:tau] = lambda_1
	out[tau:] = lambda_2
	return out


# observation = pm.Poisson("obs", compute_lamda, value=diff_list, observed=True)
observation = pm.Poisson("obs", compute_lamda, value=total_diff, observed=True)
dgm = pm.Model([observation, tau, lambda_1, lambda_2])
mcmc_model = pm.MCMC(dgm)

mcmc_model.sample(200000, 10000, 50)
lambda1_samples = mcmc_model.trace('lambda_1')[:]
lambda2_samples = mcmc_model.trace('lambda_2')[:]
tau_samples = mcmc_model.trace('tau')[:]

plt.hist(tau_samples,  alpha=1,
         label= "posterior of $\tau$",
         color="#467821", rwidth=2.)


# plt.hist(lambda1_samples,  alpha=1,
#          label= "posterior of $\lambda1$",
#          color="red", rwidth=2.)


# plt.hist(lambda2_samples,  alpha=1,
#          label= "posterior of $\lambda1$",
#          color="blue", rwidth=2.)


plt.show()



