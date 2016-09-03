# @author:akashkumar

import numpy as np 
import scipy.stats as stats
import pymc as pm 
import matplotlib.pyplot as plt 




class BayesianAB:
	''' class will take multiple candidates, which can have
	varying number of success and total trials and then find the 
	best candidate out of them on the basis of highest posterior mean.
	Prior could be chosen beforehand and we will have multiple priors 
	to pick on. '''

	def __init__(self, filename, skip_header=False, prior='Uniform'):
		self.filename = filename
		self.skip_header = skip_header
		self.data = None
		self.posterior_mean = []
		self.posterior_median = []
		self.posterior_low95 = []
		self.posterior_high95 = []
		self.posterior_samples = []
		# self.success_chance = None
		self.num_candidates = None
		self.skip_header = skip_header
		self.prior = prior


	def get_data(self):
		# do some sanity checks
		if self.data is None:
			self.data = np.genfromtxt(self.filename, delimiter=',', skip_header=self.skip_header, usecols=[1,2], missing_values="NA")
			self.num_candidates = self.data.shape[0]
		# if self.data[0,:]
		return self.data


	def _setPrior(self):
		# setting prior distribution for success_chance should be between 0 and 1.
		if self.prior == "Uniform":
			self.success_chance = pm.Uniform('success_chance', 0, 1)
		elif self.prior == "Beta":
			if use_eb is True:
				alpha, beta = self._empiricalBayes()

			self.success_chance = pm.Beta('success_chance', 1, 1)


# use empirical bayes to guess the hyper-parameters-  of beta prior.
	def _empiricalBayes(self):
		pass


	def predictOne(self, observed_avg, N, num_samples=50000, burn_number=20000):
		observations = None
		self._setPrior()
		print N, observed_avg
		# if observations:
		# 	del observations
		observed_avg = int(observed_avg * N / 100)
		observations = pm.Binomial('obs', p=self.success_chance, n=N, value=observed_avg, observed=True)
		model = pm.Model([observations, self.success_chance])
		mcmc = pm.MCMC(model)
		mcmc.sample(num_samples, burn_number)
		true_avg = mcmc.trace('success_chance')[:]
		print "true avergae:", true_avg
		return true_avg



	def predictAll(self):
		self.get_data()
		self._setPrior()
		for i in range(self.num_candidates):
			observed_avg, num_attempts = self.data[i,0], self.data[i,1]
			true_avg = self.predictOne(observed_avg, num_attempts)
			self.posterior_samples.append(true_avg)
			true_avg_mean = np.mean(true_avg)
			true_avg_median = np.median(true_avg)
			self.posterior_mean.append(true_avg_mean)
			self.posterior_median.append(true_avg_median)
		return self.posterior_median


# this should be used as a static method, offers plotting functionality
	@staticmethod
	def plotHistogram(posterior_dist):
		# handle the case when its a list of posterior distributions 
		if isinstance(posterior_dist, (list, tuple)):
			print 'hi'
		plt.hist(posterior_dist, bins =30, histtype='stepfilled', alpha=0.9)
		plt.xlim(0,1)


#  this function gets a list of scores.
	@staticmethod
	def sortScoresWithName(namelist, score):
		if len(score) != len(namelist):
			raise Exception('Number of names should be same as number of scores.')
		scoreMap = {}
		for x,y in zipped(namelist, score):
			scoreMap[x] = y

		scoreMap_sorted = sorted(scoreMap.items(), key=lambda x:x[1])
		print scoreMap_sorted

	@staticmethod
	def sortScoresWithOrder(score):
		scoreMap = {}
		for i in range(len(score)):
			scoreMap[i] = score[i]

		scoreMap_sorted = sorted(scoreMap.items(), key=lambda x:x[1])
		print scoreMap_sorted









