# !usr/bin/env python

# __author__:"Akash"
# __copyright__:"MIT License"
# __license__:"GPL"
# __version__:"0.0.1"
# __email__:"akahsd@kth.se"


import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt 


dist = stats.beta

n_trials = []
x = np.linspace(0, 5, 100)

# variable for exponential distribution. 
expon = stats.expon

# E[exp disttribution] = 1/lambda
lambda1 = 0.72
lambda2 = 0.18

lambda_values = [lambda1, lambda2]
color_values = ['red', 'blue']

y1 = expon.pdf(x, scale=1/lambda1)
y2 = expon.pdf(x, scale= 1/lambda2)

for l,c in zip(lambda_values, color_values):
	plt.plot(x, expon.pdf(x, scale=1/l), color=c)
	plt.fill_between(x, expon.pdf(x, scale=1/l), color=c)


plt.ylim(0, 1.2)
plt.show()


