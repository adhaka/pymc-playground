# @author:akashkumar


from BayesianAB import BayesianAB




def main():
	ABTestingModel = BayesianAB(filename='../../data/football-averages.csv', skip_header=True)
	median_score = ABTestingModel.predictAll()
	# ABTestingModel.sortScoresWithName(median_score, )
	ABTestingModel.sortScoresWithOrder(median_score)


if __name__== '__main__':
	main()