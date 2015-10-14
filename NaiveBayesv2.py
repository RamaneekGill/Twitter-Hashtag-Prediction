from __future__ import division
import csv
import random
import math
import sys
import string
import re
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.sparse import *
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy import ndimage
from pandas import *
import numpy
from numpy import *
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook


def main():
	CONST_EPSILON_INTERVALS = [0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001]
	CONST_ALPHA_INTERVALS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1]
	CONST_HASHTAGS_TO_PREDICT = [56, 100, 150, 200, 250, 300]
	CONST_HASHTAG_PREDICTION_RANGE = [20, 15, 10, 5, 3, 1]
	CONST_TEST_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	BEST_ALPHA = 0.92
	BEST_EPSILON = 0.01

	# For 500 hashtags and predicting top 5
	BEST_ALPHA = 0.9
	BEST_EPSILON = 1e-09

	naiveBayes = NaiveBayes()
	naiveBayes.setEpsilon(BEST_EPSILON)
	naiveBayes.setAlpha(BEST_ALPHA)
	naiveBayes.testAgainst(naiveBayes.testSet)
	print('CORRECT PREDICTIONS~~~~~~~~~~~~~~~~~~~~~~~~~')
	naiveBayes.getProbabilityResults(naiveBayes.correctPredictions)
	print('\n\n\n\n\n\n\n\n\n\n\n')
	print('INCORRECT PREDICTIONS~~~~~~~~~~~~~~~~~~~~~~~~~')
	naiveBayes.getProbabilityResults(naiveBayes.incorrectPredictions)

	# print('Performing cross validation to find the best epsilon and alpha values')
	# accuracies = []
	# maxAccuracy = 0
	# for epsilon in CONST_EPSILON_INTERVALS:
	# 	for alpha in CONST_ALPHA_INTERVALS:
	# 		naiveBayes.setEpsilon(epsilon)
	# 		naiveBayes.setAlpha(alpha)
	# 		naiveBayes.testAgainst(naiveBayes.validationSet)
	# 		accuracy = naiveBayes.getAccuracy()
	# 		accuracies.append(accuracy)
	# 		if max(accuracies) > maxAccuracy:
	# 			BEST_EPSILON = epsilon
	# 			BEST_ALPHA = alpha
	# 			maxAccuracy = max(accuracies)
	# print('Validation tests have shown that the best epsilon value to use is: {}, best alpha value is: {}'.format(BEST_EPSILON, BEST_ALPHA))

	# print('Generating graph for varying number of hashtags predicted contain target')
	# accuracies = []
	# for hitRange in CONST_HASHTAG_PREDICTION_RANGE:
	# 	naiveBayes.setHitRange(hitRange)
	# 	naiveBayes.testAgainst(naiveBayes.testSet)
	# 	accuracy = naiveBayes.getAccuracy()
	# 	accuracies.append(accuracy)
	# plt.plot(CONST_HASHTAG_PREDICTION_RANGE, accuracies)
	# plt.xlabel('Number Of Hashtags Predicted')
	# plt.ylabel('Accuracy')
	# plt.title('Accuracy when Varying Number of Hashtags Predicted Contain Target Hashtag')
	# plt.show()

	# print('Generating graph for varying number of hashtags to predict')
	# accuracies = []
	# for numHashtags in CONST_HASHTAGS_TO_PREDICT:
	# 	naiveBayes.setHashtagsToPredict(numHashtags)
	# 	naiveBayes.testAgainst(naiveBayes.testSet)
	# 	accuracy = naiveBayes.getAccuracy()
	# 	accuracies.append(accuracy)
	# plt.plot(CONST_HASHTAGS_TO_PREDICT, accuracies)
	# plt.xlabel('Number Of Hashtags To Predict')
	# plt.ylabel('Accuracy')
	# plt.title('Accuracy when Varying Number of Hashtags to Predict')
	# plt.show()

	# print('Generating graph for epsilon accuracies')
	# epsilonAccuracies = []
	# alpha = BEST_ALPHA
	# for epsilon in CONST_EPSILON_INTERVALS:
	# 	naiveBayes.setEpsilon(epsilon)
	# 	naiveBayes.setAlpha(alpha)
	# 	naiveBayes.testAgainst(naiveBayes.testSet)
	# 	accuracy = naiveBayes.getAccuracy()
	# 	epsilonAccuracies.append(accuracy)
	# plt.plot(CONST_EPSILON_INTERVALS, epsilonAccuracies)
	# plt.xlabel('Epsilon')
	# plt.ylabel('Accuracy')
	# plt.title('Accuracy on Test Set Using Alpha = {}'.format(alpha))
	# plt.show()

	# print('Generating graph for alpha accuracies')
	# alphaAccuracies = []
	# epsilon = BEST_EPSILON
	# for alpha in CONST_ALPHA_INTERVALS:
	# 	naiveBayes.setEpsilon(epsilon)
	# 	naiveBayes.setAlpha(alpha)
	# 	naiveBayes.testAgainst(naiveBayes.testSet)
	# 	accuracy = naiveBayes.getAccuracy()
	# 	alphaAccuracies.append(accuracy)
	# plt.plot(CONST_ALPHA_INTERVALS, alphaAccuracies)
	# plt.xlabel('Alpha')
	# plt.ylabel('Accuracy')
	# plt.title('Accuracy on Test Set Using Epsilon = {}'.format(epsilon))
	# plt.show()

	# print('Generating graph for test set size variances')
	# accuracies = []
	# for testRatio in CONST_TEST_RATIOS:
	# 	naiveBayes = NaiveBayes(BEST_EPSILON, BEST_ALPHA, 0.1, testRatio)
	# 	naiveBayes.testAgainst(naiveBayes.testSet)
	# 	accuracy = naiveBayes.getAccuracy()
	# 	accuracies.append(accuracy)
	# plt.plot(CONST_TEST_RATIOS, accuracies)
	# plt.xlabel('Ratio Of Test Size')
	# plt.ylabel('Accuracy')
	# plt.title('Accuracy on Test Set Using Epsilon = {}, Alpha = {}, Validation Ratio = {}'.format(BEST_EPSILON, BEST_ALPHA, 0.1))
	# plt.show()


class NaiveBayes:
	# Need to test these still
	BEST_EPSILON = 0.01
	BEST_ALPHA = 0.92

	CONST_RANDOM_SEED = 20150819
	CONST_TO_PREDICT = 56
	CONST_HIT_RANGE = 20

	# For our tests
	CONST_TO_PREDICT = 500
	CONST_HIT_RANGE = 5

	CONST_TEST_RATIO = 0.5
	CONST_VALIDATION_RATIO = 0.1

	def __init__(self, epsilon = 0.01, alpha = 0.92, validation_ratio = 0.1, test_ratio = 0.5):
		self.epsilon = epsilon
		self.alpha = alpha
		self.validation_ratio = validation_ratio
		self.test_ratio = test_ratio

		with open('stopwords.txt') as f:
			self.stopWords = f.read().splitlines()

		filename = 'training.1600000.processed.noemoticon.csv'
		self.dataset = self.readCsv(filename) # Contains only tweets with hashtags

		self.validationSet, self.dataset = self.splitDataset(self.dataset, self.validation_ratio)
		self.testSet, self.trainSet = self.splitDataset(self.dataset, self.test_ratio)

		self.wordCounts, self.hashtagCounts = self.generateCounts()
		self.wordsMappedToHashtags = self.generateHashtagSpecificVocabulary()
		self.hashtagsToPredict = self.getHashtagsToPredict()
		self.hitRange = NaiveBayes.CONST_HIT_RANGE

		self.correctPredictions = []
		self.incorrectPredictions = []

	def generateCounts(self):
		wordCounts = {}
		hashtagCounts = {}

		for tweet in self.trainSet:
			hashtags = []
			for word in tweet.split():
				if word.startswith('#') and len(word) > 2:
					word = word.lower().translate(string.maketrans("",""), string.punctuation) # remove punctuation
					hashtags.append(word)
					if word not in wordCounts:
						wordCounts[word] = 1
					else:
						wordCounts[word] += 1
				else:
					if '@' in word:
						continue
					if word in self.stopWords:
						continue
					word = word.lower().translate(string.maketrans("",""), string.punctuation) # remove punctuation
					if word not in wordCounts:
						wordCounts[word] = 1
					else:
						wordCounts[word] += 1

			for hashtag in hashtags:
				if hashtag not in hashtagCounts:
					hashtagCounts[hashtag] = 1.0
				else:
					hashtagCounts[hashtag] += 1.0

		return wordCounts, hashtagCounts

	def readCsv(self, filename):
		corpus = read_csv(filename)
		corpus.columns = ["1", "2", "3", "4", "5", "tweet"]
		corpus = corpus["tweet"]
		dataset = [tweet for tweet in corpus if '#' in tweet]
		return dataset

	def splitDataset(self, dataset, ratio):
		idx = int(len(dataset) * ratio)
		numpy.random.seed(NaiveBayes.CONST_RANDOM_SEED)
		numpy.random.shuffle(dataset)
		set1 = dataset[:idx]
		set2 = dataset[idx:]
		return set1, set2

		numpy.random.seed(NaiveBayes.CONST_RANDOM_SEED)
		idx = numpy.random.permutation(len(dataset))
		testSet = array(dataset)[idx[len(idx)/2:]]
		trainSet = array(dataset)[idx[:len(idx)/2]]
		return testSet, trainSet

	def generateHashtagSpecificVocabulary(self):
		wordsMappedToHashtags = {}

		for tweet in self.trainSet:
			words = []
			hashtags = []
			for word in tweet.split():
				if word.startswith('#') and len(word) > 2:
					word = word.lower().translate(string.maketrans("",""), string.punctuation) # remove punctuation
					hashtags.append(word)
					words.append(word)
				else:
					if '@' in word:
						continue
					if word in self.stopWords:
						continue
					word = word.lower().translate(string.maketrans("",""), string.punctuation) # remove punctuation
					words.append(word)

			for hashtag in hashtags:
				if hashtag not in wordsMappedToHashtags:
					wordsMappedToHashtags[hashtag] = {}
				for word in words:
					if word not in wordsMappedToHashtags[hashtag]:
						wordsMappedToHashtags[hashtag][word] = 1.0
					else:
						wordsMappedToHashtags[hashtag][word] += 1.0

		return wordsMappedToHashtags


	def getHashtagsToPredict(self):
		return map(operator.itemgetter(0), sorted(self.hashtagCounts.items(), key=operator.itemgetter(1))[-NaiveBayes.CONST_TO_PREDICT:])

	def testAgainst(self, testSet):
		self.time = time.time()
		i = 0
		hits = 0
		for tweet in testSet:
			words = []
			hashtags = []

			for word in tweet.split():
				if word.startswith('#') and len(word) > 2:
					word = word.lower().translate(string.maketrans("",""), string.punctuation) # remove punctuation
					hashtags.append(word)
					# words.append(word) DON'T WANT TO BE AWARE OF THE HASHTAG IN TESTSET
				else:
					if '@' in word:
						continue
					if word in self.stopWords:
						continue
					word = word.lower().translate(string.maketrans("",""), string.punctuation) # remove punctuation
					words.append(word)

			if len(set(hashtags).intersection(self.hashtagsToPredict)) == 0:
				continue # Can't predict any hashtags for this tweet

			i += 1
			probabilitiesMappedToHashtagsToPredict = {}
			for hashtag in self.hashtagsToPredict:
				prob = 0
				for word in words:
					prob += log(self.epsilon + self.wordsMappedToHashtags[hashtag].get(word, 0.0)) - log(self.hashtagCounts[hashtag])
				probabilitiesMappedToHashtagsToPredict[hashtag] = self.alpha*log(self.hashtagCounts[hashtag]) + (1-self.alpha)*prob - log(len(self.trainSet))
				# the hashtagCounts*alpha is the priors
			topProbabilities = map(operator.itemgetter(0), sorted(probabilitiesMappedToHashtagsToPredict.items(), key=operator.itemgetter(1))[-self.hitRange:])

			if len(set(hashtags).intersection(topProbabilities)) > 0:
				hits += 1
				if len(self.correctPredictions) < 20:
					self.correctPredictions.append(tweet)
			else:
				if len(self.incorrectPredictions) < 20:
					self.incorrectPredictions.append(tweet)

		self.accuracy = hits/i
		self.time = time.time() - self.time
		print('Processed {} tweets, only {} had a predictable hashtag, accuracy was: {}, this took {} seconds. EPSILON: {} | ALPHA: {}'.format(len(testSet), i, self.accuracy, self.time, self.epsilon, self.alpha))

	def getProbabilityResults(self, testSet):
		i = 0
		hits = 0
		for tweet in testSet:
			words = []
			hashtags = []

			for word in tweet.split():
				if word.startswith('#') and len(word) > 2:
					word = word.lower().translate(string.maketrans("",""), string.punctuation) # remove punctuation
					hashtags.append(word)
					# words.append(word) DON'T WANT TO BE AWARE OF THE HASHTAG IN TESTSET
				else:
					if '@' in word:
						continue
					if word in self.stopWords:
						continue
					word = word.lower().translate(string.maketrans("",""), string.punctuation) # remove punctuation
					words.append(word)

			if len(set(hashtags).intersection(self.hashtagsToPredict)) == 0:
				continue # Can't predict any hashtags for this tweet

			i += 1
			probabilitiesMappedToHashtagsToPredict = {}
			probPerWord = {}
			for hashtag in self.hashtagsToPredict:
				probPerWord[hashtag] = {}
				prob = 0
				for word in words:
					prob += log(self.epsilon + self.wordsMappedToHashtags[hashtag].get(word, 0.0)) - log(self.hashtagCounts[hashtag])
					probPerWord[hashtag][word] = prob
				probabilitiesMappedToHashtagsToPredict[hashtag] = self.alpha*log(self.hashtagCounts[hashtag]) + (1-self.alpha)*prob - log(len(self.trainSet))

			topProbabilities = map(operator.itemgetter(0), sorted(probabilitiesMappedToHashtagsToPredict.items(), key=operator.itemgetter(1))[-50:])
			print('These are the probability results for the tweet with words: {}, hashtags associated = {}'.format(', '.join(words), ', '.join(hashtags)))

			medianIndex = int(len(topProbabilities)/2)
			i = 0
			total = 0
			minVal = 1000000000
			maxVal = -1000000000

			print(probPerWord[hashtag])
			for hashtag in topProbabilities:
				prob = sum(probPerWord[hashtag].values())
				total += prob

				if i == medianIndex:
					median = hashtag

				if prob < minVal:
					minVal = prob
					minHashtag = hashtag

				if prob > maxVal:
					maxVal = prob
					maxHashtag = hashtag

				i += 1
				print(hashtag, probPerWord[hashtag])

			print('Median is: |||{}||| with sum of {} for {}'.format(median, sum(probPerWord[median].values()), probPerWord[median]))
			print('Min is: |||{}||| with sum of {} for {}'.format(minHashtag, sum(probPerWord[minHashtag].values()), probPerWord[minHashtag]))
			print('Max is: |||{}||| with sum of {} for {}'.format(maxHashtag, sum(probPerWord[maxHashtag].values()), probPerWord[maxHashtag]))
			print('Average of the summations is: {}'.format(total / len(topProbabilities)))

			print('This is the probability result for the TARGET:')
			print(hashtag, probPerWord[hashtag])

	def getAccuracy(self):
		return self.accuracy

	def getEpsilon(self):
		return self.epsilon

	def getAlpha(self):
		return self.alpha

	def setEpsilon(self, value):
		self.epsilon = value

	def setAlpha(self, value):
		self.alpha = value

	def getTimePassed(self):
		return self.time

	def setHashtagsToPredict(self, numHashtags):
		self.hashtagsToPredict = map(operator.itemgetter(0), sorted(self.hashtagCounts.items(), key=operator.itemgetter(1))[-numHashtags:])

	def setHitRange(self, hitRange):
		self.hitRange = hitRange

main()
