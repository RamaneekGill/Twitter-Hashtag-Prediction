from __future__ import division
import csv
import random
import math
import sys
import string
import re
import time
from PIL import Image
from pylab import *
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
	CONST_EPSILON_INTERVALS = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001, 0.00000000001]
	CONST_ALPHA_INTERVALS = [0.0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7]
	CONST_ALPHA_INTERVALS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	naiveBayes = NaiveBayes(1, 1)

	print('Performing cross validation to find the best epsilon and alpha values')
	accuracies = []
	maxAccuracy = 0
	for epsilon in CONST_EPSILON_INTERVALS:
		for alpha in CONST_ALPHA_INTERVALS:
			naiveBayes.setEpsilon(epsilon)
			naiveBayes.setAlpha(alpha)
			naiveBayes.testAgainst(naiveBayes.testSet)
			accuracy = naiveBayes.getAccuracy()
			accuracies.append(accuracy)
			if max(accuracies) > maxAccuracy:
				BEST_EPSILON = epsilon
				BEST_ALPHA = alpha
				maxAccuracy = max(accuracies)
	print('Validation tests have shown that the best epsilon value to use is: {}, best alpha value is: {}'.format(BEST_EPSILON, BEST_ALPHA))


	print('Generating graph for epsilon accuracies')
	epsilonAccuracies = []
	alpha = BEST_ALPHA
	for epsilon in CONST_EPSILON_INTERVALS:
		naiveBayes.setEpsilon(epsilon)
		naiveBayes.setAlpha(alpha)
		naiveBayes.testAgainst(naiveBayes.testSet)
		accuracy = naiveBayes.getAccuracy()
		epsilonAccuracies.append(accuracy)
	plt.plot(CONST_EPSILON_INTERVALS, epsilonAccuracies)
	plt.xlabel('Epsilon')
	plt.ylabel('Accuracy')
	plt.title('Accuracy on Test Set Using Alpha = {}'.format(alpha))
	plt.show()


	print('Generating graph for alpha accuracies')
	alphaAccuracies = []
	epsilon = BEST_EPSILON
	for alpha in CONST_ALPHA_INTERVALS:
		naiveBayes.setEpsilon(epsilon)
		naiveBayes.setAlpha(alpha)
		naiveBayes.testAgainst(naiveBayes.testSet)
		accuracy = naiveBayes.getAccuracy()
		alphaAccuracies.append(accuracy)
	plt.plot(CONST_ALPHA_INTERVALS, alphaAccuracies)
	plt.xlabel('Alpha')
	plt.ylabel('Accuracy')
	plt.title('Accuracy on Test Set Using Epsilon = {}'.format(epsilon))
	plt.show()


class NaiveBayes:
	# Need to test these still
	BEST_EPSILON = 1e-5
	BEST_ALPHA = 0.9

	CONST_RANDOM_SEED = 20150819
	CONST_TO_PREDICT = 56
	CONST_HIT_RANGE = 20
	CONST_SPLIT_TRAIN_TEST_RATIO = 0.5
	CONST_SPLIT_TRAIN_VALIDATION_RATIO = 0.1

	def __init__(self, epsilon, alpha):
		self.epsilon = epsilon
		self.alpha = alpha

		with open('stopwords.txt') as f:
			self.stopWords = f.read().splitlines()
		filename = 'training.1600000.processed.noemoticon.csv'
		self.dataset = self.readCsv(filename) # Contains only tweets with hashtags
		self.testSet, self.trainSet = self.splitDataset(self.CONST_SPLIT_TRAIN_TEST_RATIO)
		self.wordCounts, self.hashtagCounts = self.generateCounts()
		self.wordsMappedToHashtags = self.generateHashtagSpecificVocabulary()
		self.hashtagsToPredict = self.getHashtagsToPredict()
		self.testAgainst(self.testSet)

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

	def splitDataset(self, ratio):
		numpy.random.seed(NaiveBayes.CONST_RANDOM_SEED)
		idx = numpy.random.permutation(len(self.dataset))
		testSet = array(self.dataset)[idx[len(idx)/2:]]
		trainSet = array(self.dataset)[idx[:len(idx)/2]]
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

			topProbabilities = map(operator.itemgetter(0), sorted(probabilitiesMappedToHashtagsToPredict.items(), key=operator.itemgetter(1))[-NaiveBayes.CONST_HIT_RANGE:])

			hits += len(set(hashtags).intersection(topProbabilities)) > 0

		self.accuracy = hits/i
		self.time = time.time() - self.time
		print('Processed {} tweets, only {} had a predictable hashtag, accuracy was: {}, this took {} seconds. EPSILON: {} | ALPHA: {}'.format(len(self.testSet), i, self.accuracy, self.time, self.epsilon, self.alpha))

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

main()
