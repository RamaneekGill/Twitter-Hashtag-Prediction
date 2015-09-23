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
	naiveBayes = NaiveBayes()


class NaiveBayes:
	BEST_EPSILON = 1e-9
	BEST_ALPHA = 1

	CONST_RANDOM_SEED = 20150819
	CONST_TO_PREDICT = 56
	CONST_HIT_RANGE = 20
	CONST_SPLIT_TRAIN_TEST_RATIO = 0.5
	CONST_SPLIT_TRAIN_VALIDATION_RATIO = 0.1

	def __init__(self):
		with open('stopwords.txt') as f:
			self.stopWords = f.read().splitlines()
		filename = 'training.1600000.processed.noemoticon.csv'
		self.dataset = self.readCsv(filename) # Contains only tweets with hashtags
		self.testSet, self.trainSet = self.splitDataset(NaiveBayes.CONST_SPLIT_TRAIN_TEST_RATIO)
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
					# if word not in wordCounts:
					# 	wordCounts[word] = 1
					# else:
					# 	wordCounts[word] += 1
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
		i = 0
		hits = 0
		for tweet in testSet:
			words = []
			hashtags = []

			for word in tweet.split():
				if word.startswith('#') and len(word) > 2:
					word = word.lower().translate(string.maketrans("",""), string.punctuation) # remove punctuation
					# word = word.lower().strip('#!,.')
					hashtags.append(word)
					# words.append(word)
				else:
					if '@' in word:
						continue
					if word in self.stopWords:
						continue
					# word = word.lower().strip('#!,.')
					word = word.lower().translate(string.maketrans("",""), string.punctuation) # remove punctuation
					words.append(word)

			if len(set(hashtags).intersection(self.hashtagsToPredict)) == 0:
				continue # Can't predict any hashtags for this tweet

			i += 1
			probabilitiesMappedToHashtagsToPredict = {}
			for hashtag in self.hashtagsToPredict:
				prob = 0
				for word in words:
					prob += log(NaiveBayes.BEST_EPSILON + self.wordsMappedToHashtags[hashtag].get(word, 0.0)) - log(self.hashtagCounts[hashtag])
				probabilitiesMappedToHashtagsToPredict[hashtag] = NaiveBayes.BEST_ALPHA*log(self.hashtagCounts[hashtag]) + prob - log(len(self.trainSet))

			topProbabilities = map(operator.itemgetter(0), sorted(probabilitiesMappedToHashtagsToPredict.items(), key=operator.itemgetter(1))[-NaiveBayes.CONST_HIT_RANGE:])

			hits += len(set(hashtags).intersection(topProbabilities)) > 0

		print('Processed {} tweets, only {} had a predictable hashtag, accuracy was: {}'.format(len(self.testSet), i, hits/i))


main()
