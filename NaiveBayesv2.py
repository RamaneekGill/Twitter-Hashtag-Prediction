from __future__ import division
import csv
import random
import math
import sys
import string
import re
import time
from numpy import *
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt


class NaiveBayes:
	BEST_EPSILON = 1e-9
	BEST_ALPHA = 1

	CONST_RANDOM_SEED = 20150819
	CONST_NUM_HASHTAGS = 56
	CONST_TO_PREDICT = 20
	CONST_USE_USAGE_PRIORS = 1
	CONST_SPLIT_TRAIN_TEST_RATIO = 0.5
	CONST_SPLIT_TRAIN_VALIDATION_RATIO = 0.1

	def __init__(self):
		filename = 'training.1600000.processed.noemoticon.csv'
		self.dataset = self.readCsv(filename) # Contains only tweets with hashtags
		self.testSet, self.trainSet = self.splitDataset(self.corpus, NaiveBayes.CONST_SPLIT_TRAIN_TEST_RATIO)
		self.wordCounts, self.hashtagCounts = self.generateCounts()
		self.wordsMappedToHashtags = self.generateHashtagSpecificVocabulary()
		self.hashtagToPredict = self.getHashtagsToPredict()

	def generateCounts(self):
		with open('stopwords.txt') as f:
			stopWords = f.read().splitlines()

		wordCounts = {}
		hashtagCounts = {}

		for tweet in self.trainSet:
			hashtags = []
			for word in tweet.split():
				if word.startswith('#') and len(word) > 2:
					word = word.translate(string.maketrans("",""), string.punctuation) # remove punctuation
					hashtags.append(word.lower())
				else:
					if '#' in word or '@' in word or len(word) < 3:
						continue
					if word in stopWords:
						continue
					word = word.lower()
					word = word.translate(string.maketrans("",""), string.punctuation) # remove punctuation
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
		dataset = tweet for tweet in corpus if '#' in tweet)
		return dataset

	def splitDataset(self, ratio):
		numpy.random.seed(NaiveBayes.CONST_RANDOM_SEED)
		idx = numpy.random.permutation(len(self.dataset))
		testSet = array(self.dataset)[idx[len(idx)/2:]]
		trainSet = array(self.dataset)[idx[:len(idx)/2]]
		return testSet, trainSet

	def generateHashtagSpecificVocabulary(self):
		with open('stopwords.txt') as f:
			stopWords = f.read().splitlines()

		wordsMappedToHashtags = {}

		for tweet in self.trainSet:
			words = []
			hashtags = []
			for word in tweet.split():
				if word.startswith('#') and len(word) > 2:
					word = word.translate(string.maketrans("",""), string.punctuation) # remove punctuation
					hashtags.append(word.lower())
					words.append(word)
				else:
					if '#' in word or '@' in word or len(word) < 3:
						continue
					if word in stopWords:
						continue
					word = word.lower()
					word = word.translate(string.maketrans("",""), string.punctuation) # remove punctuation
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
