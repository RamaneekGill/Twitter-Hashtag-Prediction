from __future__ import division
import csv
import random
import math
import sys
import string
import re
import time
import pickle
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
		self.corpus = self.readCsv(filename)
		self.testSet, self.trainSet = self.splitDataset(self.corpus, NaiveBayes.CONST_SPLIT_TRAIN_TEST_RATIO)
		self.hashtagCounts = self.countHashtags()
		self.wordCounts = self.countWords()
		self.wordsMappedToHashtagCounts = self.generateHashtagSepecificVocabulary()
		self.hashtagToPredict = self.getHashtagsToPredict()

	def readCsv(self, filename):
