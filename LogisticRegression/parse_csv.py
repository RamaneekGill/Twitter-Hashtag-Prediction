import pickle
import csv
from pandas import *
import numpy
from numpy import *
import string


print('loading stopwords')
with open('../stopwords.txt') as f:
	STOP_WORDS = f.read().splitlines()

def read_dataset():

	# read csv
	print('reading from csv')
	filename = '../training.1600000.processed.noemoticon.csv'
	corpus = read_csv(filename)
	corpus.columns = ["1", "2", "3", "4", "5", "tweet"]
	corpus = corpus["tweet"]

	print('filtering out tweets that don\'t contain a hashtag')
	# filter out tweets without a hashtag
	dataset = [tweet for tweet in corpus if '#' in tweet]

	# split dataset into train, test, validation set
	print('splitting dataset into train, validation and test sets')
	validation_ratio = 0.1
	test_ratio = 0.1 # Therefore train ratio is 0.8
	validation_set, dataset = splitDataset(dataset, validation_ratio)
	test_set, train_set = splitDataset(dataset, test_ratio)

	# strip out hashtags in to seperate target lists
	print('splitting train, validation, and test sets into inputs and targets and stripping punctuation')
	train_inputs, train_targets = splitInputsTargets(train_set)
	validation_inputs, validation_targets = splitInputsTargets(validation_set)
	test_inputs, test_targets = splitInputsTargets(test_set)

	print('done parsing csv!')
	return train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets

def splitDataset(dataset, ratio):
	CONST_RANDOM_SEED = 20150819

	idx = int(len(dataset) * ratio)
	numpy.random.seed(CONST_RANDOM_SEED)
	numpy.random.shuffle(dataset)
	set1 = dataset[:idx]
	set2 = dataset[idx:]
	return set1, set2

def splitInputsTargets(dataset):
	inputs = []
	targets = []
	indexes_to_remove = []

	for i in range(len(dataset)):
		inputs.append([])
		targets.append([])

		for word in dataset[i].split():
			if word.startswith('#') and len(word) > 2:
				word = stripPunctuationAndCreateList(word)
				if word:
					targets[i].append(word)
				continue
			else:
				word = stripPunctuationAndCreateList(word, strip_stopwords=True)
				if word:
					inputs[i].append(word)

		if not targets[i]: # no predictable hashtags
			indexes_to_remove.append(i)

	for i in indexes_to_remove[::-1]:
		targets.pop(i)
		inputs.pop(i)

	return inputs, targets

def stripPunctuationAndCreateList(word, strip_stopwords=False):
	word = word.lower().translate(string.maketrans("",""), string.punctuation)

	if strip_stopwords:
		if word in STOP_WORDS:
			return ''

	return word
