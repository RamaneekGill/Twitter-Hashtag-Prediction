from __future__ import division
import csv
import random
import math
import sys
import string
import re
import time
import pickle

CONST_EPSILON = 0.000000001
CONST_NUM_HASHTAGS = 56
CONST_PREDICT_TOP_N_HASHTAGS = 20
CONST_USE_USAGE_PRIORS = 1 # 1 for yes, 0 for no
CONST_SPLIT_TRAIN_TEST_RATIO = 0.8 # Percentage the training set should be
CONST_SPLIT_TRAIN_VALIDATION_RATIO = 0.1

def readCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for line in range(len(dataset)):
		dataset[line] = [str(i) for i in dataset[line]]
	return dataset


# Finds hashtags in a tweet and extracts them
# Also returns a dataset of tweets that don't contain hashtags
def extractHashtagsFromTweets(corpus, tweetIndex):
	hashtagSet = []
	dataset = []

	for line in corpus:
		tempHashtagSet = []

		if '#' in line[tweetIndex]:
			for word in line[tweetIndex].split():
				if len(word) < 3 : # Skip words that are less than 3 characters
					continue

				if word.startswith('#') and not isNumber(word[1:]):
					tempHashtagSet.append(word)
				if word.startswith('@'): # Remove @ mentions
					line[tweetIndex] = line[tweetIndex].replace(word, "")

			if len(tempHashtagSet) > 0:
				hashtagSet.append(tempHashtagSet)
				line[tweetIndex] = re.sub(r"http\S+", "", line[tweetIndex]) # Remove URLs

				# Remove the hastag from this tweet
				for word in tempHashtagSet:
					line[tweetIndex] = line[tweetIndex].replace(word, "")
				dataset.append(line[tweetIndex])

	return hashtagSet, dataset


def seperateDatasetInTwo(corpus, ratio):
	trainSize = int(len(corpus) * ratio)
	random.seed(10)
	random.shuffle(corpus)
	trainSet = corpus[:trainSize]
	testSet = corpus[trainSize:]
	return trainSet, testSet

	trainSet = list()
	testSet = corpus
	while len(trainSet) < trainSize:
		index = random.randrange(len(testSet))
		trainSet.append(testSet.pop(index))
	return trainSet, testSet


def groupByHashtag(dataset, hashtagSet):
	separated = {}

	for i in range(len(hashtagSet)):
		for hashtag in hashtagSet[i]:
			hashtag = removePunctuation(hashtag.lower())
			# Seed the dictionary with unique hashtags
			if hashtag not in separated:
				separated[hashtag] = []

			separated[hashtag].append(removePunctuation(dataset[i]).lower())

	return separated


def removePunctuation(myString):
	return myString.translate(string.maketrans("",""), string.punctuation)


def tokenize(string):
    string = removePunctuation(string)
    string = string.lower()
    return string.split()

def isNumber(string):
	try:
		float(string)
		return True
	except ValueError:
		return False

def countWords(words):
    wordCount = {}
    for word in words:
        wordCount[word] = wordCount.get(word, 0.0) + 1.0
    return wordCount

def doesTweetHaveAPredictableHashtag(tweetIndex, test_hashtagSet, uniquePopularHashtags):
	shouldProcess = False
	for hashtag in test_hashtagSet[tweetIndex]:
		if hashtag[1:] in uniquePopularHashtags:
			shouldProcess = True
	return shouldProcess

def createVocabulary(tweetsMappedToHashtags):
	vocabulary = {} # The overall vocabulary word count in our entire dataset
	hashtagSpecificVocabulary = {} # The words from tweets that are associated with the hashtag (key of dict)

	for hashtag in tweetsMappedToHashtags.keys():
		# Add this hashtag to the vocabulary
		if hashtag not in vocabulary:
			vocabulary[hashtag] = 0.0
		vocabulary[hashtag] += 1.0

		# Initialize the dictionary for a vocabulary associated with this hashtag
		if hashtag not in hashtagSpecificVocabulary:
			hashtagSpecificVocabulary[hashtag] = {}

		for tweet in tweetsMappedToHashtags[hashtag]:
			for word in tokenize(tweet):
				# Add word to global vocabulary
				if word not in vocabulary:
					vocabulary[word] = 0.0
				vocabulary[word] += 1.0

				# Add word to this hashtag's vocabulary
				if word not in hashtagSpecificVocabulary[hashtag]:
					hashtagSpecificVocabulary[hashtag][word] = 0.0
				hashtagSpecificVocabulary[hashtag][word] += 1.0

	return vocabulary, hashtagSpecificVocabulary


def keepNMostPopularHashtags(dictionary, upperLimit):
	count = 0
	for key in sorted(dictionary, key=lambda key: len(dictionary[key]), reverse=True):
		count += 1
		if count > upperLimit:
			del dictionary[key]

	return dictionary

def countWordOccursInDict(word, dictionary):
	if word not in dictionary:
		return 1.0 # since it is used inside of a log()
	return dictionary[word]

def countHashtagOccurrence(hashtag, dictionary):
	if hashtag not in dictionary:
		return 1.0 # since it is used inside of a log()
	return len(dictionary[hashtag])


# Returns P(w,h), i.e. probability word occurs when associated with the hashag
def countWordAndHashtagOccursTogether(word, hashtag, vocabulary, hashtagSpecificVocabulary):
	return hashtagSpecificVocabulary[hashtag][word] + CONST_EPSILON

# Returns P(w|h)
def logProbWordGivenHashtag(word, hashtag, vocabulary, hashtagSpecificVocabulary, tweetsMappedToHashtags):
	# return P(w,h) / P(h)
	if word in vocabulary and hashtag in hashtagSpecificVocabulary:
		if word in hashtagSpecificVocabulary[hashtag]:
			log_p_w_occurs_with_h = math.log(countWordAndHashtagOccursTogether(word, hashtag, vocabulary, hashtagSpecificVocabulary))
			log_count_hashtag_occurence = math.log(countHashtagOccurrence(hashtag, tweetsMappedToHashtags))
			return log_p_w_occurs_with_h - log_count_hashtag_occurence

	return 0.0


# Returns P(h|W)
def logProbHashtagGivenWord(hashtag, words, vocabulary, hashtagSpecificVocabulary, tweetsMappedToHashtags, trainTweetsLength):
	# return P(w|h) * P(h) / P(w) if we have the data to compute it, otherwise return 0.0
	# log of fract = log(numerator) - log(denominator)

	log_p_W_given_h = 0
	log_p_W = 0
	if CONST_USE_USAGE_PRIORS:
		for word in words:
			log_p_W_given_h += logProbWordGivenHashtag(word, hashtag, vocabulary, hashtagSpecificVocabulary, tweetsMappedToHashtags)
		# log_p_W += math.log(countWordOccursInDict(word, vocabulary)) - math.log(trainTweetsLength)

	log_p_h = math.log(countHashtagOccurrence(hashtag, tweetsMappedToHashtags)) - math.log(trainTweetsLength)
	return log_p_h + CONST_USE_USAGE_PRIORS * log_p_W_given_h # - log_p_W


def testWithTrainSet(test_tweets, test_hashtagSet, uniquePopularHashtags, vocabulary, hashtagSpecificVocabulary, tweetsMappedToPopularHashtags, train_tweetsLength):
	tweetsMappedToHashtagsProbabilities = {}
	countHashtagPredictedInTopTwenty = 0
	countHashtagNotPredictedInTopTwenty = 0
	start = time.time()

	for i in range(len(test_tweets)): # for every tweet
		# Check if this test_tweet actually has a hashtag that we've seen in the filtered training set
		if doesTweetHaveAPredictableHashtag(i, test_hashtagSet, uniquePopularHashtags) == False:
			continue # Since this tweet can't be predicted skip it

		probHashtagGivenTweet = {}
		for hashtag in uniquePopularHashtags:
			probHashtagGivenTweet[hashtag] = logProbHashtagGivenWord(hashtag, test_tweets[i], vocabulary, hashtagSpecificVocabulary, tweetsMappedToPopularHashtags, train_tweetsLength)

		# Only keep the top 20 recommended hashtags
		tweetsMappedToHashtagsProbabilities[i] = sorted(probHashtagGivenTweet, key=probHashtagGivenTweet.get, reverse=False)[:CONST_PREDICT_TOP_N_HASHTAGS]

		# If at least one of the actual hashtags associated with the tweet was in top 20 recommended, increment counter
		countHashtagNotPredictedInTopTwenty += 1
		for hashtag in test_hashtagSet[i]:
			if hashtag[1:] in tweetsMappedToHashtagsProbabilities[i]:
				countHashtagPredictedInTopTwenty += 1
				countHashtagNotPredictedInTopTwenty -= 1
				break

	timePassed = time.time() - start
	percentage = countHashtagPredictedInTopTwenty/(countHashtagPredictedInTopTwenty + countHashtagNotPredictedInTopTwenty)
	print('Processed ', i, 'tweets, accuracy so far is: {}, time passed so far: {}'.format(percentage, timePassed))

	return percentage

#################################  MAIN  #######################################


def main():
	filename = 'testdata.manual.2009.06.14.csv'
	filename = 'training.1600000.processed.noemoticon.csv'

	# Read the CSV
	corpus = readCsv(filename)

	print('Read the dataset')

	# Seperate the CSV into a train and test set
	trainSet, testSet = seperateDatasetInTwo(corpus, CONST_SPLIT_TRAIN_TEST_RATIO)
	validationSet = seperateDatasetInTwo(trainSet, CONST_SPLIT_TRAIN_VALIDATION_RATIO)

	print('Processed the dataset into train and test sets')

	# Get only the tweets from the set passed in
	# Also extract (which means remove as well) the hashtags from the tweet
	train_hashtagSet, train_tweets = extractHashtagsFromTweets(trainSet, -1)
	test_hashtagSet, test_tweets = extractHashtagsFromTweets(testSet, -1)
	validation_hashtagSet, validation_tweets = extractHashtagsFromTweets(validationSet, -1)


	print('Processed train, test and validation set and extracted hashtags and tweets')

	# Process the trainSet and get the data necessary for calculating probabilities
	tweetsMappedToHashtags = groupByHashtag(train_tweets, train_hashtagSet)
	tweetsMappedToPopularHashtags = keepNMostPopularHashtags(tweetsMappedToHashtags, CONST_NUM_HASHTAGS)
	uniquePopularHashtags = tweetsMappedToPopularHashtags.keys() # Get all the unique hashtags in the training set with the '#' stripped
	vocabulary, hashtagSpecificVocabulary = createVocabulary(tweetsMappedToPopularHashtags)

	print('Generated vocabularies and kept tweets only with the top {} hashtags').format(CONST_NUM_HASHTAGS)


############################################################################################################################################################
############################################################################################################################################################

	print('Testing the test set')
 	accuracy = testWithTrainSet(test_tweets, test_hashtagSet, uniquePopularHashtags, vocabulary, hashtagSpecificVocabulary, tweetsMappedToPopularHashtags, len(train_tweets))
	print(accuracy)


############################################################################################################################################################
############################################################################################################################################################


main()
