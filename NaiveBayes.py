import csv
import random
import math
import sys
import string
import re

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
	return dictionary[word]

def countHashtagOccurrence(hashtag, dictionary):
	return len(dictionary[hashtag])

def probWordOccursInDict(word, dictionary):
	return countWordOccursInDict(word, dictionary) / sum(dictionary.values())

# Returns P(w,h), i.e. probability word occurs when associated with the hashag
def probWordOccursWithHashtag(word, hashtag, vocabulary, hashtagSpecificVocabulary):
	return hashtagSpecificVocabulary[hashtag][word] / vocabulary[word]

# Returns P(w|h)
def probWordGivenHashtag(word, hashtag, vocabulary, hashtagSpecificVocabulary, tweetsMappedToHashtags):
	# return P(w,h) / P(h)
	p_w_occurs_with_h = probWordOccursWithHashtag(word, hashtag, vocabulary, hashtagSpecificVocabulary)
	p_h = countHashtagOccurrence(hashtag, tweetsMappedToHashtags)
	return p_w_occurs_with_h / p_h # add 0.0000001 to numerator before dividing

# Returns P(h|w)
def probHashtagGivenWord(hashtag, word, vocabulary, hashtagSpecificVocabulary, tweetsMappedToHashtags):
	# return P(w|h) * P(h) / P(w) if we have the data to compute it, otherwise return 0.0

	if word in vocabulary and hashtag in hashtagSpecificVocabulary:
		if word in hashtagSpecificVocabulary[hashtag]:
			p_w_given_h = probWordGivenHashtag(word, hashtag, vocabulary, hashtagSpecificVocabulary, tweetsMappedToHashtags)
			p_h = countHashtagOccurrence(hashtag, tweetsMappedToHashtags)
			p_w = countWordOccursInDict(word, vocabulary)
			return p_w_given_h * p_h / p_w

	return 0.0

#################################  MAIN  #######################################


def main():
	filename = 'testdata.manual.2009.06.14.csv'
	filename = 'training.1600000.processed.noemoticon.csv'

	# Read the CSV
	corpus = readCsv(filename)

	# Seperate the CSV into a train and test set
	trainSet, testSet = seperateDatasetInTwo(corpus, 0.5)

	# Get only the tweets from the set passed in
	# Also extract (which means remove as well) the hashtags from the tweet
	train_hashtagSet, train_tweets = extractHashtagsFromTweets(trainSet, -1)
	test_hashtagSet, test_tweets = extractHashtagsFromTweets(testSet, -1)


	# Process the trainSet and get the data necessary for calculating probabilities
	tweetsMappedToHashtags = groupByHashtag(train_tweets, train_hashtagSet)
	tweetsMappedToPopularHashtags = keepNMostPopularHashtags(tweetsMappedToHashtags, 56)
	uniquePopularHashtags = tweetsMappedToPopularHashtags.keys() # Get all the unique hashtags in the training set with the '#' stripped
	vocabulary, hashtagSpecificVocabulary = createVocabulary(tweetsMappedToPopularHashtags)

	probPerTweetPerWordPerHashtag = {}
	for i in range(len(test_tweets)): # for every tweet
		# Check if this test_tweet actually has a hashtag that we've seen in the filtered training set
		if doesTweetHaveAPredictableHashtag(i, test_hashtagSet, uniquePopularHashtags) == False:
			continue # Since this tweet can't be predicted skip it

		probPerWordPerHashtag = {}
		for word in test_tweets[i].split(): # for every word in the tweet

			probPerHashtag = {}
			for hashtag in uniquePopularHashtags: # For every unique hashtag in the filtered training set
				probPerHashtag[hashtag] = 0.0

				# Calculate P(h|w) value, otherwise it is 0
				probPerHashtag[hashtag] += probHashtagGivenWord(hashtag, word, vocabulary, hashtagSpecificVocabulary, tweetsMappedToPopularHashtags)

			probPerWordPerHashtag[word] = probPerHashtag # associate prob per hashtag to the word
		probPerTweetPerWordPerHashtag[test_tweets[i]] = probPerWordPerHashtag # associate prob per word per hashtag to the tweet


	print(len(probPerTweetPerWordPerHashtag.keys()), len(test_tweets))

main()
