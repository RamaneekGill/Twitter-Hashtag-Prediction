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
	hashtagSet = list()
	dataset = list()

	for line in corpus:
		tempHashtagSet = list()

		if '#' in line[tweetIndex]:
			for word in line[tweetIndex].split():
				if word.startswith('#') and len(word) > 1 and not isNumber(word[1:]):
					tempHashtagSet.append(word)

			if len(tempHashtagSet) > 0:
				hashtagSet.append(tempHashtagSet)
				line[tweetIndex] = re.sub(r"http\S+", "", line[tweetIndex]) # Remove URLs
				if word.startswith('@'): # Remove @ mentions
					line[tweetIndex] = line[tweetIndex].replace(word, "")


			# Remove the hastag from this tweet
			for word in tempHashtagSet:
				line[tweetIndex] = line[tweetIndex].replace(word, "")
			dataset.append(line[tweetIndex])

	return hashtagSet, dataset


def seperateDatasetInTwo(dataset, ratio):
	trainSize = int(len(dataset) * ratio)
	trainSet = list()
	testSet = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(testSet))
		trainSet.append(testSet.pop(index))
	return [trainSet, testSet]


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
	return p_w_occurs_with_h / p_h

# Returns P(h|w)
def probHashtagGivenWord(hashtag, word, vocabulary, hashtagSpecificVocabulary, tweetsMappedToHashtags):
	# return P(w|h) * P(h) / P(w)
	p_w_given_h = probWordGivenHashtag(word, hashtag, vocabulary, hashtagSpecificVocabulary, tweetsMappedToHashtags)
	p_h = countHashtagOccurrence(hashtag, tweetsMappedToHashtags)
	p_w = countWordOccursInDict(word, vocabulary)
	return p_w_given_h * p_h / p_w


#################################  MAIN  #######################################


def main():
	filename = 'testdata.manual.2009.06.14.csv'
	filename = 'training.1600000.processed.noemoticon.csv'

	corpus = readCsv(filename)

	hashtagSet, dataset = extractHashtagsFromTweets(corpus, -1)
	# trainSet, testSet = seperateDatasetInTwo(dataset, 0.8)

	tweetsMappedToHashtags = groupByHashtag(dataset, hashtagSet)
	tweetsMappedToPopularHashtags = keepNMostPopularHashtags(tweetsMappedToHashtags, 10)

	vocabulary, hashtagSpecificVocabulary = createVocabulary(tweetsMappedToPopularHashtags)

	# count = 0
	# for hashtag in tweetsMappedToPopularHashtags.keys():
	# 	count += 1
	# 	if count < 10:
	# 		print(hashtag, tweetsMappedToPopularHashtags[hashtag])


	print(len(vocabulary.keys()), len(hashtagSpecificVocabulary))
	print('sdfkdjsfbsdfk')
	for key in hashtagSpecificVocabulary.keys():
		print(key, len(hashtagSpecificVocabulary[key]))

	print(hashtagSpecificVocabulary['fail']['office'])

	print(probWordGivenHashtag('office', 'fail', vocabulary, hashtagSpecificVocabulary, tweetsMappedToPopularHashtags))
	print(probHashtagGivenWord('fail', 'office', vocabulary, hashtagSpecificVocabulary, tweetsMappedToPopularHashtags))


main()
