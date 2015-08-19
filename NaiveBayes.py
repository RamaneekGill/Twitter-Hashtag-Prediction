import csv
import random
import math
import sys

def readCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for line in range(len(dataset)):
		dataset[line] = [str(i) for i in dataset[line]]
	return dataset


# Finds hashtags in a tweet and extracts them
# Also returns a dataset of tweets that don't contain hashtags
def extractTweets(corpus, tweetIndex, maxNumHashtags=sys.maxsize):
	hashtagSet = list()
	dataset = list()

	for line in corpus:
		tempHashtagSet = list()

		for word in line[tweetIndex].split():
			if word.startswith('#'):
				tempHashtagSet.append(word)

		if len(tempHashtagSet) > 0:
			hashtagSet.append(tempHashtagSet)

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


def getUniqueHashtags(hashtagSet):
	uniqueHashtags = list()
	for i in range(len(hashtagSet)):
		for j in range(len(hashtagSet[i])):
			hashtag = hashtagSet[i][j][1:]
			hashtag = hashtag.lower()
			if hashtag not in uniqueHashtags:
				uniqueHashtags.append(hashtag)

	return uniqueHashtags;


def groupByHashtag(dataset, hashtagSet):
	uniqueHashtags = getUniqueHashtags(hashtagSet)
	separated = {}
	for i in range(len(dataset)):
		for uniqueHashtag in uniqueHashtags:
			if uniqueHashtag not in separated:
				separated[uniqueHashtag] = []

			# If the uniqueHashtag belongs to the tweet dataset[i]
			if any(uniqueHashtag in str for str in hashtagSet[i]):
				separated[uniqueHashtag].append(dataset[i])

	return uniqueHashtags, separated



#################################  MAIN  #######################################


def main():
	filename = 'testdata.manual.2009.06.14.csv'
	# filename = 'training.1600000.processed.noemoticon.csv'
	corpus = readCsv(filename)
	hashtagSet, dataset = extractTweets(corpus, -1, 50)
	trainSet, testSet = seperateDatasetInTwo(dataset, 0.8)
	uniqueHashtags, tweetsMappedToHashtag = groupByHashtag(dataset, hashtagSet)
	print(tweetsMappedToHashtag)

main()
