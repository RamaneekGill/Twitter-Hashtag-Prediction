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

# Finds hashtags in a tweet
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
			dataset.append(line[tweetIndex])

	return hashtagSet, dataset

###############################  MAIN  #########################################

def main():
	filename = 'testdata.manual.2009.06.14.csv'
	corpus = readCsv(filename)
	hashtagSet, dataset = extractTweets(corpus, -1, 50)
	print(hashtagSet)
	print(len(hashtagSet), len(dataset), len(corpus))

main()
