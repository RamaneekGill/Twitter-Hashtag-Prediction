import csv
import random
import math

def readCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for line in range(len(dataset)):
		dataset[line] = [str(i) for i in dataset[line]]
	return dataset

def extractTweets(dataset, tweetIndex, maxNumHashtags):
	copy = list(dataset)
	hashtagSet = list()
	for line in copy:
		for word in line[tweetIndex].split():
			if word.startswith('#'):
				hashtagSet.insert(-1, [word, copy.index(line)])
				line[tweetIndex].replace(word, "")
				if len(hashtagSet) == maxNumHashtags:
					return hashtagSet, copy

	return hashtagSet, copy


def main():
	filename = 'testdata.manual.2009.06.14.csv'
	dataset = readCsv(filename)
	hashtagSet, hashtagExtractedDataset = extractTweets(dataset, -1, 2)
	print(hashtagSet)

main()
