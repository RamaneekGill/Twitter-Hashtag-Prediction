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


def removePunctuation(string):
    table = string.maketrans("","")
    return string.translate(table, string.punctuation)


def tokenize(string):
    string = remove_punctuation(string)
    string = string.lower()
    return re.split("\W+", string)


def countWords(words):
    wordCount = {}
    for word in words:
        wordCount[word] = wordCount.get(word, 0.0) + 1.0
    return wordCount


def createVocabulary(dataset, hashtagSet, uniqueHashtags):
	vocabulary = {} # The overall vocabulary word count in our entire dataset
	hashtagWordMap = {} # The word count for each word that is associated with a hashtag

	# Generate the keys for the hashtagWordMap
	for uniqueHashtag in uniqueHashtags:
		# Add this hashtag to the vocabulary
		if uniqueHashtag not in vocabulary:
			vocabulary[uniqueHashtag] = 0.0
		vocabulary[uniqueHashtag] += 1.0

		# Make this hashtag a key in the map if it does not exist already
		if uniqueHashtag not in hashtagWordMap:
			hashtagWordMap[uniqueHashtag] = 0.0
		hashtagWordMap[uniqueHashtag] += 1.0

	# Map the words in dataset[i] to all the hashtags in the list hashtagSet[i] into hashtagWordMap
	for i in range(len(dataset)):
		countWords = countWords(tokenize(dataset[i]))
		for word, count in list(countWords.items()): # For each word and count in this tweet
			# Add this word to the vocabulary
			if word not in vocabulary:
				vocabulary[word] = 0.0
			vocabulary[word] += 1.0

			for hashtag in hashtagSet[i]: # All the hashtags associated with this tweet
				if word not in hashtagWordMap[hashtag]: # If this word is not associated with the hashtag
					hashtagWordMap[hashtag][word] = 0.0  # Initialize it
				hashtagWordMap[hashtag][word] += 1.0


#################################  MAIN  #######################################


def main():
	filename = 'testdata.manual.2009.06.14.csv'
	# filename = 'training.1600000.processed.noemoticon.csv'
	corpus = readCsv(filename)
	hashtagSet, dataset = extractTweets(corpus, -1, 50)
	trainSet, testSet = seperateDatasetInTwo(dataset, 0.8)
	uniqueHashtags, tweetsMappedToHashtag = groupByHashtag(dataset, hashtagSet)
	vocabularyFrequency = createVocabulary(dataset, uniqueHastags);


	print(tweetsMappedToHashtag)

main()
