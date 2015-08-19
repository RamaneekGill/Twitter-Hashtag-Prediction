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

	For every hashtag in hashtagSet (i.e. the hashtags for the ith tweet)
		Add the hashtag from the set to the vocabulary
		Get the wordCount for the ith tweet
		for each word and count in wordCount
			add the word to the vocabulary
			add the count to hashTagWordMap[hashtag for hashtag in hashtagSet][word]


	for uniqueHashtag in uniqueHashtags:
		if uniqueHashtag not in vocabulary:
			vocabulary[word] = 0.0
		else:
			vocabulary[word] += 1.0
		for i in range(len(dataset)):
			if (uniqueHashtag in hashtagSet[i]):
				wordCount = countWords(tokenize(dataset[i]))
				for word, count in wordCount.items:
					if word not in vocabulary:
						vocabulary[word] = 0.0
					if word not in hashtagWordMap[uniqueHashtag]




	words = tokenize(text)
    counts = count_words(words)
    for word, count in counts.items():
        # if we haven't seen a word yet, let's add it to our dictionaries with a count of 0
        if word not in vocab:
            vocab[word] = 0.0 # use 0.0 here so Python does "correct" math
        if word not in word_counts[category]:
            word_counts[category][word] = 0.0
        vocab[word] += count
        word_counts[category][word] += count


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
