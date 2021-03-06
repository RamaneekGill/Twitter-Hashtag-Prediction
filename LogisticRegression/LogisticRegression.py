from __future__ import division
import csv
from random import randint
import math
import sys
import string
import re
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import operator
from scipy.sparse import *
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy import ndimage
from pandas import *
import numpy
from numpy import *
from PIL import Image
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import tensorflow as tf


def getWords(vector, vocabulary):
	words = []
	for i in range(len(vector)):
		if vector[i] > 0:
			words.append(vocabulary[i])
	return words

def getWordsWithIndexes(indexes, vocabulary):
	words = []
	for index in indexes:
		words.append(vocabulary[index])
	return words


def main():
	import input_data # File responsible for parsing train, test, validation set
	# data contains the 3 different sets, Each a DataSet class
	numWords, numHashtags, data, tweet_vocabulary, hashtag_vocabulary, raw_test_input = input_data.read_data_sets()

	LEARNING_RATE = 0.03
	EPOCHS = 100
	BATCH_SIZE = 1000
	DISPLAY_STEP = 1 # To print cost every n number of epochs
	PREDICTION_RANGE = 5 # If actual target is in range of predictions then it is correct
	numWords = int(numWords)
	numHashtags = int(numHashtags)

	# Setup variables for batch feeding, these are hydrated when running/evaluating sessions
	x = tf.placeholder("float", [None, numWords]) # The inputs
	y = tf.placeholder("float", [None, numHashtags]) # The correct answers

	# Setup the model
	W = tf.Variable(tf.random_normal([numWords, numHashtags], stddev=0.01)) # weights matrix
	b = tf.Variable(tf.zeros([numHashtags])) # bias
	activation = tf.nn.softmax(tf.matmul(x,W) + b) # the predictions, choose between sigmoid and softmax

	# Specify the cost and optimization
	cost = -tf.reduce_sum(y*tf.log(activation)) # cross entropy is the cost function
	# cost = tf.reduce_sum((y-1)*tf.log(activation))
	# cost = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(activation, y))
	# cost = tf.reduce_sum(tf.square(y - activation))
	# cost = -tf.reduce_sum(activation*y + (1-activation)*(1-y))
	# cost = tf.reduce_sum()
	optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

	# Initialize session and its variables
	init = tf.initialize_all_variables()
	sess = tf.Session()

	print("Starting tensorflow session")
	start = time.time()
	sess.run(init)
	saver = tf.train.Saver() # Add ops to save and restore all the variables.

	# Train the model
	print("Training the model")
	print("Epochs to run {}".format(EPOCHS))
	print("Batches to run {}".format(int(data.train_set.num_examples() / BATCH_SIZE)))

	validation_accuracies = []


	# Load the model if it exists
	saver.restore(sess, "mymodel")
	print("Loaded a trained model")
	weights = sess.run(W)

	# Hard coded since I don't want to always wait for loop to run each script
	predictionAccuracyPerHashtag = np.array([
-2,-1, 0,-2, 0,-1,-1, 3, 2,-1,-2, 0,
   -4,-3, 0, 0,-1,-2, 3,-1,-1,-1,-1, 0,
   -1,  -12,-1,-2,-5,-8,-1, 1, 0, 0, 0,-2,
    1,-2,-1,-1, 1,-2, 2,-1, 0, 0,-1, 1,
   -2,-4,-1,35,-3, 0, 0, 0,-1,-1,-8,-3,
    3,-8, 1, 1,-1,-2, 0, 2,-6,-1, 0, 0,
    6,-2, 0, 0,-6,-3,14, 0,-4, 1,-1, 1,
    0, 1,-3, 0,-1,-1, 0,-2, 2, 3,-1,-2,
   -1, 9, 0, 0,-1, 4,-4,-4, 0, 0, 0,-6,
    0, 0, 0, 0, 0, 0,-1,-3, 0,-8,-2, 0,
   -1, 0, 1,-1,-2, 137, 68, 0,-1, 0,-2, 0,
   -2, 0, 0,-1,-1, 0, 0, 4, 0, 0, 0, 0,
    0,-2, 0,-2,-1, 0,-1, 0, 196,-2,-1, 0,
    0, 0,-3, 3, 1, 0,-1, 0, 0,-1, 5,-8,
   -1, 0,-1,-1,-4, 0, 0, 0,-1,-3,-1,-2,
    0, 3,-2, 0,-4, -21, 0,-5, 0,-3, 2,-1,
   -2, 1,-2,-8,-1, 0, 1,-2, 0,-3, 0,-1,
    0, 0,-4,-4, 0,-4,-5,-1, 3,-1,-4,-2,
    0, 2,-2,-3,-2,-1, 0,-1,-1,-2,-6, 0,
    1, 0,-5,-2,-2,-2,-2, 0, 0,-3,-3, 0,
    1, 3,-2, 1,-3, 0, 0,-1,-1,-1, 2, 0,
   -3, 0, 0, 9,-1, 0, 0,-1,-1, 0,46,-1,
   -2,-2,-3,-1, 0, 0,-2,-2, 2,-3,-3,-1,
    1, 0,-4, 1,-1, 1, 1,-4,-2,-1,-4,-1,
    0,-4, 1,-1, 0,-1,-3, 0,-3, 0, 9, 0,
    0, 1, 3,-2, 0,-3, 0,-2,-3,-2, 1,-1,
  -10,-1,-1,-1, 2,-1,-3, 0,-1, 2,-1, 1,
   -1, 0,-4, 3,-1, 0, 0,-2,-1,-2, 0, 3,
   -4, 0, 4,-1,-1, 2, 0, 0,-1,-1,-3, 0,
   -4, 0,-1, 0,-5,-3, 0,-3, 0, 1, 2,-1,
    7,-6,-1, 0,-1,-1,-1,-1,-3, 0, 0,-2,
   -2,-1, 0,-2, 4,-3, 0,-2,-1, 0, 0,-1,
    0,-1,-4, 0, 0,-4, 0,-1,-1,-4,-3,-6,
   -4, 0,-1,-2, 0, 2,-6,-1, 0, 0, 0, 0,
   -1,-3,-1,-1, 0,-1,-3, 0,-7,-1,-3,-4,
    0, 0, 0,-2,-1,-1,-1, 1, 0,-2, 0, 0,
   -4, 0, 0, 0, 0, 1, 1,-1,-2,-1,-1,-2,
    0, 0,-2, 2, 0, 0,-2, 1,-2,-1,-2,-4,
    0, 0, 0, 0,-1, 2, 0,-5, 9,-1,-1,-1,
    1,-1, 0,-1, 0,-3,-1, 4,-4, 0, 0, 2,
    3, 1,-3,-1, 0,-1,-4,-3, 2,-2,-1,-2,
    0,-2, 1,-2, 0, 0, 0,-3])

	mostConsistentlyCorrectHashtagsIndexes = predictionAccuracyPerHashtag.argsort()[-5:][::-1]
	mostCorrectHashtags = getWordsWithIndexes(mostConsistentlyCorrectHashtagsIndexes, hashtag_vocabulary)
	randomIndexes = [randint(0, len(predictionAccuracyPerHashtag)-1) for i in range(5)]
	randomHashtags = getWordsWithIndexes(randomIndexes, hashtag_vocabulary)

	print("Most correct hashtags:")
	print mostCorrectHashtags

	print("random hashtags:")
	print randomHashtags

	for i in range(len(data.test_set.inputs())):

		# correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
		# one_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		# one_accuracy = sess.run(one_accuracy, feed_dict={x: data.test_set.inputs(), y: data.test_set.targets()})
		#
		# print(one_accuracy)
		# exit()


		is_correct_prediction = False
		tweet = getWords(data.test_set.inputs()[i], tweet_vocabulary)
		hashtags = getWords(data.test_set.targets()[i], hashtag_vocabulary)

		if len(set(randomHashtags).intersection(set(hashtags))) == 0:
			if len(set(mostCorrectHashtags).intersection(set(hashtags))) == 0:
				continue

		predictions_probabilities = sess.run(activation, feed_dict={x: [data.test_set.inputs()[i]]})
		top_prediction_probability_indexes = predictions_probabilities[0].argsort()[-5:][::-1]
		predicted_hashtags = getWordsWithIndexes(top_prediction_probability_indexes, hashtag_vocabulary)

		print raw_test_input[i]

		# Check if prediction was correct:
		for index in top_prediction_probability_indexes:
			if hashtag_vocabulary[index] in hashtags:
				is_correct_prediction = True
				predictionAccuracyPerHashtag += data.test_set.targets()[i]
				break

		if is_correct_prediction == False:
			predictionAccuracyPerHashtag -= data.test_set.targets()[i]

		print("This is a {} prediction\t The correct hashtag(s) are: {}".format(is_correct_prediction, hashtags))
		output_str = ''
		for word in tweet:
			output_str += '\t' + word.ljust(10)

		print("Tweet Words: ".ljust(10) + output_str)
		print("Hashtags")

		# Get the weight values
		for j in range(len(predicted_hashtags)):
			output_str = ''
			for k in range(len(data.test_set.inputs()[i])):
				if data.test_set.inputs()[i][k] > 0:
					output_str += '\t' + '%.2f'.ljust(10) % weights[k][top_prediction_probability_indexes[j]]
			print(predicted_hashtags[j].ljust(10) + output_str)

		print("Correct Hashtags")
		for j in range(len(hashtags)):
			hashtag_index = hashtag_vocabulary.index(hashtags[j])
			output_str = ''
			for k in range(len(data.test_set.inputs()[i])):
				if data.test_set.inputs()[i][k] > 0:
					output_str += '\t' + '%.2f'.ljust(10) % weights[k][hashtag_index]
			print(hashtags[j].ljust(10) + output_str)

		print("\n")

	exit()

	print("ONE PREDICTION \t TRAIN \t VALID \t TEST \t COST \t EPOCH")

	for epoch in range(EPOCHS):
		avg_cost = 0.0
		num_batches = int(data.train_set.num_examples() / BATCH_SIZE)

		if epoch == 3:
			cost = -tf.reduce_sum(activation*y + (1-activation)*(1-y))
			activation = tf.nn.sigmoid(tf.matmul(x,W) + b)

		for batch in range(num_batches):
			batch_xs, batch_ys = data.train_set.next_batch(BATCH_SIZE)
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / num_batches

		if epoch % DISPLAY_STEP == 0:

			# Training set accuracy on one hashtag:
			correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
			one_accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			one_accuracy = sess.run(one_accuracy, feed_dict={x: data.train_set.inputs(), y: data.train_set.targets()})

			# Training Set Accuracy
			num_correct = 0
			predictions = sess.run(activation, feed_dict={x: data.train_set.inputs()})
			targets = data.train_set.targets()
			for i in range(len(predictions)):
				top_prediction_indexes = predictions[i].argsort()[-PREDICTION_RANGE:][::-1]

				for index in top_prediction_indexes:
					if targets[i][index]:
						num_correct += 1
						break

			train_accuracy = float(num_correct) / float(len(predictions))

			# Validation Set Accuracy
			num_correct = 0
			predictions = sess.run(activation, feed_dict={x: data.validation_set.inputs()})
			targets = data.validation_set.targets()
			for i in range(len(predictions)):
				top_prediction_indexes = predictions[i].argsort()[-PREDICTION_RANGE:][::-1]

				for index in top_prediction_indexes:
					if targets[i][index]:
						num_correct += 1
						break

			validation_accuracy = float(num_correct) / float(len(predictions))
			validation_accuracies.append(validation_accuracy)

			# Test Set Accuracy
			num_correct = 0
			predictions = sess.run(activation, feed_dict={x: data.test_set.inputs()})
			targets = data.test_set.targets()
			for i in range(len(predictions)):
				top_prediction_indexes = predictions[i].argsort()[-PREDICTION_RANGE:][::-1]

				for index in top_prediction_indexes:
					if targets[i][index]:
						num_correct += 1
						break

			test_accuracy = float(num_correct) / float(len(predictions))

			print("{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {}".format(one_accuracy,
				train_accuracy,
				validation_accuracy,
				test_accuracy,
				avg_cost,
				epoch))


		# If performance is getting worse stop training
		if validation_accuracy < max(validation_accuracies) and epoch > 5:
			continue
			break

	print("Training is done! It took {} minutes to train".format((time.time() - start)/60))
	# Test the model
	correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy of model is predicting 1 hashtag {}".format(sess.run(accuracy, feed_dict={x: data.test_set.inputs(), y: data.test_set.targets()})))

	num_correct = 0
	predictions = sess.run(activation, feed_dict={x: data.test_set.inputs()})
	targets = data.test_set.targets()
	for i in range(len(predictions)):
		top_prediction_indexes = predictions[i].argsort()[-PREDICTION_RANGE:][::-1]

		for index in top_prediction_indexes:
			if targets[i][index]:
				num_correct += 1
				break

	percentage = float(num_correct) / float(len(predictions))
	print("Accuracy of model prediction {} hashtags: num correct = {}, out of {}, percentage = {}".format(PREDICTION_RANGE, num_correct, len(predictions), percentage))


	print(validation_accuracies)

	# Save the model
	print("Saving weights")
	save_path = saver.save(sess, "mymodel")
	print "Model saved in file: ", save_path



main()
