from __future__ import division
import csv
import random
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


def main():
	import input_data # File responsible for parsing train, test, validation set
	# data contains the 3 different sets, Each a DataSet class
	numWords, numHashtags, data = input_data.read_data_sets()

	LEARNING_RATE = 0.02
	EPOCHS = 200
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
	activation = tf.nn.softmax(tf.matmul(x,W) + b) # the predictions
	# choose between sigmoid and softmax

	# Specify the cost and optimization
	cost = -tf.reduce_sum(y*tf.log(activation)) # cross entropy is the cost function
	# cost = tf.reduce_mean(tf.square(y - activation))
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

	for epoch in range(EPOCHS):
		avg_cost = 0.0
		num_batches = int(data.train_set.num_examples() / BATCH_SIZE)

		for batch in range(num_batches):
			batch_xs, batch_ys = data.train_set.next_batch(BATCH_SIZE)
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / num_batches

		if epoch % DISPLAY_STEP == 0:
			print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

			# Test it against the validation set
			print("Testing against validation set")
			correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
			accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			print("Accuracy on validation set predicting 1 hashtag: {}".format(sess.run(accuracy, feed_dict={x: data.validation_set.inputs(), y: data.validation_set.targets()})))

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
			print("Accuracy on validation set prediction out of 5 hashtags: {}".format(validation_accuracy))

			# Just for kicks print out accuracy on test set
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
