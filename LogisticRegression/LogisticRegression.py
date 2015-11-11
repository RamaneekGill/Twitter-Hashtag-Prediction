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

	# Read dataset

	# Parse in to training, test, validation input sets

	# Parse in to training, test, validation target output sets

	# Create a dictionary of the words in the training input and output set
		# Remember to remove stop words and punctuation
		# Remember to strip out the '#' and hashtag from the input

	# Create a dictionary of all the words in the training set
		# Remember to remove stop words and punctuation first

	# Create a dictionary of all the hashtags (target outputs) from training set
		# Remember to remove the '#'

	# Add the keys from hashtag dictionary to the vocabulary dictionary

	# Input matrix is 1xlen(vocabulary)
		# Should be a 0 if word not present, 1 if present
	# Weights matrix is len(vocabulary)xlen(hashtags)
	# bias matrix is 1xlen(hashtags)
	# prediction = Weights * input + b
################################################################################

	import input_data # File responsible for parsing train, test, validation set
	# data contains the 3 different sets, Each a DataSet class
	data = input_data.read_data_sets()

	LEARNING_RATE = 0.01

	vocabulary = data.train.vocabulary
	numWords = len(vocabulary.keys())
	hashtags = data.train.hashtags
	numHashtags = len(hashtags.keys())

	x = tf.placeholder("float", [None, numWords]) # the inputs, this is hydrated with batches
	y_ = tf.placeholder("float", [None, 10]) # the correct answers
	W = tf.Variable(tf.zeros([numWords, numHashtags])) # weights matrix
	b = tf.Variable(tf.zeros(numHashtags)) # bias
	y = tf.nn.softmax(tf.matmul(x,W) + b) # the predictions

	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)


############################# Reference Code ###################################
	import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	# NOTE: Find out what mnist is below:
	# https://tensorflow.googlesource.com/tensorflow/+/master/tensorflow/g3doc/tutorials/mnist/input_data.py

	x = tf.placeholder("float", [None, 784]) # None means can be any length
	W = tf.Variable(tf.zeros([784,10]))
	b = tf.Variable(tf.zeros([10]))
	y = tf.nn.softmax(tf.matmul(x,W) + b) # the predictions
	y_ = tf.placeholder("float", [None,10]) # the correct answers
	cross_entropy = -tf.reduce_sum(y_*tf.log(y))
	train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

	init = tf.initialize_all_variables()
	sess = tf.Session()
	sess.run(init)

	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
