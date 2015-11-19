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

	LEARNING_RATE = 0.01
	EPOCHS = 1000
	BATCH_SIZE = 100
	DISPLAY_STEP = 10 # To print results every n number of epochs

	# Setup the model
	x = tf.placeholder("float", [None, numWords]) # the inputs, this is hydrated with batches
	y = tf.placeholder("float", [None, numHashtags]) # the correct answers
	W = tf.Variable(tf.zeros([numWords, numHashtags])) # weights matrix
	b = tf.Variable(tf.zeros(numHashtags)) # bias
	activation = tf.nn.softmax(tf.matmul(x,W) + b) # the predictions
	# TODO: use sigmoid instead of softmax
	# tf.sigmoid(x, name=None)
	# tf.nn.softmax(logits, name=None)

	# Specify the cost and optimization
	cost = -tf.reduce_sum(y*tf.log(activation)) # cross entropy is the cost function
	optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

	# Initialize session and its veriables
	init = tf.initialize_all_variables()
	sess = tf.Session()
	print("Starting tensorflow session")
	sess.run(init)
	print("Ran tensorflow session")

	# Train the model
	for epoch in range(EPOCHS):
		avg_cost = 0.0
		num_batches = data.train.num_examples / BATCH_SIZE

		for batch in range(num_batches):
			batch_xs, batch_ys = data.train.next_batch(BATCH_SIZE)
			sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
			avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / num_batches

		if epoch % display_step == 0:
			print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

	print "Training is done!"

	# Test the model
	correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print sess.run(accuracy, feed_dict={x: data.test.input, y: data.test.targets})
	# print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

main()
