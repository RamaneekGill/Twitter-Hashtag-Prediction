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
	# Weights matrix is len(vocabulary)xlen(hashtags)
	# bias matrix is 1xlen(hashtags)
	
