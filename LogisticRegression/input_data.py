"""Functions used for parsing tweet data"""

import numpy
from numpy import *
import numpy as np
import time

def create_vocabulary(dataset):
    vocabulary = []
    total_words = 0
    for words in dataset:
        vocabulary += words
        total_words += len(words)
    vocabulary = list(set(vocabulary))

    print ("Compressed %d words into a vocabulary of size %d !" % (total_words, len(vocabulary)))

    return vocabulary

def convert_to_one_hot_vector(vector, words, vocabulary):
    for i in range(len(words)):
        index = vocabulary.index(words[i])
        vector[index] = 1.0

    return vector

def create_matrix(dataset, vocabulary):
    matrix = np.zeros((len(dataset), len(vocabulary)))
    print("Creating a %d x %d matrix" % (matrix.shape[0], matrix.shape[1]))
    for i in range(len(matrix)):
        if i % 100 == 0:
            print("On the %dth row out of %d" % (i, matrix.shape[0]))
        # matrix[i] is the input vector, dataset[i] are the words of the input
        matrix[i] = convert_to_one_hot_vector(matrix[i], dataset[i], vocabulary)

    return matrix

class DataSet:
    def __init__(self, inputs, targets):
        # Make sure inputs and targets have same number of data points
        assert inputs.shape[0] == targets.shape[0]
        self._num_examples = inputs.shape[0]

        self._inputs = inputs
        self._targets = targets
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def inputs(self):
        return self._inputs

    def targets(self):
        return self._targets

    def num_examples(self):
        return self._num_examples

    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size = 100):
        CONST_RANDOM_SEED = 20150819
        np.random.seed(CONST_RANDOM_SEED)

        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._inputs = self._inputs[perm]
            self._targets = self._targets[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples

        end = self._index_in_epoch

        return self._inputs[start:end], self._targets[start:end]


def read_data_sets(debug_mode=False):
  class DataSets(object):
    pass

  data_sets = DataSets()
  start = time.time()

  # Read csv and get inputs and targets for train, validation, and test set
  import parse_csv
  train_inputs, train_targets, validation_inputs, validation_targets, test_inputs, test_targets = parse_csv.read_dataset()

  print("Training set size: %d \t Validation set size: %d \t Test set size: %d" % (len(train_inputs), len(validation_inputs), len(test_inputs)))

  print('creating vocabularies')
  tweet_vocabulary = create_vocabulary(train_inputs + validation_inputs + test_inputs)
  hashtag_vocabulary = create_vocabulary(train_targets + validation_targets + test_targets)

  print('creating matrixes')
  train_inputs = create_matrix(train_inputs[:int(0*len(train_inputs))], tweet_vocabulary).astype(np.int32)
  print('finished train_inputs')
  train_targets = create_matrix(train_targets[:int(0*len(train_targets))], hashtag_vocabulary).astype(np.int32)
  print('finished train_targets')
  validation_inputs = create_matrix(validation_inputs[:int(0*len(validation_inputs))], tweet_vocabulary).astype(np.int32)
  print('finished validation_inputs')
  validation_targets = create_matrix(validation_targets[:int(0*len(validation_targets))], hashtag_vocabulary).astype(np.int32)
  print('finished validation_targets')
  test_inputs = create_matrix(test_inputs[:int(1*len(test_inputs))], tweet_vocabulary).astype(np.int32)
  print('finished test_inputs')
  test_targets = create_matrix(test_targets[:int(1*len(test_targets))], hashtag_vocabulary).astype(np.int32)
  print('finished test_targets')

  data_sets.train_set = DataSet(train_inputs, train_targets)
  data_sets.validation_set = DataSet(validation_inputs, validation_targets)
  data_sets.test_set = DataSet(test_inputs, test_targets)

  print('Finished setting up data! Took {} seconds'.format(time.time() - start))
  test = DataSet(train_inputs, train_targets)

  return len(tweet_vocabulary), len(hashtag_vocabulary), data_sets, tweet_vocabulary, hashtag_vocabulary
