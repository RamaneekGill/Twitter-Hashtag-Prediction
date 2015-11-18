"""Functions used for parsing tweet data"""

import numpy
from numpy import *
import numpy as np

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

def vocabulary(dataset):
    vocabulary = []
    for words in dataset:
        vocabulary += words
    vocabulary = list(set(vocabulary))
    return vocabulary

def convert_to_one_hot_vector(vector, words, vocabulary):
    for i in range(len(words)):
        index = vocabulary.index(words[i])
        vector[index] = 1.0

    return vector

def create_matrix(dataset, vocabulary):
    matrix = np.zeros(len(dataset), len(vocabulary))
    for i in range(len(matrix)):
        # matrix[i] is the input vector, dataset[i] are the words of the input
        matrix[i] = convert_to_one_hot_vector(matrix[i], dataset[i], vocabulary)

    return matrix

class DataSet(object):
    def __init__(self, inputs, targets):
        # Make sure inputs and targets have same number of data points
        assert inputs.shape[0] == targets.shape[0]
        self._num_examples = inputs.shape[0]

        self._inputs = inputs
        self._targets = targets
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def inputs(self):
        return self._inputs

    @property
    def targets(self):
        return self._targets

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size

        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epoch_completed += 1
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

        return self._inputs[start:end], self.targets[start:end]


def read_data_sets(train_dir, fake_data=False, one_hot=False):
  class DataSets(object):
    pass
  data_sets = DataSets()

  if fake_data:
    data_sets.train = DataSet([], [], fake_data=True)
    data_sets.validation = DataSet([], [], fake_data=True)
    data_sets.test = DataSet([], [], fake_data=True)
    return data_sets

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
  VALIDATION_SIZE = 5000

  local_file = maybe_download(TRAIN_IMAGES, train_dir)
  train_images = extract_images(local_file)
  local_file = maybe_download(TRAIN_LABELS, train_dir)
  train_labels = extract_labels(local_file, one_hot=one_hot)
  local_file = maybe_download(TEST_IMAGES, train_dir)
  test_images = extract_images(local_file)
  local_file = maybe_download(TEST_LABELS, train_dir)
  test_labels = extract_labels(local_file, one_hot=one_hot)

  validation_images = train_images[:VALIDATION_SIZE]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_images = train_images[VALIDATION_SIZE:]
  train_labels = train_labels[VALIDATION_SIZE:]

  data_sets.train = DataSet(train_images, train_labels)
  data_sets.validation = DataSet(validation_images, validation_labels)
  data_sets.test = DataSet(test_images, test_labels)

  return data_sets
