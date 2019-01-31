# coding: utf-8
###
 # @file   mnistAttack.py
 # @author Georgios Damaskinos <georgios.damaskinos@epfl.ch>
 #         Sébastien Rouault <sebastien.rouault@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2019 Georgios DAMASKINOS.
 #
 # Permission is hereby granted, free of charge, to any person obtaining a copy
 # of this software and associated documentation files (the "Software"), to deal
 # in the Software without restriction, including without limitation the rights
 # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 # copies of the Software, and to permit persons to whom the Software is
 # furnished to do so, subject to the following conditions:
 #
 # The above copyright notice and this permission notice shall be included in all
 # copies or substantial portions of the Software.
 #
 # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 # SOFTWARE.
 #
 # @section DESCRIPTION
 #
 # Basic feed-forward NN for MNIST with a built-in, malformed input-based (i.e. poisoning) attack.
###

import numpy
import os
import pathlib
import sys
import tensorflow as tf

import tools

from . import _Experiment, register
from .mnist import MNIST

# ---------------------------------------------------------------------------- #
# MNIST malformed input attack experiment class

class MNISTAttack(MNIST):
  """ MNIST experiment class.
  """
  def _datasets(self, malformed_severity=0):
    """ Lazily build the iterator over both the training and testing dataset.
    malformed_severity (int): 0 => no malformed input
        1 => scaling by 100 + permutation
        2 => scaling by 1000000000000 + permutation
    Returns:
      Training set iterator, testing set iterator
    """
    # Loading dataset
    def keras_to_dataset(inputs, labels, is_train):
      # Transformation
      inputs = numpy.reshape(inputs, (inputs.shape[0], inputs.shape[1] * inputs.shape[2])).astype(numpy.float32) / 255.
      labels = labels.astype(numpy.int32)
      assert len(inputs) == len(labels), "Corrupted MNIST (sub-)dataset: 'inputs' and 'labels' have different lengths"
      # Generation
      def gen():
        for i in range(len(inputs)):
          yield inputs[i], labels[i]
      if is_train:
        batch_size = self.__args["batch-size"]
        prefetch_size = 4 * batch_size
        if prefetch_size > batch_size:
          prefetch_size = batch_size
        return tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32)).prefetch(prefetch_size).shuffle(prefetch_size).batch(batch_size).repeat()
      else:
        batch_size = len(inputs)
        prefetch_size = batch_size
        return tf.data.Dataset.from_generator(gen, (tf.float32, tf.int32)).prefetch(prefetch_size).batch(batch_size).repeat()
    if self.__datasets is None:
      train, test = self.__raw_data

      xs, ys = train
      if malformed_severity == 1:
        for x in numpy.nditer(xs, op_flags=['readwrite']):
          x[...] = -100 * x
      elif malformed_severity == 2:
        for x in numpy.nditer(xs, op_flags=['readwrite']):
          x[...] = -1000000000000 * x
        xs = numpy.random.permutation(xs)
        ys = numpy.random.permutation(ys)

      train = (xs, ys)

      ds_train = keras_to_dataset(*train, True).make_one_shot_iterator().get_next()
      ds_test  = keras_to_dataset(*test, False).make_one_shot_iterator().get_next()
      self.__datasets = ds_train, ds_test
    return self.__datasets

  @staticmethod
  def _inference(dims, inputs):
    """ Build an inference network, does not manage placement.
    Args:
      dims   Requested dimensions
      inputs Input layer
    Returns:
      Logit layer
    """
    hidden = inputs
    length = len(dims)
    for i in range(length - 1):
      with tf.variable_scope("dense_" + str(i + 1), reuse=tf.AUTO_REUSE) as vs:
        with tf.name_scope(vs.original_name_scope):
          dim_in  = dims[i]
          dim_out = dims[i + 1]
          weights = tf.get_variable("weights", shape=(dim_in, dim_out), dtype=tf.float32)
          biases  = tf.get_variable("biases", shape=(dim_out,), dtype=tf.float32)
          if i == length - 2: # Output linear layer
            return tf.matmul(hidden, weights) + biases
          else: # Hidden, dense layer
            hidden = tf.nn.relu(tf.matmul(hidden, weights) + biases)

  def __init__(self, args):
    # Parse key:val arguments
    args = tools.parse_keyval(args, defaults={"batch-size": 32})
    if args["batch-size"] <= 0:
      raise tools.UserException("Cannot make batches of non-positive size")
    with tools.Context("mnist", None):
      print("Loading MNIST dataset...")
      raw_data = tf.keras.datasets.mnist.load_data()
    # Finalization
    self.__args     = args
    self.__raw_data = raw_data
    self.__datasets = None
    self.__cntr_wk  = 0 # Worker instantiation counter
    self.__cntr_ev  = 0 # Evaluator instantiation counter


  def losses(self, device_dataset, device_models, trace=False):
    # Lazy-build dataset on the parameter server (default if no tf.device specified)
    inputs, labels = self._datasets(malformed_severity=2)[0]
    # Model definitions and loss computations
    losses = []
    for device_model in device_models:
      with tf.device(device_model):
        counter = self.__cntr_wk

        if counter == 0:
          inputs, labels = self._datasets(malformed_severity=2)[0]

        self.__cntr_wk += 1
        with tf.name_scope("worker_" + str(counter)) as worker_scope:
          logits = type(self)._inference([784, 100, 10], inputs)
          losses.append(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))
    return losses

  def accuracy(self, device_dataset, device_models, trace=False):
    # Lazy-build dataset on the parameter server (default if no tf.device specified)
    inputs, labels = self._datasets()[1]
    # Model definitions and accuracy computations
    accuracies = []
    for device_model in device_models:
      with tf.device(device_model):
        counter = self.__cntr_ev
        self.__cntr_ev += 1
        with tf.name_scope("evaluator_" + str(counter)) as evaluator_scope:
          logits = type(self)._inference([784, 100, 10], inputs)
          accuracies.append(tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32)))
    return {"top1-X-acc": tf.add_n(accuracies, name="sum_top1Xacc") / float(len(accuracies))}


# ---------------------------------------------------------------------------- #
# Experiment registering

register("mnistAttack", MNISTAttack)
