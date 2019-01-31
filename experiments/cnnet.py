# coding: utf-8
###
 # @file   cnnet.py
 # @author Sébastien Rouault <sebastien.rouault@epfl.ch>
 #         Georgios Damaskinos <georgios.damaskinos@epfl.ch>
 #
 # @section LICENSE
 #
 # Copyright © 2018-2019 Sébastien ROUAULT.
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
 # Custom convolution network for CIFAR-10/100, using TF-Slim's dataset helpers.
###

import os
import pathlib
import sys
import tensorflow as tf
import traceback

import tools

from . import _Experiment, register

# ---------------------------------------------------------------------------- #
# Dataset directories

# CIFAR-10 dataset expected name and directory
dataset_name      = "cifar10"
dataset_directory = None

# ---------------------------------------------------------------------------- #
# CNN experiment class

class CNNetExperiment(_Experiment):
  """ Simple CNN experiment class.
  """

  @staticmethod
  def __network_fn(images):
    """ Build the inference layer of the experience.
    Args:
      images Input image
    Returns:
      Infered logits
    """
    # Convolutional + max pool layer 1
    kernel = tf.get_variable(name="conv1_weights", shape=[5, 5, 3, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2))
    feed   = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding="SAME")
    biases = tf.get_variable(name="conv1_biases", shape=[64], initializer=tf.constant_initializer(0.0))
    feed   = tf.nn.relu(tf.nn.bias_add(feed, biases), name="conv1")
    feed   = tf.nn.max_pool(feed, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")
    # Convolutional + max pool layer 2
    kernel = tf.get_variable(name="conv2_weights", shape=[5, 5, 64, 64], initializer=tf.truncated_normal_initializer(stddev=5e-2))
    feed   = tf.nn.conv2d(feed, kernel, strides=[1, 1, 1, 1], padding="SAME")
    biases = tf.get_variable(name="conv2_biases", shape=[64], initializer=tf.constant_initializer(0.1))
    feed   = tf.nn.relu(tf.nn.bias_add(feed, biases), name="conv2")
    feed   = tf.nn.max_pool(feed, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")
    # Flatten feature maps
    shape = feed.get_shape().as_list()
    feed  = tf.reshape(feed, [-1, shape[1] * shape[2] * shape[3]])
    dim   = feed.get_shape()[1].value
    # Dense layer 3
    weights = tf.get_variable(name="dense3_weights", shape=[dim, 384], initializer=tf.truncated_normal_initializer(stddev=0.04))
    biases  = tf.get_variable(name="dense3_biases", shape=[384], initializer=tf.constant_initializer(0.1))
    feed    = tf.nn.relu(tf.matmul(feed, weights) + biases, name="dense3")
    # Dense layer 4
    weights = tf.get_variable(name="dense4_weights", shape=[384, 192], initializer=tf.truncated_normal_initializer(stddev=0.04))
    biases  = tf.get_variable(name="dense4_biases", shape=[192], initializer=tf.constant_initializer(0.1))
    feed    = tf.nn.relu(tf.matmul(feed, weights) + biases, name="dense4")
    # Linear, output layer
    weights = tf.get_variable(name="linear5_weights", shape=[192, 10], initializer=tf.truncated_normal_initializer(stddev=1/192.0))
    biases  = tf.get_variable(name="linear5_biases", shape=[10], initializer=tf.constant_initializer(0.0))
    logits  = tf.add(tf.matmul(feed, weights), biases, name="linear5")
    # Return the infered logits
    return logits

  def __init__(self, args):
    # Parse key:val arguments
    nbcores = len(os.sched_getaffinity(0))
    if nbcores == 0:
      nbcores = 4 # Arbitrary fallback
    args = tools.parse_keyval(args, defaults={
      "batch-size": 32,
      "eval-batch-size": 1024,
      "nb-fetcher-threads": nbcores,
      "nb-batcher-threads": nbcores })
    if args["batch-size"] <= 0:
      raise tools.UserException("Cannot make batches of non-positive size")
    # Finalization
    self.__args    = args
    self.__preproc = args["preprocessing"] if "preprocessing" in args else "cifarnet"
    self.__cntr_wk = 0 # Worker instantiation counter
    self.__cntr_ev = 0 # Evaluator instantiation counter

  def losses(self, device_dataset, device_models, trace=False):
    # Dataset management
    dataset_scope = "dataset_train_" + str(self.__cntr_wk)
    if len(device_models) > 1:
      dataset_scope += "-" + str(self.__cntr_wk + len(device_models) - 1)
    with tf.name_scope(dataset_scope):
      # Select dataset and preprocessing, network functions
      with tf.device(device_dataset):
        dataset    = dataset_factory.get_dataset(dataset_name, "train", dataset_directory)
        preproc_fn = preprocessing_factory.get_preprocessing(self.__preproc, is_training=True)
      # Create a dataset provider that loads data from the dataset
      with tf.device(device_dataset):
        provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers=self.__args["nb-fetcher-threads"], common_queue_capacity=(20 * self.__args["batch-size"]), common_queue_min=(10 * self.__args["batch-size"]))
        [image, label] = provider.get(["image", "label"])
        image = preproc_fn(image, 32, 32)
        images, labels = tf.train.batch([image, label], batch_size=self.__args["batch-size"], num_threads=self.__args["nb-batcher-threads"], capacity=(5 * self.__args["batch-size"]))
        # labels = one_hot_encoding...
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images, labels], capacity=(2 * len(device_models)))
    # Model definitions and loss computations
    losses = []
    with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
      for device_model in device_models:
        with tf.device(device_model):
          counter = self.__cntr_wk
          self.__cntr_wk += 1
          with tf.name_scope("worker_" + str(counter)) as worker_scope:
            images, labels = batch_queue.dequeue()
            logits = type(self).__network_fn(images)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits), name="loss")
            tf.add_to_collection(tf.GraphKeys.LOSSES, loss)
            losses.append(loss)
    return losses

  def accuracy(self, device_dataset, device_models, trace=False):
    # Dataset management
    dataset_scope = "dataset_test_" + str(self.__cntr_ev)
    if len(device_models) > 1:
      dataset_scope += "-" + str(self.__cntr_ev + len(device_models) - 1)
    with tf.name_scope(dataset_scope):
      # Select dataset and preprocessing, network functions
      with tf.device(device_dataset):
        dataset    = dataset_factory.get_dataset(dataset_name, "test", dataset_directory)
        preproc_fn = preprocessing_factory.get_preprocessing(self.__preproc, is_training=False)
      # Create a dataset provider that loads data from the dataset
      with tf.device(device_dataset):
        provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers=self.__args["nb-fetcher-threads"], common_queue_capacity=(4 * self.__args["eval-batch-size"]), common_queue_min=(2 * self.__args["eval-batch-size"]))
        [image, label] = provider.get(["image", "label"])
        image = preproc_fn(image, 32, 32)
        images, labels = tf.train.batch([image, label], batch_size=self.__args["eval-batch-size"], num_threads=self.__args["nb-batcher-threads"], capacity=(2 * self.__args["eval-batch-size"]))
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images, labels], capacity=(2 * len(device_models)))
    # Model definitions and accuracy computations
    accuracies = []
    with tf.variable_scope("shared", reuse=tf.AUTO_REUSE):
      for device_model in device_models:
        with tf.device(device_model):
          counter = self.__cntr_ev
          self.__cntr_ev += 1
          with tf.name_scope("evaluator_" + str(counter)) as evaluator_scope:
            images, labels = batch_queue.dequeue()
            logits = type(self).__network_fn(images)
            accuracies.append(tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32)))
    return {"top1-X-acc": tf.add_n(accuracies, name="sum_top1Xacc") / float(len(accuracies))}

# ---------------------------------------------------------------------------- #
# Experiment registering

# (Try to) import slim package
with tools.ExpandPath(pathlib.Path(__file__).parent / "slim"):
  from .                   import slim
  from .slim.datasets      import dataset_factory
  from .slim.preprocessing import preprocessing_factory

# Check whether CIFAR-10 dataset is available
dspath = pathlib.Path(__file__).parent / "datasets" / dataset_name
if not dspath.is_dir():
  raise tools.UserException("slim dataset " + repr(dataset_name) + " in 'datasets' must be a directory")
if not tools.can_access(dspath, read=True):
  raise tools.UserException("slim dataset " + repr(dataset_name + "/*") + " in 'datasets' must be read-able")

# Register dataset directory and experiment
dataset_directory = str(dspath)
register("cnnet", CNNetExperiment)
