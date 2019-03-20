# coding: utf-8
###
 # @file   slims.py
 # @author Sébastien Rouault <sebastien.rouault@epfl.ch>
 # @author Georgios Damaskinos <georgios.damaskinos@epfl.ch>
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
 # TF-Slim's networks and datasets automated loading and wrapping.
###

import os
import pathlib
import sys
import tensorflow as tf
import traceback

import tools

from . import _Experiment, register

# ---------------------------------------------------------------------------- #
# Slim experiment class

class SlimExperiment(_Experiment):
  """ Slim experiment class.
  """

  @staticmethod
  def _make(dataset, model):
    """ Wrap a slim constructor with a model name and dataset name.
    Args:
      dataset (Dataset name to bind, dataset directory)
      model   Model name to bind
    Returns:
      Bound slim constructor
    """
    def wrapper(args):
      return SlimExperiment(dataset, model, args)
    return wrapper

  def __init__(self, dataset, model, args):
    # Parse key:val arguments
    nbcores = len(os.sched_getaffinity(0))
    if nbcores == 0:
      nbcores = 4 # Arbitrary fallback
    args = tools.parse_keyval(args, defaults={
      "batch-size": 32,
      "eval-batch-size": 1024,
      "weight-decay": 0.00004,
      "label-smoothing": 0.,
      "labels-offset": 0,
      "nb-fetcher-threads": nbcores,
      "nb-batcher-threads": nbcores })
    if args["batch-size"] <= 0:
      raise tools.UserException("Cannot make batches of non-positive size")
    # Report experiments
    with tools.Context("slim", None):
      print("Dataset name in use:   " + repr(dataset[0]) + " (in " + repr(dataset[1]) + ")")
      print("Dataset preprocessing: " + (repr(args["preprocessing"]) if "preprocessing" in args else "<model default>"))
      print("Model name in use:     " + repr(model))
    # Finalization
    self.__args    = args
    self.__dataset = dataset
    self.__preproc = args["preprocessing"] if "preprocessing" in args else model
    self.__model   = model
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
        dataset    = dataset_factory.get_dataset(self.__dataset[0], "train", self.__dataset[1])
        preproc_fn = preprocessing_factory.get_preprocessing(self.__preproc, is_training=True)
        network_fn = nets_factory.get_network_fn(self.__model, num_classes=(dataset.num_classes - self.__args["labels-offset"]), weight_decay=self.__args["weight-decay"], is_training=True)
      # Create a dataset provider that loads data from the dataset
      with tf.device(device_dataset):
        provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers=self.__args["nb-fetcher-threads"], common_queue_capacity=(20 * self.__args["batch-size"]), common_queue_min=(10 * self.__args["batch-size"]))
        [image, label] = provider.get(["image", "label"])
        label -= self.__args["labels-offset"]
        image = preproc_fn(image, network_fn.default_image_size, network_fn.default_image_size)
        images, labels = tf.train.batch([image, label], batch_size=self.__args["batch-size"], num_threads=self.__args["nb-batcher-threads"], capacity=(5 * self.__args["batch-size"]))
        labels = tf.contrib.slim.one_hot_encoding(labels, dataset.num_classes - self.__args["labels-offset"])
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
            logits, end_points = network_fn(images)
            if "AuxLogits" in end_points:
              tf.contrib.slim.losses.softmax_cross_entropy(end_points["AuxLogits"], labels, label_smoothing=self.__args["label-smoothing"], weights=0.4, scope="aux_loss")
            tf.contrib.slim.losses.softmax_cross_entropy(logits, labels, label_smoothing=self.__args["label-smoothing"], weights=1.0)
            losses.append(tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES, worker_scope), name="sum_loss"))
    return losses

  def accuracy(self, device_dataset, device_models, trace=False):
    # Dataset management
    splitname = "validation" if self.__dataset[0] in ["imagenet", "flowers"] else "test"
    dataset_scope = "dataset_" + splitname + "_" + str(self.__cntr_ev)
    if len(device_models) > 1:
      dataset_scope += "-" + str(self.__cntr_ev + len(device_models) - 1)
    with tf.name_scope(dataset_scope):
      # Select dataset and preprocessing, network functions
      with tf.device(device_dataset):
        dataset    = dataset_factory.get_dataset(self.__dataset[0], splitname, self.__dataset[1])
        preproc_fn = preprocessing_factory.get_preprocessing(self.__preproc, is_training=False)
        network_fn = nets_factory.get_network_fn(self.__model, num_classes=(dataset.num_classes - self.__args["labels-offset"]), weight_decay=self.__args["weight-decay"], is_training=False)
      # Create a dataset provider that loads data from the dataset
      with tf.device(device_dataset):
        provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(dataset, num_readers=self.__args["nb-fetcher-threads"], common_queue_capacity=(4 * self.__args["eval-batch-size"]), common_queue_min=(2 * self.__args["eval-batch-size"]))
        [image, label] = provider.get(["image", "label"])
        label -= self.__args["labels-offset"]
        image = preproc_fn(image, network_fn.default_image_size, network_fn.default_image_size)
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
            logits, _ = network_fn(images)
            accuracies.append(tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32)))
    return {"top1-X-acc": tf.add_n(accuracies, name="sum_top1Xacc") / float(len(accuracies))}

# ---------------------------------------------------------------------------- #
# Experiment registering

# (Try to) import slim package
with tools.ExpandPath(pathlib.Path(__file__).parent / "slim"):
  from .                       import slim
  from .slim.datasets          import dataset_factory
  from .slim.preprocessing     import preprocessing_factory
  from .slim.nets              import nets_factory
  from .slim.nets.nets_factory import networks_map

# List available models
models = list(networks_map.keys())
if len(models) == 0:
  raise tools.UserException("no model available in slim package")

# List available datasets
datasets = dict()
dspath   = pathlib.Path(__file__).parent / "datasets"
if not dspath.is_dir():
  raise tools.UserException("slim dataset at 'datasets' must be a directory")
for path in dspath.iterdir():
  if not tools.can_access(path, read=True):
    with tools.Context(None, "warning"):
      print("slim dataset " + repr(path.name + "/*") + " in 'datasets' is not read-able and has been ignored")
    continue
  if not path.is_dir(): # Must be after to first check for access rights...
    continue
  datasets[path.name] = str(path)
if len(datasets) == 0:
  raise tools.UserException("no dataset available in slim package")

# Register cross-product models-datasets
for model in models:
  for dspair in datasets.items():
    register("slim-" + model + "-" + dspair[0], SlimExperiment._make(dspair, model))
