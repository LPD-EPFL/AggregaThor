# coding: utf-8
###
 # @file   krum.py
 # @author Sébastien Rouault <sebastien.rouault@epfl.ch>
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
 # Multi-Krum GAR.
###

import math
import numpy as np
import tensorflow as tf
import warnings

import tools
import native
from . import _GAR, register, deprecated_native

# ---------------------------------------------------------------------------- #
# Krum GAR class

class PYKrumGAR(_GAR):
  """ Full-Python/(deprecated) native Multi-Krum GAR class.
  """

  def _aggregate(self, *gradients):
    """ Aggregate the gradient using the associated (deprecated) native helper.
    Args:
      gradients List of submitted gradients, as numpy arrays
    Returns:
      Aggregated gradient, as a numpy array
    """
    if self.__nbselected == self.__nbworkers:
      # Fast path average
      result = gradients[0]
      for i in range(1, self.__nbworkers):
        result += gradients[i]
      result /= float(self.__nbworkers)
      return result
    else:
      # Compute list of scores
      scores = [list() for i in range(self.__nbworkers)]
      for i in range(self.__nbworkers - 1):
        score = scores[i]
        for j in range(i + 1, self.__nbworkers):
          # With: 0 <= i < j < nbworkers
          distance = deprecated_native.squared_distance(gradients[i], gradients[j])
          if math.isnan(distance):
            distance = math.inf
          score.append(distance)
          scores[j].append(distance)
      nbinscore = self.__nbworkers - self.__nbbyzwrks - 2
      for i in range(self.__nbworkers):
        score = scores[i]
        score.sort()
        scores[i] = sum(score[:nbinscore])
      # Return the average of the m gradients with the smallest score
      pairs = [(gradients[i], scores[i]) for i in range(self.__nbworkers)]
      pairs.sort(key=lambda pair: pair[1])
      result = pairs[0][0]
      for i in range(1, self.__nbselected):
        result += pairs[i][0]
      result /= float(self.__nbselected)
      return result

  def __init__(self, nbworkers, nbbyzwrks, args):
    warnings.warn("Python/native implementation of Krum has been deprecated in favor of the CO implementations", category=DeprecationWarning, stacklevel=3)
    self.__nbworkers  = nbworkers
    self.__nbbyzwrks  = nbbyzwrks
    self.__nbselected = nbworkers - nbbyzwrks - 2

  def aggregate(self, gradients):
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    return tf.py_func(self._aggregate, gradients, gradients[0].dtype, stateful=False, name="GAR_krum")

class TFKrumGAR(_GAR):
  """ Full-TensorFlow Multi-Krum GAR class.
  """

  def __init__(self, nbworkers, nbbyzwrks, args):
    self.__nbworkers  = nbworkers
    self.__nbbyzwrks  = nbbyzwrks
    self.__nbselected = nbworkers - nbbyzwrks - 2

  def aggregate(self, gradients):
    with tf.name_scope("GAR_krum_tf"):
      # Assertion
      assert len(gradients) > 0, "Empty list of gradient to aggregate"
      # Distance computations
      distances = []
      for i in range(self.__nbworkers - 1):
        dists = list()
        for j in range(i + 1, self.__nbworkers):
          sqr_dst = tf.reduce_sum(tf.squared_difference(gradients[i], gradients[j]))
          dists.append(tf.negative(tf.where(tf.is_finite(sqr_dst), sqr_dst, tf.constant(np.inf, dtype=sqr_dst.dtype)))) # Use of 'negative' to get the smallest distances and score indexes in 'nn.top_k'
        distances.append(dists)
      # Score computations
      scores = []
      for i in range(self.__nbworkers):
        dists = []
        for j in range(self.__nbworkers):
          if j == i:
            continue
          if j < i:
            dists.append(distances[j][i - j - 1])
          else:
            dists.append(distances[i][j - i - 1])
        dists = tf.parallel_stack(dists)
        dists, _ = tf.nn.top_k(dists, k=(self.__nbworkers - self.__nbbyzwrks - 2), sorted=False)
        scores.append(tf.reduce_sum(dists))
      # Average of the 'nbselected' smallest scoring gradients
      gradients = tf.parallel_stack(gradients)
      scores = tf.parallel_stack(scores)
      _, indexes = tf.nn.top_k(scores, k=self.__nbselected, sorted=False)
      return tf.reduce_mean(tf.gather(gradients, indexes), axis=0)

class COKrumGAR(_GAR):
  """ Full-custom operation Multi-Krum GAR class.
  """

  # Name of the associated custom operation
  co_name = "krum"

  def __init__(self, nbworkers, nbbyzwrks, args):
    self.__nbworkers  = nbworkers
    self.__nbbyzwrks  = nbbyzwrks
    self.__nbselected = nbworkers - nbbyzwrks - 2

  def aggregate(self, gradients):
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    return native.instantiate_op(type(self).co_name, tf.parallel_stack(gradients), f=self.__nbbyzwrks, m=self.__nbselected)

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rules
register("krum-py", PYKrumGAR)
register("krum-tf", TFKrumGAR)
if COKrumGAR.co_name in native.itemize_op():
  register("krum-co", COKrumGAR)
else:
  tools.warning("GAR 'krum-co' could not be registered since the associated custom operation " + repr(COKrumGAR.co_name) + " is unavailable")
