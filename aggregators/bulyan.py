# coding: utf-8
###
 # @file   bulyan.py
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
 # Bulyan over Multi-Krum GAR.
###

import tensorflow as tf
import warnings

import tools
import native
from . import _GAR, register, deprecated_native

# ---------------------------------------------------------------------------- #
# Bulyan GAR class

class PYBulyanGAR(_GAR):
  """ Full-Python/(deprecated) native Bulyan of Multi-Krum GAR class.
  """

  def _aggregate(self, gradients):
    """ Aggregate the gradient using the associated (deprecated) native helper.
    Args:
      gradients Stacked list of submitted gradients, as a numpy array
    Returns:
      Aggregated gradient, as a numpy array
    """
    return deprecated_native.bulyan(gradients, self.__f, self.__s)

  def __init__(self, nbworkers, nbbyzwrks, args):
    warnings.warn("Python/native implementation of Bulyan has been deprecated in favor of the CO implementations", category=DeprecationWarning, stacklevel=3)
    self.__f = nbbyzwrks
    self.__s = nbworkers - 2 * nbbyzwrks - 2

  def aggregate(self, gradients):
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    gradients = tf.parallel_stack(gradients)
    return tf.py_func(self._aggregate, [gradients], gradients.dtype, stateful=False, name="GAR_bulyan")

class COBulyanGAR(_GAR):
  """ Full-custom operation Bulyan of Multi-Krum GAR class.
  """

  # Name of the associated custom operation
  co_name = "bulyan"

  def __init__(self, nbworkers, nbbyzwrks, args):
    self.__nbworkers = nbworkers
    self.__nbbyzwrks = nbbyzwrks
    self.__multikrum = nbworkers - nbbyzwrks - 2

  def aggregate(self, gradients):
    # Assertion
    assert len(gradients) > 0, "Empty list of gradient to aggregate"
    # Computation
    return native.instantiate_op(type(self).co_name, tf.parallel_stack(gradients), f=self.__nbbyzwrks, m=self.__multikrum)

# ---------------------------------------------------------------------------- #
# GAR registering

# Register aggregation rules
register("bulyan-py", PYBulyanGAR)
if COBulyanGAR.co_name in native.itemize_op():
  register("bulyan-co", COBulyanGAR)
else:
  tools.warning("GAR 'bulyan-co' could not be registered since the associated custom operation " + repr(COBulyanGAR.co_name) + " is unavailable")
