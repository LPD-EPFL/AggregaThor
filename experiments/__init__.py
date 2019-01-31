# coding: utf-8
###
 # @file   __init__.py
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
 # Base experiment (= model + dataset) class and loading of the local modules.
###

import pathlib

import tools

# ---------------------------------------------------------------------------- #
# Base experiment class

class _Experiment:
  """ Base experiment class.
  """

  def __init__(self, args):
    """ Unimplemented constructor, no graph available at this time.
    Args:
      args Command line argument list
    """
    raise NotImplementedError

  def loss(self, device_dataset, device_models, trace=False):
    """ Build loss tensors on the specified devices, placement on parameter server by default.
    Args:
      device_dataset Dataset device name/function (same instance between calls if same task, i.e. can use 'is')
      device_models  Model device names/functions, one per worker on the associated task
      trace          Whether to add trace prints for every important step of the computations
    Returns:
      List of loss tensors associated with 'device_models'
    """
    raise NotImplementedError

  def accuracy(self, device_dataset, device_model, trace=False):
    """ Build an accuracy tensor on the specified devices, placement on parameter server by default.
    Args:
      device_dataset Dataset device name/function (same instance between calls if same task, i.e. can use 'is')
      device_models  Model device names/functions, one per worker on the associated task
      trace          Whether to add trace prints for every important step of the computations
    Returns:
      Map of metric string name -> aggregated metric tensor associated with 'device_models'
    """
    raise NotImplementedError

# ---------------------------------------------------------------------------- #
# Experiment register and loader

# Register instance
_register   = tools.ClassRegister("experiment")
itemize     = _register.itemize
register    = _register.register
instantiate = _register.instantiate
del _register

# Load all local modules
with tools.Context("experiments", None):
  tools.import_directory(pathlib.Path(__file__).parent, globals())
