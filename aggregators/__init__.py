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
 # Base gradient aggregation rule class (GAR) and loading of the local modules.
###

import pathlib

import tools

# ---------------------------------------------------------------------------- #
# Base gradient aggregation rule class

class _GAR:
  """ Base gradient aggregation rule class.
  """

  def __init__(self, nbworkers, nbbyzwrks, args):
    """ Unimplemented constructor, no graph available at this time.
    Args:
      nbworkers Total number of workers
      nbbyzwrks Declared number of Byzantine workers
      args      Command line argument list
    """
    raise NotImplementedError

  def aggregate(self, gradients):
    """ Build the gradient aggregation operation of the given gradients.
    Args:
      gradients Computed gradient tensors
    Returns:
      Aggregated gradient tensor
    """
    raise NotImplementedError

# ---------------------------------------------------------------------------- #
# GAR script loader

# Register instance
_register   = tools.ClassRegister("GAR")
itemize     = _register.itemize
register    = _register.register
instantiate = _register.instantiate
del _register

# Load all local modules
with tools.Context("aggregators", None):
  tools.import_directory(pathlib.Path(__file__).parent, globals())
