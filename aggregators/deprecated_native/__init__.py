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
 # (Deprecated) Native code automated building and loading + python wrappers.
###

import ctypes
import numpy as np
import pathlib
import shlex
import subprocess

import tools

# ---------------------------------------------------------------------------- #

def get_module(lib, src):
  """ (Build then) load the native shared object.
  Args:
    lib Path to the shared object
    src Path to the unique source file
  Returns:
    Module instance
  """
  # Conversion if necessary
  if not isinstance(lib, pathlib.Path):
    lib = pathlib.Path(lib)
  if not isinstance(src, pathlib.Path):
    src = pathlib.Path(src)
  assert src.exists(), "Source file '" + str(src) + "' does not exist"
  # Build if necessary
  if not lib.exists() or src.stat().st_mtime > lib.stat().st_mtime:
    command = shlex.split("c++ -Wall -Wextra -Wfatal-errors -O2 -std=c++14 -fPIC -shared -o " + shlex.quote(str(lib)) + " " + shlex.quote(str(src)))
    command = subprocess.run(command)
    if command.returncode != 0:
      raise tools.UserException("Compilation of '" + str(src.resolve()) + "' failed with error code " + str(command.returncode))
  # Load module
  return ctypes.CDLL(str(lib.resolve()))

# (Build then) load the native shared object
cwd    = pathlib.Path(__file__).parent
module = get_module(cwd / "lib.so", cwd / "native.cpp")

# ---------------------------------------------------------------------------- #

# Return type declaration
module.squared_distance_float.restype  = ctypes.c_float
module.squared_distance_double.restype = ctypes.c_double

def squared_distance(a, b):
  """ Compute the squared l2 distance.
  Args:
    a Selected gradients
    b Coordinates to average
  Returns:
    (a - b)²
  """
  # Function selection
  funcs = {4: module.squared_distance_float, 8: module.squared_distance_double}
  fsize = a.dtype.itemsize
  if fsize not in funcs:
    raise tools.UserException("Unsupported floating point type")
  # Actual call
  dim = ctypes.c_size_t(a.shape[0])
  a   = ctypes.c_void_p(a.ctypes.data)
  b   = ctypes.c_void_p(b.ctypes.data)
  res = funcs[fsize](dim, a, b)
  # Return computed scalar
  return res

# ---------------------------------------------------------------------------- #

def median(inputs):
  """ Compute the median coordinate by coordinate.
  Args:
    inputs Input gradients
  Returns:
    Median coordinate by coordinate
  """
  # Function selection
  funcs = {4: module.median_float, 8: module.median_double}
  fsize = inputs.dtype.itemsize
  if fsize not in funcs:
    raise tools.UserException("Unsupported floating point type")
  # Actual call
  dim = ctypes.c_size_t(inputs.shape[1])
  n   = ctypes.c_size_t(inputs.shape[0])
  ins = ctypes.c_void_p(inputs.ctypes.data)
  out = np.empty_like(inputs[0])
  funcs[fsize](dim, n, ins, ctypes.c_void_p(out.ctypes.data))
  # Return computed gradient
  return out

def averaged_median(inputs, beta):
  """ Compute the median coordinate by coordinate.
  Args:
    inputs Input gradients
    beta   Number of averaged coordinates
  Returns:
    Averaged median coordinate by coordinate
  """
  # Function selection
  funcs = {4: module.averaged_median_float, 8: module.averaged_median_double}
  fsize = inputs.dtype.itemsize
  if fsize not in funcs:
    raise tools.UserException("Unsupported floating point type")
  # Actual call
  dim  = ctypes.c_size_t(inputs.shape[1])
  n    = ctypes.c_size_t(inputs.shape[0])
  beta = ctypes.c_size_t(beta)
  ins  = ctypes.c_void_p(inputs.ctypes.data)
  out  = np.empty_like(inputs[0])
  funcs[fsize](dim, n, beta, ins, ctypes.c_void_p(out.ctypes.data))
  # Return computed gradient
  return out

def average_nan(inputs):
  """ Compute the average coordinate by coordinate, ignoring NaN coordinate.
  Args:
    inputs Input gradients
  Returns:
    Average coordinate by coordinate, ignoring NaN
  """
  # Function selection
  funcs = {4: module.average_nan_float, 8: module.average_nan_double}
  fsize = inputs.dtype.itemsize
  if fsize not in funcs:
    raise tools.UserException("Unsupported floating point type")
  # Actual call
  dim = ctypes.c_size_t(inputs.shape[1])
  n   = ctypes.c_size_t(inputs.shape[0])
  ins = ctypes.c_void_p(inputs.ctypes.data)
  out = np.empty_like(inputs[0])
  funcs[fsize](dim, n, ins, ctypes.c_void_p(out.ctypes.data))
  # Return computed gradient
  return out

# ---------------------------------------------------------------------------- #

def bulyan(inputs, f, s):
  """ Compute Bulyan of Multi-Krum.
  Args:
    inputs Input gradients
    f      Number of byzantine gradients
    s      Number of selected gradients
  Returns:
    Bulyan's output gradient
  """
  # Function selection
  funcs = {4: module.bulyan_float, 8: module.bulyan_double}
  fsize = inputs.dtype.itemsize
  if fsize not in funcs:
    raise tools.UserException("Unsupported floating point type")
  # Actual call
  d   = ctypes.c_size_t(inputs.shape[1])
  n   = ctypes.c_size_t(inputs.shape[0])
  ins = ctypes.c_void_p(inputs.ctypes.data)
  sel = np.empty((s, inputs.shape[1]), dtype=inputs.dtype)
  out = np.empty(inputs.shape[1], dtype=inputs.dtype)
  funcs[fsize](d, n, f, s, ins, ctypes.c_void_p(sel.ctypes.data), ctypes.c_void_p(out.ctypes.data))
  # Return computed gradient
  return out
