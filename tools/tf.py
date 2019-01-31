# coding: utf-8
###
 # @file   tf.py
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
 # Miscellaneous TensorFlow helpers.
###

__all__ = ["trace_graph", "device_from_tuple", "Checkpoints"]

import tools

# ---------------------------------------------------------------------------- #
# In-graph tracing

def trace_graph(optn, what):
  """ Trace the beginning and ending of a given operation.
  Args:
    optn Single operation/tensor to trace (not all operations can be traced)
    what Text describing the nature of the operations
  Returns:
    Identity/group of the given tensor/operation
  """
  import tensorflow as tf
  dummy_tn = tf.constant(0, name="print_dummy")
  begin_tn = tf.Print(dummy_tn, ["[TRACE] (begin) " + what])
  with tf.control_dependencies([begin_tn, optn]):
    end_tn = tf.Print(dummy_tn, ["[TRACE] (end)   " + what])
    with tf.control_dependencies([end_tn]):
      if isinstance(optn, tf.Operation):
        return tf.group(optn)
      else:
        return tf.identity(optn, name="print_identity")

# ---------------------------------------------------------------------------- #
# Simple device name construction

def device_from_tuple(job, taskid, devtype, devid):
  """ Simply build the device string associated with the given tuple.
  Args:
    job     Job name
    taskid  Task ID
    devtype Device type
    devid   Device ID
  Returns:
    Associated device name
  """
  return "/job:" + job + "/replica:0/task:" + taskid + "/device:" + devtype + ":" + devid

# ---------------------------------------------------------------------------- #
# Checkpoint manager

class Checkpoints:
  """ Simple checkpoint manager class.
  """

  def _saver(self):
    """ Lazily instantiate the saver.
    Returns:
      Saver instance
    """
    if self.__saver is None:
      import tensorflow as tf
      self.__saver = tf.train.Saver()
    return self.__saver

  def _update(self):
    """ Update the list of available and latest storage files.
    """
    available = []
    if self.__path.exists():
      for item in self.__path.iterdir():
        if item.is_file() and item.suffix == ".index" and item.stem[:len(self.__base)] == self.__base:
          available.append(str(item)[:-len(".index")])
      available.sort(key=lambda x: int(x[x.rindex("-") + 1:]))
    # Replacements
    self.__available = available

  def __init__(self, path, base=None):
    """ Checkpoint directory constructor.
    Args:
      path Path to the directory storing the checkpoints
      base Base name for the storage file
    """
    import pathlib
    import config
    # Finalization
    self.__path  = pathlib.Path(path)
    self.__base  = base if base is not None else config.default_checkpoint_base_name
    self.__model = str(self.__path / self.__base)
    self.__available = []    # List of storage files currently in the directory
    self.__processed = set() # Set of already restored/iterated over storage files
    self.__saver = None # Lazily instantiated

  def get(self, no_filter=False):
    """ Get a list of available storage files, excluding previously returned entries.
    Args:
      no_filter Whether not to exclude files already returned by previous call(s) of this method
    Returns:
      List of (filtered) available storage file
    """
    # Update view
    self._update()
    # Get (filtered) list
    if no_filter:
      got = self.__available
    else:
      got = []
      for entry in self.__available:
        if entry not in self.__processed:
          got.append(entry)
    for entry in got:
        self.__processed.add(entry)
    return got

  def can_restore(self):
    """ Check whether there is a storage file to restore.
    Returns:
      Whether there is a storage file to restore
    """
    # Update view
    self._update()
    # Check whether restore is possible
    return len(self.__available) > 0

  def restore(self, sess, path=None):
    """ Restore a saved session state.
    Args:
      sess Session to restore upon
      path Path to the storage file to restore (optional, use latest one if None)
    """
    # Update view
    self._update()
    # Default parameter
    if path is None:
      if not self.can_restore():
        raise tools.UserException("No storage file to restore")
      path = self.__available[-1]
    # Session restore
    self._saver().restore(sess, path)

  def save(self, sess, step):
    """ Save the current session state.
    Args:
      sess Session to save
      step Step at which to save the session
    """
    self._saver().save(sess, self.__model, global_step=step)
