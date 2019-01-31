# coding: utf-8
###
 # @file   graph.py
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
 # Graph-related management module for 'runner.py'.
###

if __name__ == "__main__":
  print("This script is not to be used as the main module")
  exit(1)

import sys
import time

import numpy as np
import tensorflow as tf

import config
import tools

# ---------------------------------------------------------------------------- #
# Supported learning rates and optimizers

# Structure: name -> (constructor, argument -> (default value, constructor's kwargs name))
learning_rates = {
  "fixed": (lambda global_step=None, initial_rate=None: tf.constant(initial_rate),
            {"initial-rate": (config.default_learning_rate, "initial_rate")}),
  "polynomial": (lambda global_step=None, initial_rate=None, decay_step=None, end_rate=None, power=None: tf.train.polynomial_decay(initial_rate, global_step, decay_step, end_rate, power=power, cycle=False),
                 {"initial-rate": (config.default_learning_rate, "initial_rate"), "end-rate": (config.default_end_learning_rate, "end_rate"), "decay-step": (config.default_decay_step, "decay_step"), "power": (1., "power")}),
  "exponential": (lambda global_step=None, initial_rate=None, decay_step=None, decay_rate=None: tf.train.exponential_decay(initial_rate, global_step, decay_step, decay_rate, staircase=False),
                 {"initial-rate": (config.default_learning_rate, "initial_rate"), "decay-step": (config.default_decay_step, "decay_step"), "decay-rate": (config.default_decay_rate, "decay_rate")}) }
optimizers = {
  "adadelta": (lambda learning_rate=None, **kwargs: tf.train.AdadeltaOptimizer(learning_rate, **kwargs),
               {"adadelta-rho": (0.95, "rho"), "opt-epsilon": (1., "epsilon")}),
  "adagrad": (lambda learning_rate=None, **kwargs: tf.train.AdagradOptimizer(learning_rate, **kwargs),
              {"initial-accumulator-value": (0.1, "initial_accumulator_value")}),
  "adam": (lambda learning_rate=None, **kwargs: tf.train.AdamOptimizer(learning_rate, **kwargs),
           {"adam-beta1": (0.9, "beta1"), "adam-beta2": (0.999, "beta2")}),
  "rmsprop": (lambda learning_rate=None: tf.train.RMSPropOptimizer(learning_rate), {}),
  "sgd": (lambda learning_rate=None: tf.train.GradientDescentOptimizer(learning_rate), {}) }

# Instantiation helper
def build(struct, name, select, args, **kwargs):
  """ Call the constructor associated with the given selection and the given keyword + parsed arguments.
  Args:
    struct Structure defining constructors and their respective arguments
    name   Name of what is built by the constructor
    select Constructor to select
    args   List of "key:value" command line arguments
    ...    Key-value arguments forwarded to the constructor
  """
  # Recover constructor and argument structure
  if select not in struct:
    raise tools.UserException("Unknown " + name + " " + repr(select) + ", " + ("no " + name + " available" if len(struct) == 0 else "expected one of: '" + ("', '").join(struct.keys()) + "'"))
  construct, args_struct = struct[select]
  # Translate parameters
  defaults = {}
  for key, val in args_struct.items():
    defaults[key] = val[0]
  args_parsed = tools.parse_keyval(args, defaults=defaults)
  # Instantiate and return
  args_kw = {}
  for key, val in args_struct.items(): # Ignore supplementary parameters by using '_struct' instead of '_parsed'
    args_kw[args_struct[key][1]] = args_parsed[key]
  return construct(**args_kw, **kwargs)

# ---------------------------------------------------------------------------- #
# PS-worker device setter producer

# Operation types that goes on the parameter server
_ps_ops = ("Variable", "VariableV2", "VarHandleOp", "AutoReloadVariable",
           "MutableHashTableV2", "MutableDenseHashTableV2", "MutableHashTable",
           "MutableHashTableOfTensorsV2", "MutableDenseHashTable",
           "BoostedTreesEnsembleResourceHandleOp", "MutableHashTableOfTensors")

def replica_device_setter(device_ps, device_wk):
  """ Generate a PS-worker device setter.
  Args:
    device_ps Parameter server device name/function
    device_wk Current worker device name/function
  Returns:
    Device setter closure
  """
  def setter(op):
    global _ps_ops
    if op.type in _ps_ops:
      if callable(device_ps):
        return device_ps(op)
      return device_ps
    else:
      if callable(device_wk):
        return device_wk(op)
      return device_wk
  return setter

# ---------------------------------------------------------------------------- #
# l1/l2 regularization helpers

def regularization(norm):
  """ Compute the regularization loss.
  Args:
    norm Norm to use (i.e. 1 or 2)
  Returns:
    Regularization loss
  """
  # Loss computation
  variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
  if norm == 1:
    return tf.reduce_sum([tf.reduce_sum(tf.abs(variable)) for variable in variables], name="l1_loss")
  elif norm == 2:
    return tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(variable)) for variable in variables]), name="l2_loss")
  else:
    "Invalid value " + repr(norm) + " for parameter 'norm'"

# ---------------------------------------------------------------------------- #
# Gradient flattening-inflating helpers

def flatten(tensors, flatmap=None):
  """ Flatten the tensor from the list of (tensor, variable).
  Args:
    tensors List of (tensor, variable)
    flatmap Mapping between variables and their gradient's position in the flattened tensor (optional, build it if None)
  Returns:
    Flattened tensor, mapping variable/position (only if 'flatmap' was None)
  """
  with tf.name_scope("flatten"):
    if flatmap is None:
      flatmap = {}
      res = []
      for gradient, variable in tensors:
        if gradient is None:
          continue
        flatmap[variable] = len(res)
        res.append(tf.reshape(gradient, (-1,)))
      return tf.concat(res, 0), flatmap
    else:
      res = [None] * len(flatmap)
      for gradient, variable in tensors:
        if gradient is None:
          continue
        res[flatmap[variable]] = tf.reshape(gradient, (-1,))
      return tf.concat(res, 0)

def mapflat(flatmap):
  """ Transform a map variable -> gradient position into the associated ordered list of variables.
  Args:
    flatmap Mapping between variables and their gradient's position in the flattened tensor
  Returns:
    List of variables in the order defined by their respective position in the flat gradient
  """
  res = [None] * len(flatmap)
  for variable, position in flatmap.items():
    res[position] = variable
  return res

def inflate(tensor, mapflat):
  """ Inflate the tensor to a list of (tensor, variable).
  Args:
    tensor  Flattened tensor
    mapflat List of variables in the order defined by their respective position in the flat gradient
  Returns:
    List of (tensor, variable)
  """
  res = []
  pos = 0
  with tf.name_scope("inflate"):
    for variable in mapflat:
      shape = variable.shape
      size = shape.num_elements()
      tnsr = tf.reshape(tf.slice(tensor, [pos], [size]), shape)
      res.append((tnsr, variable))
      pos += size
  return res

# ---------------------------------------------------------------------------- #
# Graph manager class

class Manager:
  """ Graph manager class.
  """

  def __init__(self, experiment, aggregator, dev_tuples, optimizer, optimizer_args, learning_rate, learning_rate_args, regularizations=(-1., -1.), trace=False):
    """ Full graph (training + evaluation) constructor.
    Args:
      experiment         Experiment instance to use
      aggregator         Aggregator instance to use
      dev_tuples         Tuple of devices (i.e. tuples of strings (job name, task ID, device type, device ID)) for (parameter server, each workers' inference/loss/gradient computation, evaluator)
      optimizer          Optimizer name to use
      optimizer_args     Additional optimizer key-value arguments
      learning_rate      Learning rate name to use
      learning_rate_args Additional learning rate key-value arguments
      regularizations    Pair of (l1, l2) regularization values, non-positive values for no regularization
      trace              Whether to add trace prints for every important step of the computations
    """
    # Tuple extraction and device name reconstruction
    ps_tuple, wk_tuples, ev_tuple = dev_tuples
    ps_device = tools.device_from_tuple(*ps_tuple)
    wk_jobs = {} # Map job -> taskid -> list of pairs of (devtype, devid)
    for job, taskid, devtype, devid in wk_tuples:
      if job in wk_jobs:
        wk_tasks = wk_jobs[job]
        if taskid in wk_tasks:
          wk_tasks[taskid].append((devtype, devid))
        else:
          wk_tasks[taskid] = [(devtype, devid)]
      else:
        wk_jobs[job] = {taskid: [(devtype, devid)]}
    # Graph building
    graph = tf.Graph()
    with graph.as_default():
      with tf.name_scope("ps/"):
        with tf.device(ps_device):
          # Instantiate global step counter, optimizer and learning rate
          global_step   = tf.train.create_global_step()
          learning_rate = build(learning_rates, "learning rate decay", learning_rate, learning_rate_args, global_step=global_step)
          optimizer     = build(optimizers, "optimizer", optimizer, optimizer_args, learning_rate=learning_rate)
          tf.summary.scalar("learning_rate", learning_rate)
          # Create workers' gradient computation
          totlosses = [] # List of losses, for summary (and printing) only
          gradients = [] # List of gradients, one per non-Byzantine worker
          flatmap = None # Flat map used to flatten the gradients coherently
          with tf.name_scope("workers/"):
            for job, wk_tasks in wk_jobs.items():
              for taskid, models in wk_tasks.items():
                device_dataset = tools.device_from_tuple(job, taskid, "CPU", "*")
                device_models  = [replica_device_setter(ps_device, tools.device_from_tuple(job, taskid, devtype, devid)) for devtype, devid in models]
                # Compute losses
                losses = experiment.losses(device_dataset, device_models, trace=trace)
                totlosses += losses
                # Compute gradients
                for i in range(len(device_models)):
                  with tf.device(device_models[i]):
                    loss = losses[i]
                    for norm in [1, 2]:
                      strength = regularizations[norm - 1] # 'norm - 1' is just a basic numbering trick...
                      if strength > 0.:
                        loss = loss + strength * regularization(norm)
                    if trace:
                      loss = tools.trace_graph(loss, "Worker " + str(len(gradients)) + ": loss computation")
                    grad_vars = optimizer.compute_gradients(loss)
                    if flatmap is None:
                      gradient, flatmap = flatten(grad_vars)
                    else:
                      gradient = flatten(grad_vars, flatmap)
                    if trace:
                      gradient = tools.trace_graph(gradient, "Worker " + str(len(gradients)) + ": gradient computation")
                    gradients.append(gradient)
          total_loss = tf.add_n(totlosses, name="total_loss")
          tools.info("Created workers' dataset, inference, loss and gradient computation nodes")
          # Aggregate and apply the workers' gradients
          with tf.name_scope("GAR"):
            aggregated = aggregator.aggregate(gradients)
            if trace:
              aggregated = tools.trace_graph(aggregated, "Master: aggregated gradient computation")
          apply_op = optimizer.apply_gradients(inflate(aggregated, mapflat(flatmap)), global_step=global_step)
          if trace:
            apply_op = tools.trace_graph(apply_op, "Master: aggregated gradient application")
          tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, apply_op)
          tools.info("Created parameter server's gradient aggregation and application nodes")
          # Create accuracy computation
          with tf.name_scope("eval/"):
            device_dataset = tools.device_from_tuple(ev_tuple[0], ev_tuple[1], "CPU", "*")
            device_model   = tools.device_from_tuple(*ev_tuple)
            accuracy_tns = experiment.accuracy(device_dataset, [replica_device_setter(ps_device, device_model)], trace=trace)
          for key, val in accuracy_tns.items():
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, tf.summary.scalar("eval-" + key, val))
          tools.info("Created evaluator's dataset, inference and accuracy computation nodes")
          # Global summary protocol buffer
          summary_tn = tf.summary.merge(list(set(tf.get_collection(tf.GraphKeys.SUMMARIES))))
          # Full initialization operation
          rsrc_init_ops = []
          for resource in tf.get_collection(tf.GraphKeys.RESOURCES):
            rsrc_init_ops.append(resource.initializer)
          for resource in tf.get_collection(tf.GraphKeys.LOCAL_RESOURCES):
            rsrc_init_ops.append(resource.initializer)
          init_op = tf.group(tf.variables_initializer(tf.global_variables() + tf.local_variables()), tf.tables_initializer(), *rsrc_init_ops)
          # Build the training operation
          with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_tn = tf.identity(total_loss, name="train_tn")
    # Finalization
    self.graph      = graph
    self.step       = global_step
    self.rate       = learning_rate
    self.optimizer  = optimizer
    self.total_loss = total_loss
    self.summary_tn = summary_tn
    self.init_op    = init_op
    self.train_tn   = train_tn
    self.eval_tns   = accuracy_tns
