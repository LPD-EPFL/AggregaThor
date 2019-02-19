# coding: utf-8
###
 # @file   runner.py
 # @author Sébastien Rouault <sebastien.rouault@epfl.ch>
 #         Georgios Damaskinos <georgios.damaskinos@epfl.ch>
 #	       Arsany Guirguis <arsany.guirguis@epfl.ch>
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
 # Main execution script.
###

if __name__ != "__main__":
  raise tools.UserException("Script " + repr(__file__) + " is to be used as the main module only")

import tools
tools.success("Python module loading phase...")

import argparse
import fractions
import math
import os
import pathlib
import signal
import sys
import threading
import time

import numpy as np
import tensorflow as tf

import config
import cluster
import experiments
import aggregators
import graph

# ---------------------------------------------------------------------------- #
# Graceful termination

exit_pending = False

def mark_exit(*args, **kwargs):
  """ Simply mark exit as pending.
  """
  global exit_pending
  exit_pending = True

# Signal handlers
signal.signal(signal.SIGINT, mark_exit)
signal.signal(signal.SIGTERM, mark_exit)

# ---------------------------------------------------------------------------- #
# Command line
tools.success("Command line parsing phase...")

# Description
parser = argparse.ArgumentParser(description="Start/continue a distributed training session.", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--client",
  type=str,
  default="",
  help="Trusted node URL in the cluster (usually the parameter server) to connect to as a client; one and only one of '--server' and '--client' must be specified")
parser.add_argument("--server",
  type=str,
  default="",
  help="Full JSON cluster specification, on which to act as the only parameter server" + ("" if len(tools.cluster_parsers) == 0 else ", or special value(s): " + tools.cluster_parsers) + "; one and only one of '--server' and '--client' must be specified")
parser.add_argument("--ps-job-name",
  type=str,
  default=config.default_ps_job_name,
  help="Parameter server job name")
parser.add_argument("--ev-job-name",
  type=str,
  default=config.default_ev_job_name,
  help="Evaluation job name (may be the parameter server job name)")
parser.add_argument("--wk-job-name",
  type=str,
  default=config.default_wk_job_name,
  help="Worker job name")
parser.add_argument("--experiment",
  type=str,
  required=True,
  help="Experiment to run on the cluster")
parser.add_argument("--experiment-args",
  nargs="*",
  help="Additional arguments to pass to the underlying experiment")
parser.add_argument("--aggregator",
  type=str,
  required=True,
  help="Gradient aggregation rule to use")
parser.add_argument("--aggregator-args",
  nargs="*",
  help="Additional arguments to pass to the underlying GAR")
parser.add_argument("--optimizer",
  type=str,
  default="sgd",
  help="Optimizer to use")
parser.add_argument("--optimizer-args",
  nargs="*",
  help="Additional arguments to pass to the underlying optimizer")
parser.add_argument("--learning-rate",
  type=str,
  default="fixed",
  help="Type of learning rate decay to use")
parser.add_argument("--learning-rate-args",
  nargs="*",
  help="Additional arguments to pass to the underlying learning rate")
parser.add_argument("--l1-regularize",
  type=float,
  default=-1.,
  help="l1 regularization strength to use, non-positive for none, non-positive by default")
parser.add_argument("--l2-regularize",
  type=float,
  default=-1.,
  help="l2 regularization strength to use, non-positive for none, non-positive by default")
parser.add_argument("--nb-workers",
  type=int,
  required=True,
  help="Total number of workers")
parser.add_argument("--nb-decl-byz-workers",
  type=int,
  default=0,
  help="Number of declared Byzantine workers (i.e. value of 'f')")
parser.add_argument("--nb-real-byz-workers",
  type=int,
  default=0,
  help="Number of real Byzantine workers")
parser.add_argument("--attack",
  type=str,
  default="",
  help="Attack to use (ignored if --nb-real-byz-workers is 0)")
parser.add_argument("--attack-args",
  nargs="*",
  help="Additional arguments to pass to the underlying attack (ignored if --nb-real-byz-workers is 0)")
parser.add_argument("--max-step",
  type=int,
  default=config.default_max_step,
  help="Number of additional steps to perform before stopping the training, non-positive for no limit")
parser.add_argument("--checkpoint-dir",
  type=str,
  default="",
  help="Checkpoint directory to use, will be created if inexistent")
parser.add_argument("--checkpoint-delta",
  type=int,
  default=config.default_checkpoint_delta,
  help="Save checkpoint after the given step delta, negative for unused")
parser.add_argument("--checkpoint-period",
  type=float,
  default=config.default_checkpoint_period,
  help="Save checkpoint at least every given period (in s), negative for unused")
parser.add_argument("--summary-dir",
  type=str,
  default="",
  help="Summary directory to use, '-' for none, defaults to '--checkpoint-dir'")
parser.add_argument("--summary-delta",
  type=float,
  default=config.default_summary_delta,
  help="Save summaries after the given step delta, negative for unused")
parser.add_argument("--summary-period",
  type=float,
  default=config.default_summary_period,
  help="Save summaries at least every given period (in s), negative for unused")
parser.add_argument("--evaluation-file",
  type=str,
  default="",
  help="File in which to write the accuracy evaluations (format: wall time (in s)<tab>global step<tab>name:value<tab>...), '-' for none, defaults to '<checkpoint dir>/" + config.default_evaluation_file_name + "'")
parser.add_argument("--evaluation-delta",
  type=int,
  default=config.default_evaluation_delta,
  help="Evaluate the model after the given step delta, negative for unused")
parser.add_argument("--evaluation-period",
  type=float,
  default=config.default_evaluation_period,
  help="Evaluate the model at least every given period (in s), negative for unused")
parser.add_argument("--use-gpu",
  action="store_true",
  default=False,
  help="Use target GPU devices if available")
parser.add_argument("--reuse-gpu",
  action="store_true",
  default=False,
  help="Allow target GPU devices to be used by several entities, implies '--use-gpu'")
parser.add_argument("--use-tpu",
  action="store_true",
  default=False,
  help="Use target TPU devices if available")
parser.add_argument("--reuse-tpu",
  action="store_true",
  default=False,
  help="Allow target TPU devices to be used by several entities, implies '--use-tpu'")
parser.add_argument("--no-wait",
  action="store_true",
  default=False,
  help="Do not wait for a signal before exiting when acting as a server")
parser.add_argument("--trace",
  action="store_true",
  default=False,
  help="Print a (performance) debugging message for every important step of the graph execution")
parser.add_argument("--stdout-to",
  type=str,
  default="-",
  help="Redirect the standard output to the given file (overwritten if exists), '-' for none, '-' by default")
parser.add_argument("--stderr-to",
  type=str,
  default="-",
  help="Redirect the standard error output to the given file (overwritten if exists), '-' for none, '-' by default")
parser.add_argument("--MPI",
  action="store_true",
  default=False,
  help="Run with MPI instead of gRPC.")

with tools.Context("args", "info"):
  # Command line parsing
  args = parser.parse_args(sys.argv[1:])
  # Early redirection handling
  if args.stdout_to != "-":
    path = pathlib.Path(args.stdout_to)
    sys.stdout = tools.MethodCallReplicator(sys.stdout, tools.ContextIOWrapper(path.open("w"), nocolor=True))
    sys.stdout.write("Duplicating standard output to " + repr(str(path.resolve())) + os.linesep)
  if args.stderr_to != "-":
    path = pathlib.Path(args.stderr_to)
    sys.stderr = tools.MethodCallReplicator(sys.stderr, tools.ContextIOWrapper(path.open("w"), nocolor=True))
    sys.stderr.write("Duplicating standard error output to " + repr(str(path.resolve())) + os.linesep)
  # Command line assertions and warnings
  if args.client and args.server or not (args.client or args.server):
    raise tools.UserException("One and only one of '--client' and '--server' must be specified")
  if args.server: # JSON decoding and opportunistic checks
    args.server = tools.cluster_parse(args.server)
    for job in (args.ps_job_name, args.wk_job_name, args.ev_job_name):
      if job not in args.server:
        raise tools.UserException("Given cluster specification does not include a " + repr(job) + " job")
  if args.nb_workers <= 0:
    raise tools.UserException("Expected at least one non-Byzantine worker")
  if args.nb_workers < args.nb_real_byz_workers:
    raise tools.UserException("Got more real Byzantine workers (" + repr(args.nb_real_byz_workers) + ") than total number of workers (" + repr(args.nb_workers) + ")")
  if args.nb_workers <= 2 * args.nb_decl_byz_workers:
    tools.warning("Got more declared Byzantine workers (" + repr(args.nb_decl_byz_workers) + ") than half the total number of workers (" + repr(args.nb_workers) + ")")
  if args.nb_decl_byz_workers < args.nb_real_byz_workers:
    tools.warning("Got more real Byzantine workers (" + repr(args.nb_real_byz_workers) + ") than declared number of Byzantine workers (" + repr(args.nb_decl_byz_workers) + ")")
  if args.use_tpu or args.reuse_tpu: # NOTE: Remove this warning once the testing is done
    tools.warning("TPU support has not been tested yet and might be buggy")
  # Overriding argument defaults
  args.experiment_args    = [] if args.experiment_args is None else args.experiment_args
  args.aggregator_args    = [] if args.aggregator_args is None else args.aggregator_args
  args.learning_rate_args = [] if args.learning_rate_args is None else args.learning_rate_args
  args.optimizer_args     = [] if args.optimizer_args is None else args.optimizer_args
  args.attack_args        = [] if args.attack_args is None else args.attack_args
  # Evaluation/checkpoint/summary defaults
  if args.checkpoint_dir:
    if not args.evaluation_file:
      args.evaluation_file = str(pathlib.PurePath(args.checkpoint_dir) / config.default_evaluation_file_name)
    elif args.evaluation_file == "-":
      args.evaluation_file = ""
    if not args.summary_dir:
      args.summary_dir = args.checkpoint_dir
    elif args.summary_dir == "-":
      args.summary_dir = ""
  # Number of non-Byzantine workers
  nb_nonbyz_workers = args.nb_workers - args.nb_real_byz_workers
  # Device preferences
  if args.reuse_gpu:
    args.use_gpu = True
  if args.reuse_tpu:
    args.use_tpu = True
  device_prefs = (("TPU",) if args.use_tpu else ()) + (("GPU",) if args.use_gpu else ()) + ("CPU",)
  device_reuse = (("TPU",) if args.reuse_tpu else ()) + (("GPU",) if args.reuse_gpu else ()) + ("CPU",)
  # Print report
  print("Using a total of " + repr(args.nb_workers) + " worker(s):")
  print("· " + repr(nb_nonbyz_workers) + " non-Byzantine worker(s)")
  print("· " + repr(args.nb_decl_byz_workers) + " declared Byzantine worker(s)")
  print("  " + repr(args.nb_real_byz_workers) + " real Byzantine worker(s)")
  tools.print_args("experiment", args.experiment, args.experiment_args, head="")
  tools.print_args("gradient aggregation rule", args.aggregator, args.aggregator_args, head="")
  tools.print_args("learning rate", args.learning_rate, args.learning_rate_args, head="")
  tools.print_args("optimizer", args.optimizer, args.optimizer_args, head="")
  tools.print_args("attack", args.attack, args.attack_args, head="")

if exit_pending:
  exit(0)
# ---------------------------------------------------------------------------- #
# Cluster management
tools.success("Cluster analysis and allocation phase...")

with tools.Context("cluster", "info"):
  # Cluster manager instantiation
  if args.server: # Assume the role of the parameter server, which allows the use of 'tf.py_func'
    tools.info("Acting as node " + args.ps_job_name + ":0 in the cluster")
    if args.MPI:
      proto = 'grpc+mpi'
      print("Using MPI...........................................................")
      sys.stdout.flush()
    else:
      proto = 'grpc'
    args.client = tf.train.Server(tf.train.ClusterSpec(args.server), job_name=args.ps_job_name, task_index=0, start=True, protocol=proto).target
  cluster_mgr = cluster.Manager(args.client, devs=device_prefs, reuse=device_reuse)
  # Cluster allocations (priority for the workers in case of single machine cluster)
  wk_devices = cluster_mgr.allocate("worker", nb_nonbyz_workers, jobs={args.wk_job_name})
  if wk_devices is None:
    raise tools.UserException("Unable to allocate " + repr(nb_nonbyz_workers) + " devices for the workers on the cluster")
  ps_device = cluster_mgr.allocate("ps", 1, jobs={args.ps_job_name})
  if ps_device is None:
    raise tools.UserException("Unable to allocate a device for the parameter server on the cluster")
  else:
    ps_device = ps_device[0]
  ev_device = cluster_mgr.allocate("eval", 1, jobs={args.ev_job_name})
  if ev_device is None:
    raise tools.UserException("Unable to allocate a device for the evaluator on the cluster")
  else:
    ev_device = ev_device[0]
  # Cluster report
  cluster_mgr.report()

if exit_pending:
  exit(0)
# ---------------------------------------------------------------------------- #
# Graph construction
tools.success("Graph construction phase...")

with tools.Context("graph", "info"):
  # Experiment instantiation
  experiment = experiments.instantiate(args.experiment, args.experiment_args)
  # Gradient aggregation rule instantiation
  aggregator = aggregators.instantiate(args.aggregator, args.nb_workers, args.nb_decl_byz_workers, args.aggregator_args)
  # TODO: Eventually add support for a real attack (i.e. take into account when 'args.nb_real_byz_workers > 0' using 'args.attack' and 'args.attack_args')
  # Graph manager instantiation (and construction)
  graph_mgr = graph.Manager(experiment, aggregator, (ps_device, wk_devices, ev_device), args.optimizer, args.optimizer_args, args.learning_rate, args.learning_rate_args, (args.l1_regularize, args.l2_regularize), trace=args.trace)

if exit_pending:
  exit(0)
# ---------------------------------------------------------------------------- #
# Session construction and training
tools.success("Training and evaluation session phase...")

# Evaluation, checkpoint and summary thread entry points
def evaluation_thread(coord, mngr, sess, path, first):
  """ Evaluation thread entry point.
  Args:
    coord Coordinator to use
    mngr  Graph manager to use
    sess  Session to use
    path  Path to the storage file
    first Event notifying first evaluation is complete
  """
  # Check arguments
  global args
  delta  = args.evaluation_delta
  period = args.evaluation_period
  if delta < 0 and period < 0: # Effectively disabled
    tools.info("Evaluation is effectively disabled")
    first.set()
    return
  last_step = -delta
  last_time = -period
  # Open file (if parent exists)
  if path:
    path = pathlib.Path(path)
    try:
      path.parent.mkdir(parents=True, exist_ok=True)
      fd = path.open("a")
    except Exception:
      fd = None
  else:
    fd = None
  # Evaluate (and save) accuracy
  with mngr.graph.as_default():
    while True:
      time.sleep(config.thread_idle_delay)
      step = sess.run(mngr.step)
      now  = time.time()
      stop = coord.should_stop()
      if stop or (delta >= 0 and step - last_step >= delta) or (period >= 0. and now - last_time >= period):
        accuracies = sess.run(mngr.eval_tns)
        if fd is not None:
          line = str(now) + "\t" + str(step)
          for key, val in accuracies.items():
            line += "\t" + key + ":" + str(val)
          fd.write(line + os.linesep)
          fd.flush()
        line = ""
        for key, val in accuracies.items():
          if len(line) > 0:
            line += ", "
          line += key + " = " + str(val)
        tools.info(" Step " + str(step) + ": " + line + " (took " + repr(time.time() - now) + " s)")
        if first is not None:
          first.set()
          first = None
        last_step = sess.run(mngr.step)
        last_time = time.time()
        if stop:
          break
  # Close file (if any)
  if fd is not None:
    fd.close()

def checkpoint_thread(coord, mngr, sess, chck, rstrd):
  """ Checkpoint thread entry point.
  Args:
    coord Coordinator to use
    mngr  Graph manager to use
    sess  Session to use
    chck  Checkpoint manager to use
    rstrd Whether the model was just restored from a checkpoint
  """
  # Check arguments
  global args
  delta  = args.checkpoint_delta
  period = args.checkpoint_period
  if delta < 0 and period < 0: # Effectively disabled
    tools.info("Checkpoint saving is effectively disabled")
    return
  if rstrd:
    last_step = sess.run(mngr.step)
    last_time = time.time()
  else:
    last_step = -delta
    last_time = -period
  # Save checkpoints
  with mngr.graph.as_default():
    while True:
      time.sleep(config.thread_idle_delay)
      step = sess.run(mngr.step)
      now  = time.time()
      stop = coord.should_stop()
      if stop or (delta >= 0 and step - last_step >= delta) or (period >= 0. and now - last_time >= period):
        chck.save(sess, step)
        tools.info("Checkpoint saved (took " + repr(time.time() - now) + " s)")
        last_step = sess.run(mngr.step)
        last_time = time.time()
        if stop:
          break

def summary_thread(coord, mngr, sess, path, rstrd):
  """ Summary thread entry point.
  Args:
    coord Coordinator to use
    mngr  Graph manager to use
    sess  Session to use
    path  Path to the manager to use
    rstrd Whether the model was just restored from a checkpoint
  """
  global args
  delta  = args.summary_delta
  period = args.summary_period
  if delta < 0 and period < 0: # Effectively disabled
    tools.info("Summary saving is effectively disabled")
    return
  if mngr.summary_tn is None:
    tools.warning("No summary to save")
    return
  if rstrd:
    last_step = sess.run(mngr.step)
    last_time = time.time()
  else:
    last_step = -delta
    last_time = -period
  # Save summaries
  with mngr.graph.as_default():
    with tf.summary.FileWriter(args.summary_dir, graph=mngr.graph) as writer:
      writer.add_session_log(tf.SessionLog(status=tf.SessionLog.START), sess.run(mngr.step))
      while True:
        time.sleep(config.thread_idle_delay)
        step = sess.run(mngr.step)
        now  = time.time()
        stop = coord.should_stop()
        if stop or (delta >= 0 and step - last_step >= delta) or (period >= 0. and now - last_time >= period):
          writer.add_summary(sess.run(mngr.summary_tn), step)
          tools.info("Summaries saved (took " + repr(time.time() - now) + " s)")
          last_step = sess.run(mngr.step)
          last_time = time.time()
          if stop:
            break
      writer.add_session_log(tf.SessionLog(status=tf.SessionLog.STOP), step)

# Training session configuration
session_config = tf.ConfigProto()
session_config.allow_soft_placement = True
session_config.log_device_placement = False

# Training
with graph_mgr.graph.as_default():
  with tf.Session(target=args.client, config=session_config) as sess:
    total_runtime = -1. # Total runtime
    first_runtime = -1. # Runtime (in s) of the first training step, which is sometimes way slower...
    graph_runtime = 0.  # Total runtime (in s) in 'sess.run', excluding the first training step
    try:
      # Full initialization
      tools.success("Initializing queues and variables...")
      coord = tf.train.Coordinator()
      tf.train.start_queue_runners(sess=sess, coord=coord) # If there is any queue in the graph
      sess.run(graph_mgr.init_op)
      # Load the latest checkpoint (unless explicitly forbidden)
      with tools.Context("checkpoint", "info"):
        restored = False
        if args.checkpoint_dir:
          checkpoints = tools.Checkpoints(args.checkpoint_dir)
          if checkpoints.can_restore():
            print("Loading latest checkpoint in " + repr(args.checkpoint_dir) + "...")
            checkpoints.restore(sess)
            restored = True
          else:
            print("No checkpoint to restore")
        if exit_pending:
          raise KeyboardInterrupt
      # Launch the evaluation, checkpoint and summary threads
      tools.success("Launching evaluation, checkpoint and summary threads...")
      def launch(entry, name, *args, **kwargs):
        """ Launch and register a new thread.
        Args:
          entry Entry point
          name  Thread name
          ...   Forwarded arguments
        """
        thread = threading.Thread(target=entry, name=name, args=args, kwargs=kwargs)
        thread.start()
        coord.register_thread(thread)
      first_eval = threading.Event()
      launch(evaluation_thread, "test", coord, graph_mgr, sess, args.evaluation_file, first_eval)
      if args.checkpoint_dir:
        launch(checkpoint_thread, "checkpoint", coord, graph_mgr, sess, checkpoints, restored)
      if args.summary_dir:
        launch(summary_thread, "summary", coord, graph_mgr, sess, args.summary_dir, restored)
      # Waiting for initial evaluation (really needed only when continuing past training with an evaluation file)
      first_eval.wait()
      del first_eval
      if exit_pending:
        raise KeyboardInterrupt
      # Actual training
      tools.success("Actual training...")
      def gen():
        if args.max_step > 0:
          for step in range(args.max_step):
            yield step
        else:
          step = 0
          while True:
            yield step
            step += 1
      offstep = sess.run(graph_mgr.step)
      total_runtime = time.time()
      for rawstep in gen():
        step = rawstep + offstep
        runtime_begin = time.time()
        res = sess.run(graph_mgr.train_tn) # One training step
        if first_runtime < 0.:
          first_runtime = time.time() - runtime_begin
        else:
          graph_runtime += time.time() - runtime_begin
        if math.isfinite(res):
          tools.info("Step " + str(step) + ": total loss = " + str(res), context="train")
        else:
          tools.info("Step " + str(step) + ": total loss = NaN", context="train")
          raise tools.UserException("Model diverged with loss = NaN")
        if exit_pending:
          break
    except KeyboardInterrupt:
      pass
    finally:
      # Finish measuring total runtime
      if total_runtime > 0.:
        total_runtime = time.time() - total_runtime
      # Graceful closing
      coord.request_stop()
      coord.join()
      # Print performance measurements
      if total_runtime > 0.:
        offgraph_runtime = total_runtime - graph_runtime
        if first_runtime > 0.:
          offgraph_runtime -= first_runtime
        text  = " In-graph:   " + str(graph_runtime) + " s (" + str(graph_runtime / total_runtime * 100.) + " %)" + os.linesep
        if first_runtime > 0.:
          text += "           + " + str(first_runtime) + " s (" + str(first_runtime / total_runtime * 100.) + " %)" + os.linesep
        text += " Off-graph:  " + str(offgraph_runtime) + " s (" + str(offgraph_runtime / total_runtime * 100.) + " %)" + os.linesep
        text += " Throughput: " + str(rawstep / total_runtime) + " step(s)/s (all steps)" + os.linesep
        if first_runtime > 0.:
          text += "             " + str(rawstep / (total_runtime - first_runtime)) + " step(s)/s (excluding 1st step)"
        tools.info(text, context="perf")

# Wait for any signal before stopping if current process is a cluster node
if args.server and not args.no_wait:
  try:
    with tools.Context(None, "success"):
      sys.stdout.write("Current process is acting as a cluster node: waiting for any signal...")
      sys.stdout.flush()
    signal.pause()
  except KeyboardInterrupt:
    pass
  finally:
    print("")
