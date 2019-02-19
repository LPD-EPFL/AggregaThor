# coding: utf-8
###
 # @file   config.py
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

if __name__ == "__main__":
  print("This script is not to be used as the main module")
  exit(1)

# ---------------------------------------------------------------------------- #
# Common, default parameters

# Cluster
default_ps_job_name = "ps"
default_wk_job_name = "workers"
default_ev_job_name = "eval"

# Training
default_max_step          = 10000
default_learning_rate     = 1e-3
default_end_learning_rate = 1e-4
default_decay_step        = 10000
default_decay_rate        = 0.96

# Evaluation
default_evaluation_file_name = "eval"
default_evaluation_delta     = -1
default_evaluation_period    = 10.
default_checkpoint_base_name = "model"
default_checkpoint_delta     = -1
default_checkpoint_period    = 120.
default_summary_delta        = -1
default_summary_period       = 30.

# ---------------------------------------------------------------------------- #
# Static configuration

thread_idle_delay = 1. # Delay in loop for the eval/check/sum. threads
