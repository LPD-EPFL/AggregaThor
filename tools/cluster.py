# coding: utf-8
###
 # @file   cluster.py
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
 # Basic TF cluster specification parsers and helpers.
###

__all__ = ["cluster_parsers", "cluster_parse"]

import json
import os
import pathlib

import tools

# ---------------------------------------------------------------------------- #
# G5k cluster parser

_g5k_env_key = "OAR_FILE_NODES"
_g5k_cluster = None

def _g5k_parser():
  """ Generate the cluster specification from the G5k-specific cluster specification file.
  Returns:
    Cluster dictionary, with only 1 ps and n-1 worker(s), all using port 7000
  """
  global _g5k_env_key
  global _g5k_cluster
  if _g5k_cluster is not None:
    return _g5k_cluster
  if _g5k_env_key not in os.environ:
    raise tools.UserException("Key " + repr(_g5k_env_key) + " not found in environment; are you running on Grid5000?")
  multi = pathlib.Path(os.environ[_g5k_env_key]).read_text().strip().split(os.linesep)
  seens = set()
  nodes = []
  for node in multi:
    if node in seens:
      continue
    nodes.append(node + ":7000")
    seens.add(node)
  _g5k_cluster = {"ps": nodes[0:1], "workers": nodes[1:]}
  return _g5k_cluster

# ---------------------------------------------------------------------------- #
# Main cluster parser helper

_cluster_parsers = {
  "G5k": _g5k_parser }

# String representing the list of supported special parsers
cluster_parsers = ("', '").join(_cluster_parsers.keys())
if len(cluster_parsers) > 0:
  cluster_parsers = "'" + cluster_parsers + "'"

def cluster_parse(text):
  """ Parse the given cluster representation.
  Args:
    text Cluster JSON representation or a special parser name
  Returns:
    Cluster dictionary
  """
  global _cluster_parsers
  if text in _cluster_parsers:
    return _cluster_parsers[text]()
  return json.loads(text)
