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
 # Cluster-related management module for 'runner.py'.
###

if __name__ == "__main__":
  print("This script is not to be used as the main module")
  exit(1)

import tensorflow as tf

import config
import tools

# ---------------------------------------------------------------------------- #
# Cluster manager

class Manager:
  """ Cluster manager class.
  """

  def __init__(self, target, devs=None, reuse=None):
    """ Cluster structure available at the target constructor.
    Args:
      target Target cluster server to use
      devs   Iterable of device types to allocate from, by order of preference (optional)
      reuse  Set of device types for first reallocation is allowed (optional, None for no reallocation)
    """
    # Cluster query
    with tf.Session(graph=tf.Graph(), target=target) as sess:
      devices = sess.list_devices()
    # Structure extraction
    has_replicas = False
    structure = {}
    for device in devices:
      # Extract field values
      def extract_field(text, name, conv):
        lim = "/" + name + ":"
        pos = text.find(lim)
        if pos < 0:
          raise tools.UserException("Missing field " + repr(name) + " in device name " + repr(device.name))
        text = text[pos + len(lim):]
        pos = text.find("/")
        if pos < 0:
          return conv(text), ""
        else:
          return conv(text[:pos]), text[pos:]
      devname = device.name
      jobname, devname = extract_field(devname, "job", str)
      replica, devname = extract_field(devname, "replica", int)
      if replica != 0:
        has_replicas = True
        continue
      taskid, devname = extract_field(devname, "task", int)
      # Get or create task list for this job
      if jobname in structure:
        tasks = structure[jobname]
      else:
        tasks = {}
        structure[jobname] = tasks
      # Get or create task for this task list
      if taskid in tasks:
        task = tasks[taskid]
      else:
        task = {}
        tasks[taskid] = task
      # Get or create device list
      if device.device_type in task:
        devlist = task[device.device_type]
      else:
        devlist = []
        task[device.device_type] = devlist
      # Insert device
      devlist.append({"attr": device, "used": []})
    # If needed, warn about what is currently unsupported
    if has_replicas:
      tools.warning("Replicas are unsupported and will be ignored.")
    # Filtered, special copy of all available (i.e. allowed and free) devices
    def restricted(iterator, restrict=None):
      if restrict is None:
        for item in iterator:
          yield item
      else:
        for item in iterator:
          if item in restrict:
            yield item
    if reuse is None:
      reuse = {}
    tasks = []
    for job, taskdevs in structure.items():
      for taskid, devtypes in taskdevs.items():
        taskcopy = []
        for devtype in restricted(devtypes.keys(), restrict=devs):
          for entry in devtypes[devtype]:
            if devtype in reuse or len(entry["used"]) == 0: # Reusable or unallocated
              devname = entry["attr"].name
              taskcopy.append((entry["used"], (job, str(taskid), devtype, devname[devname.rindex(":") + 1:])))
        if len(taskcopy) > 0:
          tasks.append(taskcopy)
    # Finalization
    self.__structure = structure
    self.__tasks     = tasks
    self.__devs      = devs
    self.__reuse     = reuse

  def report(self):
    """ Report on the cluster structure and allocations.
    """
    print("Cluster structure and allocation report:")
    for job, tasks in self.__structure.items():
      print(" · Job " + repr(job))
      for taskid, devtypes in tasks.items():
        print("   · Task " + repr(taskid))
        for devtype, devices in devtypes.items():
          for i in range(len(devices)):
            used = devices[i]["used"]
            print("     · " + devtype + " " + str(i) + ": " + ("<unallocated>" if len(used) == 0 else (", ").join(used)))

  def allocate(self, name, count, jobs=None, partial=False):
    """ Allocate (i.e. reserve) free, allowed devices from the cluster.
    Args:
      name    Identifier (for informational purpose) of the recipient of the allocation
      count   Number of devices to reserve (NB: devices of reused types can be reserved any number of times)
      jobs    Set of jobs to restrict allocation from (optional, None for no restriction)
      partial Allows partial allocation (=> len(<return list>) may be below count)
    Returns:
      List of tuples of strings (job name, task ID, device type, device ID) or None if non-partial allocation failed
    """
    # Quick path and assertions
    if count == 0:
      tools.warning("Successfully allocated 0 device")
      return []
    elif count < 0:
      raise tools.UserException("Expected non-negative number of devices to reserve, got " + repr(count))
    # Default 'jobs' and warning
    if jobs is not None:
      for job in jobs:
        if job not in self.__structure:
          tools.warning("Job " + repr(job) + " does not exist in the cluster")
    # Select devices by order of preference, maximizing task spread
    tasks = self.__tasks
    reuse = self.__reuse
    def select_device(devtype):
      nonlocal tasks
      nonlocal reuse
      for taskid in range(len(tasks)): # Look into next task
        devs = tasks[taskid]
        for devid in range(len(devs)): # Look into next available device (perhaps of wrong type)
          pair = devs[devid]
          if jobs is not None and pair[1][0] not in jobs: # Disallowed job
            continue
          if pair[1][2] != devtype: # Wrong type
            continue
          # (Re)move device and task (back)
          devs = devs[:devid] + devs[devid + 1:]
          if devtype in reuse: # Device is reusable: keep it in the list
            devs.append(pair)
          tasks = tasks[:taskid] + tasks[taskid + 1:]
          if len(devs) > 0: # Push back remaining devices
            tasks.append(devs)
          # Select device
          return pair
      # No available device of the requested type
      return None
    def mark(select):
      nonlocal self
      nonlocal tasks
      nonlocal name
      # Update the new set of tasks
      self.__tasks = tasks
      # Mark the selection as used
      res = []
      count = 0
      for used, info in select:
        res.append(info)
        used.append(name + "[" + str(count) + "]")
        count += 1
      # Return only the list of device tuples
      return res
    select = []
    for devtype in self.__devs:
      while True: # As long as we find devices of the current type
        pair = select_device(devtype)
        if pair is None:
          break
        select.append(pair)
        if len(select) == count:
          return mark(select)
    # Full allocation failed...
    if partial:
      return mark(select)
    # ...and partial allocation was forbidden
    return None
