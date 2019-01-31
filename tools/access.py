# coding: utf-8
###
 # @file   access.py
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
 # Helpers relative to ability to read/write files and directories.
###

__all__ = ["can_access"]

import os
import pathlib
import stat

# ---------------------------------------------------------------------------- #
# Read-write access check on files and directories

def can_access(path, read=False, write=False, recurse=False):
  """ Check whether a file exists and can be read/written, or a directory's subfiles (and subdirectories) can be listed/read/written.
  Args:
    path    Path to a file or directory (or symlink to it)
    read    Check whether (all sub)files can be read
    write   Check whether (all sub)files can be written
    recurse Recursive search in every subdirectory; ignored if 'path' is (a symlink to) a file
  Returns:
    Whether the access was granted every time; implies that 'path' exists
  """
  try:
    path = pathlib.Path(path)
    # Quick path existence check
    if not path.exists():
      return False
    euid = os.geteuid()
    egid = os.getegid()
    # (Recursive) access rights checks
    spath = path.stat()
    smode = spath.st_mode
    souid = spath.st_uid
    sogid = spath.st_gid
    if stat.S_ISDIR(smode):
      for subpath in path.iterdir():
        if subpath.is_dir() and not recurse:
          continue
        if not can_access(subpath, read, write, recurse):
          return False
    else:
      if euid == souid:
        return (not read or stat.S_IRUSR & smode) and (not write or stat.S_IWUSR & smode)
      elif egid == sogid:
        return (not read or stat.S_IRGRP & smode) and (not write or stat.S_IWGRP & smode)
      else:
        return (not read or stat.S_IROTH & smode) and (not write or stat.S_IWOTH & smode)
    return True
  except OSError:
    return False
