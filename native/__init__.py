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
 # Native (i.e. C++/CUDA) implementations + automated building and loading.
###

import ctypes
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tensorflow as tf
import traceback
import warnings

import tools

# ---------------------------------------------------------------------------- #
# Common helpers

def _execute(command):
  """ Execute the given command in the current directory, print error messages if the command failed.
  Args:
    command Command to execute
  Returns:
    Whether the operation is a success
  """
  with tools.Context(None, "info"):
    sys.stdout.write("Executing " + repr(command) + "...")
    sys.stdout.flush()
  command = subprocess.run(shlex.split(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  success = command.returncode == 0
  with tools.Context(None, "info" if success else "warning"):
    if success:
      print(" done")
    else:
      print(" fail (" + str(command.returncode) + ")")
    for stdname in ("stdout", "stderr"):
      text = getattr(command, stdname).decode("utf8")
      if len(text) > 0:
        with tools.Context(stdname, "trace"):
          print(text)
  return success

def _camel_to_snake(name):
  """ Transform CamelCase name to snake_case name (https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case#answer-1176023).
  Args:
    name CamelCase name string
  Returns:
    Equivalent snake_case name string
  """
  return re.sub("([a-z0-9])([A-Z])", r"\1_\2", re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)).lower()

def _build_hide_warnings(flags):
  """ Make a generator that change '-I' imports to '-isystem' imports.
  Args:
    flags List of flags to transform
  Returns:
    Generator that performs the transformation
  """
  for flag in flags:
    if len(flag) > 2 and flag[:2] == "-I":
      yield "-isystem" # Trick to hide warning from these imports
      yield flag[2:]
    else:
      yield flag

# ---------------------------------------------------------------------------- #
# Library automatic compilation

# Local constants
_build_exts_hdr  = {".h", ".hh", ".hpp", ".hxx", ".h++", ".cuh"}
_build_exts_src  = {".C", ".cc", ".cpp", ".cxx", ".c++"}
_build_exts_cuda = {".cu"} # As first or second extension
_build_exts_obj  = {".o", ".so"}
_build_cuda_incl = "/usr/local"
_build_cuda_libs = "/usr/local/cuda/lib64"
_build_inclpath  = pathlib.Path(__file__).parent / "include"
try:
  _build_include = _build_inclpath.resolve()
except Exception:
  warnings.warn("Not found include directory: " + repr(str(_build_inclpath)))
  _build_include = None
_build_self_time = pathlib.Path(__file__).stat().st_mtime
_build_compbin   = "c++"
_build_compflags = ["-std=c++11", "-O2", "-DNDEBUG"]
_build_compcopts = ["-fPIC", "-Wall", "-Wextra", "-Wfatal-errors", "-Wno-unknown-attributes"]
_build_compincls = list(_build_hide_warnings(tf.sysconfig.get_compile_flags())) + ([] if _build_include is None else ["-I" + str(_build_include)])
_build_cudabin   = "nvcc"
_build_cudaflags = ["-DGOOGLE_CUDA=1", "--expt-relaxed-constexpr"]
_build_cudacopts = ["-Wno-maybe-uninitialized"]
_build_cudaincls = _build_compincls
_build_linkbin   = "c++"
_build_linkflags = ["-shared", "-Wl,--no-as-needed"] + list(_build_hide_warnings(tf.sysconfig.get_link_flags()))
_build_linkincls = ["-L" + _build_cuda_libs, "-L/"]

# Local computed constants
_build_command_comp = (" ").join([_build_compbin] + _build_compflags + _build_compcopts + _build_compincls)
_build_command_cuda = (" ").join([_build_cudabin] + _build_compflags + ["-Xcompiler", (",").join(_build_compcopts + _build_cudacopts)] + _build_cudaflags + _build_compincls + ["-isystem", _build_cuda_incl])
_build_command_link = (" ").join([_build_linkbin] + _build_linkflags + _build_linkincls)
_build_check_ident  = None # Overwritten by loader code

def _build_cpp_cmd(src_path, obj_path, has_cuda):
  """ Build the command line to compile a CUDA source file.
  Args:
    src_path Source file path
    obj_path Object file path
    has_cuda Whether the parent library uses CUDA
  Returns:
    Command line to compile the given source
  """
  # Command line building
  # "c++ -Wall -Wextra -Wfatal-errors -O2 -std=c++11 -fPIC " + _compile_flags + ("" if len(gpu_srcs) == 0 else " -DGOOGLE_CUDA=1") + " -c -x c++ -o " + shlex.quote(str(obj_path)) + " " + shlex.quote(str(cpu_src))
  return _build_command_comp + (" -DGOOGLE_CUDA=1 " if has_cuda else "") + " -c -x c++ -o " + shlex.quote(str(obj_path)) + " " + shlex.quote(str(src_path))

def _build_cuda_cmd(src_path, obj_path):
  """ Build the command line to compile a CUDA source file.
  Args:
    src_path Source file path
    obj_path Object file path
  Returns:
    Command line to compile the given source
  """
  # Check and warn once
  global _build_cuda_cmd_checked
  if not _build_cuda_cmd_checked:
    _build_cuda_cmd_checked = True
    if not pathlib.Path(_build_cuda_incl).exists():
      warnings.warn("Not found CUDA include directory: " + repr(_build_cuda_incl), category=UserWarning)
    if not pathlib.Path(_build_cuda_libs).exists():
      warnings.warn("Not found CUDA library directory: " + repr(_build_cuda_libs), category=UserWarning)
  # Command line building
  # "nvcc -O2 -std=c++11 -Xcompiler -Wall,-Wextra,-Wfatal-errors,-fPIC " + _compile_flags + ("" if _cuda_include is None else " -isystem " + _cuda_include) + " -DGOOGLE_CUDA=1 --expt-relaxed-constexpr -x cu -c -o " + shlex.quote(str(obj_path)) + " " + shlex.quote(str(gpu_src))
  return _build_command_cuda + " -c -x cu -o " + shlex.quote(str(obj_path)) + " " + shlex.quote(str(src_path))
_build_cuda_cmd_checked = False

def _build_link_cmd(obj_paths, so_paths, so_path):
  """ Build the command line to link the given object files.
  Args:
    obj_paths List of object file paths
    so_paths  List of shared object file paths
    so_path   Shared object file path
  Returns:
    Command line to link the given object
  """
  # Command line building
  # "c++ -shared -Wl,--no-as-needed " + ("" if so_deps is None else shlex.quote("-L" + so_base) + " " + (" ").join([shlex.quote("-l:" + str(x)) for x in so_deps]) + " ") + _link_flags + ("" if len(gpu_srcs) == 0 else " -L" + _cuda_libs + " -lcudart") + " -o " + shlex.quote(str(so_path)) + " " + (" ").join([shlex.quote(str(obj)) for obj in objects])
  return _build_command_link + " " + (" ").join([shlex.quote("-l:" + str(x)) for x in so_paths]) + " -o " + shlex.quote(str(so_path)) + " " + (" ").join([shlex.quote(str(x)) for x in obj_paths])

def _build_so_path(libpath):
  """ Build the shared object path associated with the given library directory path.
  Args:
    libpath Absolute library directory path
  Returns:
    Associated shared object path
  """
  # Assertions
  assert libpath.is_absolute(), "Expected absolute library directory path, got " + repr(str(libpath))
  # Path construction
  return libpath.parent / (libpath.name + ".so") # Do not use 'with_suffix' since we want unconditional append

def _build_must_rebuild(product, sources):
  """ Tells whether the product should be rebuilt from the sources.
  Args:
    product Path to the product file (may not exist)
    sources Iterable of source files (must all exist)
  Returns:
    Whether rebuild should occur
  """
  if not product.exists():
    return True
  plast = product.stat().st_mtime
  if _build_self_time >= plast:
    return True
  for source in sources:
    if source.stat().st_mtime >= plast:
      return True
  return False

def _build_library(libpath, doneset, failset, headers, libstack=[], nocuda=False):
  """ (Re)build a library directory and its dependencies into their associated shared objects.
  Args:
    libpath  Library directory path
    doneset  Set of other, successfully built library directory paths to update
    failset  Set of other, not compiling library directory paths to update
    headers  List of shared header paths
    libstack Constant stack of dependent library directory paths
    nocuda   CUDA compiler was not found, don't try to compile these files
  Returns:
    Built library shared object path (None on failure)
  """
  with tools.Context(libpath.name, None):
    try:
      # Watch out for a dependency cycle
      libpath  = libpath.resolve()
      hascycle = libpath in libstack
      libstack += [libpath]
      if hascycle:
        raise RuntimeError("dependency cycle found")
      # List dependencies and sources (per category) to build
      depends = [] # Library directory paths to build (some may already be built/not compile)
      shareds = [] # Shared object paths this library depends on
      headers = list(headers) # Header paths (initially copy of common headers)
      srcscpu = [] # C++ source paths
      srcsgpu = [] # CUDA source paths
      libroot = libpath.parent
      for path in libpath.iterdir():
        try:
          path = path.resolve()
        except Exception:
          if path.is_symlink():
            raise RuntimeError("missing dependency " + repr(os.readlink(str(path))))
          continue # Else silently ignore file
        if path.is_dir():
          if path.parent != libroot: # Silently ignore directory
            continue
          if _build_check_ident(path): # Is a valid dependency
            depends.append(path)
        else:
          if path.parent != libpath: # Silently ignore file
            continue
          exts = path.suffixes
          if len(exts) > 0:
            if exts[-1] in _build_exts_hdr:
              headers.append(path)
              continue
            elif exts[-1] in _build_exts_cuda:
              srcsgpu.append(path)
              continue
            elif exts[-1] in _build_exts_src:
              if len(exts) > 1 and exts[-2] in _build_exts_cuda:
                srcsgpu.append(path)
              else:
                srcscpu.append(path)
              continue
            elif exts[-1] in _build_exts_obj:
              continue
          tools.trace("Ignoring file " + repr(path.name) + ": no/unrecognized extension")
      if nocuda: # No CUDA compiler => we ignore any CUDA source
        srcsgpu.clear()
      # Process dependencies first
      for path in depends:
        if path in failset:
          raise RuntimeError("dependency " + repr(path.name) + " could not be built")
        if path in doneset:
          so_path = _build_so_path(path)
        else:
          so_path = _build_library(path, doneset, failset, headers, libstack, nocuda=nocuda)
          if so_path is None:
            raise RuntimeError("dependency " + repr(path.name) + " could not be built")
        shareds.append(so_path)
      # Process sources second
      obj_paths = [] # Object paths to link
      for src_path in srcscpu:
        obj_path = pathlib.Path(str(src_path) + ".o")
        if _build_must_rebuild(obj_path, headers + [src_path]):
          if not _execute(_build_cpp_cmd(src_path, obj_path, len(srcsgpu) > 0)):
            raise RuntimeError("C++ source " + repr(src_path.name) + " did not compile")
        obj_paths.append(obj_path)
      for src_path in srcsgpu:
        obj_path = pathlib.Path(str(src_path) + ".o")
        if _build_must_rebuild(obj_path, headers + [src_path]):
          if not _execute(_build_cuda_cmd(src_path, obj_path)):
            raise RuntimeError("CUDA source " + repr(src_path.name) + " did not compile")
        obj_paths.append(obj_path)
      # (Re)link the shared object
      so_path = _build_so_path(libpath)
      if _build_must_rebuild(so_path, obj_paths):
        if not _execute(_build_link_cmd(obj_paths, shareds, so_path)):
          raise RuntimeError("final shared object " + repr(so_path.name) + " could not be linked")
      doneset.add(libpath)
      return so_path
    except Exception as err:
      tools.warning("Build failed: " + str(err))
      failset.add(libpath)
      return None

# ---------------------------------------------------------------------------- #
# Python-native import helper

def import_py(lib, name, args, ret, defs=None, echk=None):
  """ Import a native function.
  Args:
    lib  Library to load from
    name Function name in the library
    args List of argument classes
    ret  Return value class
    defs Defaulted tuple of last parameters
    echk Optional filter/error checking function
  Returns:
    Function instance
  """
  # Assertions
  if defs is not None:
    assert isinstance(defs, tuple), "Expected a 'tuple' for argument 'defs'"
    assert len(defs) <= len(args), "More defaults provided than possible arguments"
  # Import and declaration
  func = getattr(lib, name, None)
  if func is None:
    raise RuntimeError("Function " + repr(name) + " is not exported in " + repr(lib._name))
  func.argtypes = args
  func.rettype = ret
  # Optional error check
  if echk is not None:
    func.errcheck = echk
  # Application of the optional parameters
  if defs is None:
    return func
  else:
    def call(*curr):
      """ Call 'func', replacing missing and defaulted arguments.
      Args:
        ... Forwarded arguments
      Returns:
        Forwarded returned value
      """
      assert len(args) <= len(curr) + len(defs), "Not enough parameters provided in native function call"
      return func(*(curr + defs[len(curr) - len(args) + len(defs):]))
    return call

# ---------------------------------------------------------------------------- #
# Automatic building and loading

# Register foreign import instance
_register      = tools.ClassRegister("foreign import")
itemize_py     = _register.itemize
register_py    = _register.register
instantiate_py = _register.instantiate
del _register

def _loader_ctypes(so_path):
  """ Post-building ctypes loading operations.
  Args:
    so_path Shared object path
  """
  try:
    lib = ctypes.CDLL(str(so_path))
    register_py(so_path.stem[3:], lambda: lib)
  except Exception as err:
    with tools.Context(so_path.stem, "warning"):
      print("Loading failed for python interface " + repr(str(so_path)) + ": " + str(err))
      with tools.Context("traceback", "trace"):
        traceback.print_exc()

# Register custom ops instance
_register      = tools.ClassRegister("custom operation")
itemize_op     = _register.itemize
register_op    = _register.register
instantiate_op = _register.instantiate
del _register

def _loader_ops(so_path):
  """ Post-building custom ops loading operations.
  Args:
    so_path Shared object path
  """
  try:
    lib = tf.load_op_library(str(so_path))
    entries = lib.OP_LIST.ListFields()[0][1]
    try:
      while True:
        opname = entries.pop().ListFields()[0][1]
        opname = _camel_to_snake(opname)
        register_op(opname, getattr(lib, opname))
    except IndexError:
      pass
  except Exception as err:
    with tools.Context(so_path.stem, "warning"):
      print("Loading failed for custom op " + repr(str(so_path)) + ": " + str(err))
      with tools.Context("traceback", "trace"):
        traceback.print_exc()

# Loader constants
_loader_hooks = {"so_": None, "op_": _loader_ops, "py_": _loader_ctypes}
_build_check_ident = lambda path: path.name[:3] in _loader_hooks.keys()

def _loader():
  """ Incrementally rebuild all libraries and register all local operations.
  """
  try:
    # Check if the CUDA compiler is available
    nocuda = True
    for path in os.environ["PATH"].split(os.pathsep):
      if (pathlib.Path(path) / _build_cudabin).exists():
        nocuda = False
        break
    # List all common headers
    headers = []
    if _build_include is not None:
      for path in _build_include.iterdir():
        if path.suffix in _build_exts_hdr:
          headers.append(path)
    # Compile libraries and load OP
    doneset = set()
    failset = set()
    for dirpath in pathlib.Path(__file__).resolve().parent.iterdir():
      ident = dirpath.name[:3]
      if dirpath.is_dir() and ident in _loader_hooks.keys(): # Is a library directory
        if dirpath not in doneset and dirpath not in failset:
          so_path = _build_library(dirpath, doneset, failset, headers, nocuda=nocuda)
          loader  = _loader_hooks[ident]
          if so_path is not None and loader is not None: # Successful build and loader needed
            loader(so_path)
  except Exception as err:
    with tools.Context(ident, "warning"):
      print("Loading failed while compiling " + repr(ident) + ": " + str(err))
      with tools.Context("traceback", "trace"):
        traceback.print_exc()

with tools.Context("native", None):
  _loader()
