# coding: utf-8
###
 # @file   misc.py
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
 # Miscellaneous Python helpers.
###

__all__ = [
  "MethodCallReplicator", "ClassRegister", "parse_keyval", "print_args",
  "ExpandPath", "make_interface"]

import sys

import tools

# ---------------------------------------------------------------------------- #
# Simple method call replicator

class MethodCallReplicator:
  """ Simple method call replicator class.
  """

  def __init__(self, *args):
    """ Bind constructor.
    Args:
      ... Instance on which to replicate method calls (in the given order)
    """
    # Assertions
    assert len(args) > 0, "Expected at least one instance on which to forward method calls"
    # Finalization
    self.__instances = args

  def __getattr__(self, name):
    """ Returns a closure that replicate the method call.
    Args:
      name Name of the method
    Returns:
      Closure replicating the calls
    """
    # Target closures
    closures = [getattr(instance, name) for instance in self.__instances]
    # Replication closure
    def calls(*args, **kwargs):
      """ Simply replicate the calls, forwarding arguments.
      Args:
        ... Forwarded arguments
      Returns:
        List of returned values
      """
      return [closure(*args, **kwargs) for closure in closures]
    # Build the replication closure
    return calls

# ---------------------------------------------------------------------------- #
# Simple class register

class ClassRegister:
  """ Simple class register.
  """

  def __init__(self, singular, optplural=None):
    """ Denomination constructor.
    Args:
      singular  Singular denomination of the registered class
      optplural "Optional plural", e.g. "class(es)" for "class" (optional)
    """
    # Value deduction
    if optplural is None:
      optplural = singular + "(s)"
    # Finalization
    self.__denoms = (singular, optplural)
    self.__register = {}

  def itemize(self):
    """ Build an iterable over the available class names.
    Returns:
      Iterable over the available class names
    """
    return self.__register.keys()

  def register(self, name, cls):
    """ Register a new class.
    Args:
      name Class name
      cls  Associated class
    """
    # Assertions
    assert name not in self.__register, "Name " + repr(name) + " already in use while registering " + repr(getattr(cls, "__name__", "<unknown " + self.__denoms[0] + " class name>"))
    # Registering
    self.__register[name] = cls

  def instantiate(self, name, *args, **kwargs):
    """ Instantiate a registered class.
    Args:
      name Class name
      ...  Forwarded parameters
    Returns:
      Registered class instance
    """
    # Assertions
    if name not in self.__register:
      cause = "Unknown name " + repr(name) + ", "
      if len(self.__register) == 0:
        cause += "no registered " + self.__denoms[0]
      else:
        cause += "available " + self.__denoms[1] + ": '" + ("', '").join(self.__register.keys()) + "'"
      raise tools.UserException(cause)
    # Instantiation
    return self.__register[name](*args, **kwargs)

# ---------------------------------------------------------------------------- #
# Simple list of "<key>:<value>" into dictionary parser

def parse_keyval(list_keyval, defaults={}):
  """ Parse list of "<key>:<value>" into a dictionary.
  Args:
    list_keyval List of "<key>:<value>"
    defaults    Default key -> value to use (also ensure type, 'str' assumed for other keys)
  Returns:
    Associated dictionary
  """
  parsed = {}
  # Parsing
  sep = ":"
  for entry in list_keyval:
    pos = entry.find(sep)
    if pos < 0:
      raise tools.UserException("Expected list of " + repr("<key>:<value>") + ", got " + repr(entry) + " as one entry")
    key = entry[:pos]
    if key in parsed:
      raise tools.UserException("Key " + repr(key) + " had already been specified with value " + repr(parsed[key]))
    val = entry[pos + len(sep):]
    if key in defaults: # Assert type constructibility
      try:
        val = type(defaults[key])(val)
      except Exception:
        raise tools.UserException("Required key " + repr(key) + " expected a value of type " + repr(getattr(type(defaults[key]), "__name__", "<unknown>")))
    parsed[key] = val
  # Add default values (done first to be able to force a given type with 'required')
  for key in defaults:
    if key not in parsed:
      parsed[key] = defaults[key]
  # Return final dictionary
  return parsed

def print_args(name, selected, list_keyval, head="[ARGS] "):
  """ Print the given list of string key-value arguments.
  Args:
    selected    Name of the selected instance
    list_keyval Given list of string key-value arguments
    head        Text to prepend to every line
  """
  print(head + "Selected " + name + ": " + (selected if selected else "<none>"))
  parsed = parse_keyval(list_keyval)
  if len(parsed) > 0:
    for key, val in parsed.items():
      print(head + "· " + key + ": " + val)

# ---------------------------------------------------------------------------- #
# Simple temporary path expansion context manager

class ExpandPath:
  """ Simple temporary path expansion context manager class.
  """

  def __init__(self, *paths):
    """ Register constructor.
    Args:
      ... Path to temporary add to 'sys.path'
    """
    # Finalize
    self.__exp = [str(path) for path in paths]
    self.__old = None

  def __enter__(self):
    """ Enter context.
    Returns:
      self
    """
    self.__old = sys.path
    sys.path   = sys.path + self.__exp
    return self

  def __exit__(self, *args, **kwargs):
    """ Leave context.
    Args:
      ... Ignored arguments
    """
    sys.path = self.__old

# ---------------------------------------------------------------------------- #
# Interface, pointer-implementation class builder

def make_interface(_create, _destroy, **methods):
  """ Interface, pointer-implementation class builder.
  Args:
    _create  Creation (allocation + initialization) function, (...) -> void*
    _destroy Destruction (finalization + deallocation) function, (void*) -> void
    ...      Map method name to function, (void*, ...) -> ...
  Returns:
    Class providing the given methods
  """
  # Constants
  nname = "_native"
  # Class definition
  class Interface:
    """ Native interface, pointer-implementation class.
    """
    def __init__(self, *args):
      """ Creation constructor.
      Args:
        ... Forwarded arguments
      """
      setattr(self, nname, _create(*args))
    def __del__(self):
      """ Destructor finalization.
      """
      if hasattr(self, nname):
        _destroy(getattr(self, nname))
    def __getattr__(self, name):
      """ Get a method from its name.
      Args:
        name Method name
      Returns:
        Bound method value
      """
      # Assertions
      if not hasattr(self, nname):
        raise tools.UserException("Unable to access instance as its creation failed")
      # Get method from name
      method = methods[name]
      native = getattr(self, nname)
      # Bind wrapping
      def call(*args):
        """ Call the method with the first parameter as the current instance.
        Args:
          ... Forwarded next parameters
        Returns:
          Forwarded return value
        """
        return method(native, *args)
      return call
    def __call__(self):
      """ Get a pointer to the native instance.
      Returns:
        Pointer t
      """
      # Assertions
      if not hasattr(self, nname):
        raise tools.UserException("Unable to access instance as its creation failed")
      # Return pointer
      return getattr(self, nname)
  # Post-assertions
  if len(set(dir() + [nname]).intersection(set(methods.keys()))) > 0:
    raise tools.UserException("Method name " + repr(nname) + " is reserved")
  # Return class
  return Interface
