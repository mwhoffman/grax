"""Test helpers.
"""

import inspect
import numpy.testing as nt
import os.path
import pickle
import pytest


def parameterize_goldens(*inputs: dict):
  """Parameterize a collection of golden tests.

  Given a collection of testing `inputs` this provides a wrapper of the form 
  `paremeterize(func)` which will run `func(**inputs[i])` and compare its output
  previously run and saved golden outputs. Testing is performed using 
  `numpy.testing.assert_allclose`.
  """
  # Get the dirname and basename of the caller's file and remove the extension.
  dirname, basename = os.path.split(inspect.stack()[1].filename)
  basename = os.path.splitext(basename)[0]

  # Create the decorator.
  def parameterize(func):
    # This parameterizes the decorated function with the given inputs and wraps
    # it so that each parameterized call will look up the golden outputs and
    # compare against the output of the function.
    @pytest.mark.parametrize("n, input", enumerate(inputs))
    def wrapper(n, input):
      # Open the golden file alongside the file of the caller.
      with open(os.path.join(dirname, 'goldens', basename,
                             func.__name__ + f'_{n}.pkl'), 'rb') as f:
        # Get the golden outputs.
        golden = pickle.load(f)

      # This intermediate wrapper just calls the test function with the golden
      # inputs and then compares the output against the expected version,
      # raising an error unless the result is "close" as defined by numpy.
      nt.assert_allclose(func(**input), golden)

    # This is the wrapped function, i.e. the result of applying the decorator.
    return wrapper

  # Return the decorator.
  return parameterize

