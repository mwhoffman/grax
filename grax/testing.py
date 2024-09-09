"""Test helpers.
"""

import inspect
import numpy as np
import os.path
import pickle
import pytest


def parameterize_goldens(filename):
  # Get the directory of the caller's file and prepend the passed filename.
  dirname = os.path.dirname(inspect.stack()[1].filename)
  filename = os.path.join(dirname, filename)

  # Get the goldens which should be a list of tuples whose the final element is
  # the expected output and all other elements are passed into the decorated
  # test function.
  with open(filename, "rb") as f:
    goldens = pickle.load(f)

  # Create the decorator.
  def parameterize(func):
    # This parameterizes the decorated function with the goldens, i.e. it
    # creates a single test instance for each element of the goldens list.
    @pytest.mark.parametrize("golden", goldens)
    def wrapper(golden):
      # This intermediate wrapper just calls the test function with the golden
      # inputs and then compares the output against the expected version,
      # raising an error unless the result is "close" as defined by numpy.
      *args, expected = golden
      assert np.allclose(func(*args), expected)

    # This is the wrapped function, i.e. the result of applying the decorator.
    return wrapper

  # Return the decorator.
  return parameterize
