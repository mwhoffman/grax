"""Test helpers.
"""

import inspect
import numpy.testing as nt
import os.path
import pickle
import pytest


def parameterize_goldens(filename: str, *inputs: dict):
  """Parameterize a collection of golden tests.

  Given a `filename` storing golden outputs and a collection of testing `inputs`
  this provides a wrapper of the form `paremeterize(func)` which will run
  `func(**inputs[i])` and compare its output against `outputs[i]` as read from
  the given (pickled) file using `numpy.testing.assert_allclose`.
  """
  # Get the directory of the caller's file which we'll use to look for
  # `filename` specifying the golden outputs.
  dirname = os.path.dirname(inspect.stack()[1].filename)

  # Get the golden outputs.
  with open(os.path.join(dirname, filename), "rb") as f:
    goldens = pickle.load(f)

  if len(inputs) != len(goldens):
    raise ValueError(f'The number of inputs ({len(inputs)}) doesn\'t match ' +
                     f'the number of golden outputs ({len(goldens)})')

  # Create the decorator.
  def parameterize(func):
    # This parameterizes the decorated function with the goldens, i.e. it
    # creates a single test instance for each element of the goldens list.
    @pytest.mark.parametrize("input,golden", zip(inputs, goldens))
    def wrapper(input, golden):
      # This intermediate wrapper just calls the test function with the golden
      # inputs and then compares the output against the expected version,
      # raising an error unless the result is "close" as defined by numpy.
      nt.assert_allclose(func(**input), golden)

    # This is the wrapped function, i.e. the result of applying the decorator.
    return wrapper

  # Return the decorator.
  return parameterize

