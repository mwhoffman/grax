"""Test helpers.
"""

import numpy as np
import os.path
import pickle
import pytest


def parameterize_goldens(python_file):
  # Get goldens from the given filename.
  with open(os.path.splitext(python_file)[0] + '.goldens', 'rb') as f:
    goldens = pickle.load(f)

  def marker(func):
    @pytest.mark.parametrize("golden", goldens)
    def wrapper(golden):
      class_kwargs, call_kwargs, expected = golden
      result = func(class_kwargs, call_kwargs)
      assert np.allclose(result, expected)

    return wrapper

  return marker

