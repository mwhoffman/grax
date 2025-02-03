"""Test helpers."""

import inspect
import pathlib
from typing import Callable

import numpy as np
import numpy.testing as nt
import pytest


def parameterize_goldens(*inputs: dict) -> Callable:
  """Parameterize a collection of golden tests.

  Given a collection of testing `inputs` this provides a wrapper of the form
  `paremeterize(func)` which will run `func(**inputs[i])` and compare its output
  previously run and saved golden outputs. Testing is performed using
  `numpy.testing.assert_allclose`.
  """
  # Find the directory storing goldens for the current module.
  path = pathlib.Path(inspect.stack()[1].filename)
  goldendir = path.parent / "goldens" / path.stem

  # Create the decorator.
  def parameterize(func: Callable) -> Callable:
    # The following wrapper will be applied for each individual input that
    # parameterizes the test case.
    @pytest.mark.parametrize(("n", "input_"), enumerate(inputs))
    def wrapper(n: int, input_: dict, request: pytest.FixtureRequest):
      # Check for pytest options.
      save_goldens = request.config.getoption("--save-goldens")
      update_goldens = request.config.getoption("--update-goldens")

      # Evaluate the wrapped function.
      output = func(**input_)

      # Use the function name and the number of parameterized inputs to find the
      # file storing the golden output.
      goldenfile = goldendir / f"{func.__name__}_{n}.npy"

      try:
        # Try to load the golden output.
        with goldenfile.open("rb") as f:
          golden = np.load(f)

      except FileNotFoundError:
        if save_goldens:
          # If --save-goldens is given and no golden file exists save the
          # goldens and mark the test as skipped.
          goldenfile.parent.mkdir(parents=True, exist_ok=True)
          with goldenfile.open("wb") as f:
            np.save(f, output)
          pytest.skip("Saving over empty golden file")

        else:
          # Otherwise re-raise the exception.
          raise

      try:
        # Assert that the output matches the golden output.
        nt.assert_allclose(output, golden, atol=1e-6, rtol=1e-6)

      except AssertionError:
        if update_goldens:
          # If --update-goldens is given save the goldens and mark the test as
          # skipped.
          goldenfile.parent.mkdir(parents=True, exist_ok=True)
          with goldenfile.open("wb") as f:
            np.save(f, output)
          pytest.skip("Saving over a failing golden file")

        else:
          # Otherwise re-raise the exception.
          raise

    # This is the wrapped function, this is what will ultimately exist at the
    # top level, i.e. this is the result of applying the decorator onto the
    # test function.
    return wrapper

  # Return the decorator which will be applied to the test function.
  return parameterize
