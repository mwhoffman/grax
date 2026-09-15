"""Runtime shape assertion checks."""

from collections.abc import Sequence

import jax.numpy as jnp
import jaxtyping as jt


class CheckError(ValueError):
  """Error raised when a check fails."""


def check_shape(
  array: jt.ArrayLike,
  expected_shape: Sequence[int | None],
) -> None:
  """Check that `array`'s shape matches `expected_shape`.

  Args:
    array: the array to check.
    expected_shape: the expected shape; a `None` entry matches any size
      along that axis.

  Raises:
    CheckError: if `array`'s shape doesn't match `expected_shape`.
  """
  shape = jnp.shape(array)
  matches = len(shape) == len(expected_shape) and all(
    expected is None or actual == expected
    for actual, expected in zip(shape, expected_shape, strict=True)
  )
  if not matches:
    msg = f"array has shape {shape}, expected {tuple(expected_shape)}."
    raise CheckError(msg)


def check_none_or_shape(
  array: jt.ArrayLike | None,
  expected_shape: Sequence[int | None],
) -> None:
  """Check that `array`'s shape matches `expected_shape`, unless it's None.

  Args:
    array: the array to check, or None, in which case this is a no-op.
    expected_shape: the expected shape; a `None` entry matches any size
      along that axis.

  Raises:
    CheckError: if `array` is not None and its shape doesn't match
      `expected_shape`.
  """
  if array is not None:
    check_shape(array, expected_shape)
