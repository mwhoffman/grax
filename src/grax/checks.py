"""Runtime shape assertion checks."""

from collections.abc import Sequence

import jax.numpy as jnp
import jaxtyping as jt


class CheckError(ValueError):
  """Error raised when a check fails."""


def check_shape(
  array: jt.ArrayLike | Sequence[float],
  expected_shape: Sequence[int | None],
  *,
  promote: bool = False,
) -> None:
  """Check that `array`'s shape matches `expected_shape`.

  Args:
    array: the array to check.
    expected_shape: the expected shape; a `None` entry matches any size
      along that axis.
    promote: whether to prepend axes to `array` until it has as many as
      `expected_shape` before comparing, e.g. so that a scalar matches `(1,)`.
      This should only be used for values that are then promoted the same way
      by the caller, and not for batched inputs, where a missing batch axis is
      an error.

  Raises:
    CheckError: if `array`'s shape doesn't match `expected_shape`.
  """
  # Converting first also lets `array` be a sequence, which `jnp.shape` is
  # deprecating support for.
  original_shape = jnp.asarray(array).shape
  shape = original_shape
  if promote:
    shape = jnp.array(array, ndmin=len(expected_shape)).shape
  matches = len(shape) == len(expected_shape) and all(
    expected is None or actual == expected
    for actual, expected in zip(shape, expected_shape, strict=True)
  )
  if not matches:
    msg = f"array has shape {original_shape}, expected {tuple(expected_shape)}."
    raise CheckError(msg)


def check_none_or_shape(
  array: jt.ArrayLike | Sequence[float] | None,
  expected_shape: Sequence[int | None],
  *,
  promote: bool = False,
) -> None:
  """Check that `array`'s shape matches `expected_shape`, unless it's None.

  Args:
    array: the array to check, or None, in which case this is a no-op.
    expected_shape: the expected shape; a `None` entry matches any size
      along that axis.
    promote: whether to promote `array` before comparing; see `check_shape`.

  Raises:
    CheckError: if `array` is not None and its shape doesn't match
      `expected_shape`.
  """
  if array is not None:
    check_shape(array, expected_shape, promote=promote)
