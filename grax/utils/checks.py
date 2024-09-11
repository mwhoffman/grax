"""Assertion checks.
"""

from collections.abc import Sequence

import jax.dtypes
import jax.numpy as jnp

from grax.typing import ArrayLike, DTypeLike
from grax.utils import repr


def check_type(
    array: ArrayLike,
    expected_dtypes: DTypeLike | Sequence[DTypeLike],
) -> None:
  """Check the type of the given array.

  Raise a `ValueError` if `array` does not match the `expected_dtypes`. If
  multiple dtypes are given then `array` must match one of them.
  """
  # Get the dtype of the given array.
  dtype = jax.dtypes.canonicalize_dtype(jax.dtypes.scalar_type_of(array))

  # Make sure we have a sequence of dtypes.
  if not isinstance(expected_dtypes, Sequence):
    expected_dtypes = [expected_dtypes]

  # Canonicalize the given dtypes.
  expected_dtypes = [jax.dtypes.canonicalize_dtype(d) for d in expected_dtypes]

  # Check all our dtypes.
  for expected_dtype in expected_dtypes:
    if jnp.issubdtype(dtype, expected_dtype):
      return

  raise ValueError(
    f"Array dtype ({str(dtype)}) does not match an expected dtype: " +
    f"{repr.join(expected_dtypes)}"
  )


def check_rank(
    array: ArrayLike,
    expected_ranks: int | set[int],
) -> None:
    """Check the rank of a given array.

    Raise a `ValueError` if `array` does not match the `expected_ranks`. If a
    set of ranks is given then `array` must match one of them.
    """
    # Get the rank of the array.
    rank = len(jnp.shape(array))

    # Make sure we have a set of ranks.
    if not isinstance(expected_ranks, set):
        expected_ranks = {expected_ranks}

    if rank not in expected_ranks:
        raise ValueError(
            f"Array rank ({rank}) does not match an expected rank: "
            f"{repr.join(expected_ranks)}"
        )


def check_shape(
    array: ArrayLike,
    expected_shape: Sequence[int | None],
) -> None:
  """Check the shape of a given array.

  Raise a `ValueError` if `array` does not match the `expected_shape`. If any of
  the dimensions of `expected_shape` are `None` then the corresponding shape of
  `array` is ignored.
  """
  # Fail early if we're not of the right rank.
  check_rank(array, len(expected_shape))

  # Check each passed 
  shape = jnp.shape(array)
  for dim, expected_dim in zip(shape, expected_shape):
    if expected_dim is not None and dim != expected_dim:
      raise ValueError(
        f"Array shape ({repr.join(shape)}) does not match the expected shape: " +
        f"({repr.join(expected_shape)})"
      )


def check_equal_shape(*arrays: ArrayLike) -> None:
  """Check the shapes of all arrays match.
  """
  if len(arrays) == 0:
    return

  # Get the shape of the first array.
  shape = jnp.shape(arrays[0])

  # Check the shape of all arrays against the first.
  for array in arrays[1:]:
    check_shape(array, shape)


def check_type_and_rank(
    array: ArrayLike,
    expected_dtypes: DTypeLike | Sequence[DTypeLike],
    expected_ranks: int | set[int],
) -> None:
  """Check the dtype and rank of the given array.

  Raise a `ValueError` if `array` does not match the `expected_dtypes` and the
  `expected_ranks`. This is a convenience function calling `check_type` and
  `check_rank`; see those functions for more details.
  """
  check_type(array, expected_dtypes)
  check_rank(array, expected_ranks)


def check_type_and_shape(
    array: ArrayLike,
    expected_dtypes: DTypeLike | Sequence[DTypeLike],
    expected_shape: Sequence[int | None],
) -> None:
  """Check the dtype and shape of the given array.

  Raise a `ValueError` if `array` does not match the `expected_dtypes` and the
  `expected_shape`. This is a convenience function calling `check_type` and
  `check_shape`; see those functions for more details.
  """
  check_type(array, expected_dtypes)
  check_shape(array, expected_shape)


def check_positive(
    array: ArrayLike,
) -> None:
  """Check that the given array is positive."""
  if not jnp.all(jnp.greater(array, 0)):
    raise ValueError("Array is not positive")

