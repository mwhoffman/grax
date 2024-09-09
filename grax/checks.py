"""Assertion checks.
"""

from typing import Sequence, Set
from grax.typing import ArrayLike, DTypeLike

import jax.dtypes
import jax.numpy as jnp


def check_type(
    array: ArrayLike,
    expected_dtypes: DTypeLike | Sequence[DTypeLike],
) -> None:
  # Get the dtype of the given array.
  dtype = jax.dtypes.canonicalize_dtype(jax.dtypes.scalar_type_of(array))

  # Make sure we have a sequence of dtypes.
  if not isinstance(expected_dtypes, Sequence):
    expected_dtypes = (expected_dtypes,)

  for expected_dtype in expected_dtypes:
    if jnp.issubdtype(dtype, expected_dtype):
      return

  raise ValueError('Array does not match the expected dtype(s).')


def check_rank(
    array: ArrayLike,
    expected_ranks: int | Set[int],
) -> None:
  # Get the rank of the array.
  rank = len(jnp.shape(array))

  if not isinstance(expected_ranks, Set):
    expected_ranks = {expected_ranks}

  if rank not in expected_ranks:
    raise ValueError('Array does not match the expected rank(s).')


def check_dimensions(
    array: ArrayLike,
    expected_dimensions: Sequence[int | None],
) -> None:
  check_rank(array, len(expected_dimensions))
  for dim, expected_dim in zip(jnp.shape(array), expected_dimensions):
    if expected_dim is not None and dim != expected_dim:
      raise ValueError('Array does not match the expected dimensions.')


def check_type_and_rank(
    array: ArrayLike,
    expected_dtypes: DTypeLike | Sequence[DTypeLike],
    expected_ranks: int | Set[int],
) -> None:
  check_type(array, expected_dtypes)
  check_rank(array, expected_ranks)


def check_type_and_dimensions(
    array: ArrayLike,
    expected_dtypes: DTypeLike | Sequence[DTypeLike],
    expected_dimensions: Sequence[int | None],
) -> None:
  check_type(array, expected_dtypes)
  check_dimensions(array, expected_dimensions)

