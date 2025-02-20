"""Implementation of the zero-mean function."""

import jax.numpy as jnp

from grax import types
from grax.means import base
from grax.utils import checks


class Zero(base.Mean):
  """The zero-mean function."""

  def __init__(self, dim: int):
    """The zero mean function.

    This function should return zeros for all inputs and will sanity-check that
    the inputs match the required dimension.

    Args:
      dim: the dimensions of the function.
    """
    checks.check_positive(dim)

    self._dim = dim

  @property
  def dim(self) -> int:
    """Return the input dimension."""
    return self._dim

  def __call__(self, x: types.ArrayLike) -> types.Array:
    """Evaluate the mean on the given inputs."""
    x = jnp.asarray(x)
    checks.check_type_and_shape(x, types.Float, (None, self.dim))
    return jnp.zeros(x.shape[0])
