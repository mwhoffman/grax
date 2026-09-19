"""Implementation of the zero-mean function."""

import dataclasses

import jax.numpy as jnp
import jaxtyping as jt

from grax import base
from grax import checks
from grax.means import base as means_base


@dataclasses.dataclass(frozen=True, kw_only=True)
class ZeroMean(means_base.Mean[None]):
  """The zero-mean function.

  Attributes:
    dim: the dimensionality of the mean function's inputs.
  """

  dim: int

  @property
  def shape(self) -> tuple[int, ...]:
    """The expected shape of a single input (excluding batch dims)."""
    return (self.dim,)

  def init(self) -> None:
    """Construct parameters for the mean function; there are none."""
    return

  @base.typed
  def __call__(
    self,
    params: None,
    x: jt.Float[jt.Array, "n d"],
  ) -> jt.Float[jt.Array, " n"]:
    """Evaluate the mean function on given inputs.

    Args:
      params: the parameters of the mean function (unused; always None).
      x: the set of input points.

    Returns:
      An n-vector of zeros, one per input point.
    """
    del params
    checks.check_shape(x, (None, self.dim), name="x")
    return jnp.zeros(x.shape[0])
