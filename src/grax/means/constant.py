"""Implementation of the constant mean function."""

import dataclasses

import jax
import jax.numpy as jnp
import jaxtyping as jt

from grax import base
from grax import checks
from grax.means import base as means_base


@base.typed
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class ConstantParams:
  """Parameters of the constant mean function.

  Attributes:
    offset: the constant value predicted everywhere.
  """

  offset: jt.Float[jt.Array, ""]


@base.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class ConstantMean(means_base.Mean[ConstantParams]):
  """The constant mean function.

  `offset` is the initial value of the mean function's parameter, used by
  `init` to construct its params. It is not updated by fitting: the fitted
  value lives in the params held by the GP.

  Attributes:
    dim: the dimensionality of the mean function's inputs.
    offset: the initial constant offset; defaults to 0 if unset.
  """

  dim: int
  offset: jt.Float[jt.ArrayLike, ""] | None = None

  @property
  def shape(self) -> tuple[int, ...]:
    """The expected shape of a single input (excluding batch dims)."""
    return (self.dim,)

  @base.typed
  def init(self) -> ConstantParams:
    """Construct parameters for the mean function from offset.

    Falls back to a default offset of 0 if `offset` is unset.

    Returns:
      The parameters of the mean function.
    """
    offset = self.offset if self.offset is not None else 0.0
    return ConstantParams(offset=jnp.asarray(offset, dtype=float))

  @base.typed
  def __call__(
    self,
    params: ConstantParams,
    x: jt.Float[jt.Array, "n d"],
  ) -> jt.Float[jt.Array, " n"]:
    """Evaluate the mean function on given inputs.

    Args:
      params: the parameters of the mean function.
      x: the set of input points.

    Returns:
      An n-vector with `params.offset` repeated for each input point.
    """
    checks.check_shape(x, (None, self.dim))
    return jnp.full(x.shape[0], params.offset)
