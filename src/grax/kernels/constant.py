"""Implementation of the constant kernel."""

import dataclasses

import jax
import jax.numpy as jnp
import jaxtyping as jt

from grax import base
from grax import checks
from grax.kernels import base as kernels_base


@base.typed
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class ConstantParams:
  """Parameters of the constant kernel, stored in log-space.

  Attributes:
    logrho: the log of the constant covariance.
  """

  logrho: jt.Float[jt.Array, ""]


@base.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class ConstantKernel(kernels_base.Kernel[ConstantParams]):
  """The constant kernel, i.e. k(x, x') = rho for all inputs.

  `rho` is the initial value of the kernel's hyperparameter, used by `init` to
  construct its params. It is not updated by fitting: the fitted value lives
  in the params held by the GP (i.e. `logrho`).

  Attributes:
    dim: the dimensionality of the kernel's inputs.
    rho: the initial constant covariance; defaults to 1 if unset.
  """

  dim: int
  rho: jt.Float[jt.ArrayLike, ""] | None = None

  @property
  def shape(self) -> tuple[int, ...]:
    """The expected shape of a single kernel input (excluding batch dims)."""
    return (self.dim,)

  @base.typed
  def init(self) -> ConstantParams:
    """Construct parameters for the kernel from rho.

    Falls back to a default constant covariance of 1 if `rho` is unset.

    Returns:
      The parameters of the kernel, taken directly from `rho`.
    """
    rho = self.rho if self.rho is not None else 1.0
    return ConstantParams(logrho=jnp.log(jnp.asarray(rho)))

  @base.typed
  def __call__(
    self,
    params: ConstantParams,
    x1: jt.Float[jt.Array, "n d"],
    x2: jt.Float[jt.Array, "m d"],
  ) -> jt.Float[jt.Array, "n m"]:
    """Evaluate the kernel on given inputs.

    Args:
      params: the parameters of the kernel.
      x1: the first set of kernel inputs.
      x2: the second set of kernel inputs.

    Returns:
      An array K of shape (n, m) where n and m are the batch dimensions of
      `x1` and `x2` respectively, with every entry equal to `rho`.
    """
    checks.check_shape(x1, (None, self.dim))
    checks.check_shape(x2, (None, self.dim))

    rho = jnp.exp(params.logrho)
    return jnp.full((x1.shape[0], x2.shape[0]), rho)

  @base.typed
  def diag(
    self,
    params: ConstantParams,
    x: jt.Float[jt.Array, "n d"],
  ) -> jt.Float[jt.Array, " n"]:
    """Evaluate the diagonal kernel matrix on given inputs.

    Args:
      params: the parameters of the kernel.
      x: the set of kernel inputs.

    Returns:
      An n-vector with every entry equal to `rho`.
    """
    checks.check_shape(x, (None, self.dim))

    rho = jnp.exp(params.logrho)
    return jnp.full(x.shape[0], rho)
