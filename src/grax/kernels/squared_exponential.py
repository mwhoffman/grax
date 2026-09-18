"""Implementation of the squared-exponential kernel."""

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
class SEParams:
  """Parameters of the squared-exponential kernel, stored in log-space.

  Attributes:
    logell: the log of the lengthscales, one per input dimension.
  """

  logell: jt.Float[jt.Array, " d"]


@base.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class SEKernel(kernels_base.Kernel[SEParams]):
  """The squared-exponential kernel.

  The kernel has unit output variance; multiply it by a `ConstantKernel` (or a
  positive float) to scale it.

  `ell` is the initial value of the kernel's hyperparameter, used by `init` to
  construct its params. It is not updated by fitting: the fitted value lives in
  the params held by the GP (i.e. `logell`).

  Attributes:
    dim: the dimensionality of the kernel's inputs.
    ell: the initial lengthscale(s); if not given, defaults to a vector of
      ones of length `dim`.
  """

  dim: int
  ell: jt.Float[jt.ArrayLike, " d"] | None = None

  def __post_init__(self) -> None:
    """Check that, if given, ell has the right shape."""
    checks.check_none_or_shape(self.ell, (self.dim,))

  @property
  def shape(self) -> tuple[int, ...]:
    """The expected shape of a single kernel input (excluding batch dims)."""
    return (self.dim,)

  @base.typed
  def init(self) -> SEParams:
    """Construct parameters for the kernel from ell.

    Falls back to a default lengthscale of 1 per dimension if `ell` is unset.

    Returns:
      The parameters of the kernel, taken directly from `ell`.
    """
    ell = self.ell if self.ell is not None else jnp.ones(self.dim)
    return SEParams(logell=jnp.log(jnp.asarray(ell)))

  @base.typed
  def __call__(
    self,
    params: SEParams,
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
      `x1` and `x2` respectively. K[i, j] is the kernel function evaluated
      on k(x1[i], x2[j]).
    """
    checks.check_shape(params.logell, (self.dim,))
    checks.check_shape(x1, (None, self.dim))
    checks.check_shape(x2, (None, self.dim))

    ell = jnp.exp(params.logell)

    scaled1 = x1 / ell
    scaled2 = x2 / ell
    sqdist1 = jnp.sum(scaled1**2, axis=-1, keepdims=True)
    sqdist2 = jnp.sum(scaled2**2, axis=-1)
    sqdist = sqdist1 - 2 * jnp.matmul(scaled1, scaled2.T) + sqdist2

    return jnp.exp(-jnp.clip(sqdist, min=0) / 2)

  @base.typed
  def diag(
    self,
    params: SEParams,
    x: jt.Float[jt.Array, "n d"],
  ) -> jt.Float[jt.Array, " n"]:
    """Evaluate the diagonal kernel matrix on given inputs.

    Args:
      params: the parameters of the kernel.
      x: the set of kernel inputs.

    Returns:
      The pairwise kernel evaluating each element of `x` against itself,
      i.e. k[i] is the kernel function evaluated on k(x[i], x[i]).
    """
    checks.check_shape(params.logell, (self.dim,))
    checks.check_shape(x, (None, self.dim))

    return jnp.ones(x.shape[0])
