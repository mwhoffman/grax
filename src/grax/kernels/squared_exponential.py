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
    logrho: the log of the output variance.
    logell: the log of the lengthscales, one per input dimension.
  """

  logrho: jt.Float[jt.Array, ""]
  logell: jt.Float[jt.Array, " d"]


@base.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class SEKernel(kernels_base.Kernel[SEParams]):
  """The squared-exponential kernel.

  `rho` and `ell` are the initial values of the kernel's hyperparameters, used
  by `init` to construct its params. They are not updated by fitting: the
  fitted values live in the params held by the GP (e.g. `logrho`/`logell`).

  Attributes:
    dim: the dimensionality of the kernel's inputs.
    rho: the initial output variance; defaults to 1 if unset.
    ell: the initial lengthscale(s); if not given, defaults to a vector of
      ones of length `dim`.
  """

  dim: int
  rho: jt.Float[jt.ArrayLike, ""] | None = None
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
    """Construct parameters for the kernel from rho and ell.

    Falls back to a default output variance of 1 and a default lengthscale
    of 1 per dimension for whichever of `rho`/`ell` is unset.

    Returns:
      The parameters of the kernel, taken directly from `rho` and `ell`.
    """
    rho = self.rho if self.rho is not None else 1.0
    ell = self.ell if self.ell is not None else jnp.ones(self.dim)
    return SEParams(
      logrho=jnp.log(jnp.asarray(rho)),
      logell=jnp.log(jnp.asarray(ell)),
    )

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

    rho = jnp.exp(params.logrho)
    ell = jnp.exp(params.logell)

    scaled1 = x1 / ell
    scaled2 = x2 / ell
    sqdist1 = jnp.sum(scaled1**2, axis=-1, keepdims=True)
    sqdist2 = jnp.sum(scaled2**2, axis=-1)
    sqdist = sqdist1 - 2 * jnp.matmul(scaled1, scaled2.T) + sqdist2

    return rho * jnp.exp(-jnp.clip(sqdist, min=0) / 2)

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

    rho = jnp.exp(params.logrho)
    return jnp.full(x.shape[0], rho)
