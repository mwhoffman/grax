"""Implementation of the rational quadratic kernel."""

import dataclasses

import jax
import jax.numpy as jnp
import jaxtyping as jt

from grax import base
from grax import checks
from grax.kernels import base as kernels_base
from grax.kernels import distance


@base.typed
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class RQParams:
  """Parameters of the rational quadratic kernel, stored in log-space.

  Attributes:
    logell: the log of the lengthscales, one per input dimension.
    logalpha: the log of the scale mixture parameter alpha.
  """

  logell: jt.Float[jt.Array, " d"]
  logalpha: jt.Float[jt.Array, ""]


@base.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class RQKernel(kernels_base.Kernel[RQParams]):
  """The rational quadratic kernel.

  The kernel is `(1 + D / (2 alpha))^(-alpha)` where `D` is the squared
  euclidean distance between the lengthscale-scaled inputs. It is a scale
  mixture of squared-exponential kernels, and tends to the squared-exponential
  kernel as `alpha` grows. Like that kernel it has unit output variance;
  multiply it by a `ConstantKernel` (or a positive float) to scale it.

  `ell` and `alpha` are the initial values of the kernel's hyperparameters,
  used by `init` to construct its params. They are not updated by fitting: the
  fitted values live in the params held by the GP (i.e. `logell` and
  `logalpha`).

  Attributes:
    dim: the dimensionality of the kernel's inputs.
    ell: the initial lengthscales, a vector of length `dim` (or a scalar if
      `dim == 1`); if not given, defaults to a vector of ones.
    alpha: the initial scale mixture parameter, which must be positive;
      defaults to 1 if unset.
  """

  dim: int
  ell: base.VectorLike | None = None
  alpha: base.ScalarLike | None = None

  def __post_init__(self) -> None:
    """Check that, if given, ell and alpha are valid."""
    # Promoting lets a scalar stand for a length-1 vector, so it is only valid
    # when dim == 1.
    checks.check_none_or_shape(self.ell, (self.dim,), name="ell", promote=True)
    checks.check_none_or_positive(self.ell, name="ell")
    checks.check_none_or_positive(self.alpha, name="alpha")

  @property
  def shape(self) -> tuple[int, ...]:
    """The expected shape of a single kernel input (excluding batch dims)."""
    return (self.dim,)

  @base.typed
  def init(self) -> RQParams:
    """Construct parameters for the kernel from ell and alpha.

    Falls back to a default lengthscale of 1 per dimension and a default alpha
    of 1 for whichever of `ell`/`alpha` is unset.

    Returns:
      The parameters of the kernel, taken directly from `ell` and `alpha`.
    """
    if self.ell is None:
      ell = jnp.ones(self.dim)
    else:
      ell = jnp.array(self.ell, dtype=float, ndmin=1)
    alpha = self.alpha if self.alpha is not None else 1.0
    return RQParams(
      logell=jnp.log(ell),
      logalpha=jnp.log(jnp.asarray(alpha, dtype=float)),
    )

  @base.typed
  def __call__(
    self,
    params: RQParams,
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
    checks.check_shape(params.logell, (self.dim,), name="params.logell")
    checks.check_shape(x1, (None, self.dim), name="x1")
    checks.check_shape(x2, (None, self.dim), name="x2")

    ell = jnp.exp(params.logell)
    alpha = jnp.exp(params.logalpha)
    sqdist = distance.sqdist(x1 / ell, x2 / ell)

    # This is (1 + sqdist / (2 alpha))^(-alpha), but written with log1p as
    # otherwise, in float32, it loses accuracy and eventually rounds to 1 for
    # large alpha.
    return jnp.exp(-alpha * jnp.log1p(sqdist / (2 * alpha)))

  @base.typed
  def diag(
    self,
    params: RQParams,
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
    checks.check_shape(params.logell, (self.dim,), name="params.logell")
    checks.check_shape(x, (None, self.dim), name="x")

    return jnp.ones(x.shape[0])
