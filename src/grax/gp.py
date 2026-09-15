"""Implementation of a GP."""

import dataclasses
from typing import Generic
from typing import TypeVar

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import jaxtyping as jt

from grax import base
from grax import checks
from grax.kernels import base as kernels_base
from grax.means import base as means_base


KernelParams = TypeVar("KernelParams")
MeanParams = TypeVar("MeanParams")


@base.typed
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class GPParams(Generic[KernelParams, MeanParams]):
  """Parameters of a GP.

  Attributes:
    kernel: the parameters of the GP's kernel.
    mean: the parameters of the GP's mean function.
    logsn2: the log of the observation noise variance.
  """

  kernel: KernelParams
  mean: MeanParams
  logsn2: jt.Float[jt.Array, ""]


@base.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class GPData:
  """Observed input/output data.

  Attributes:
    x: the observed inputs.
    y: the observed outputs.
  """

  x: jt.Float[jt.Array, "n ..."]
  y: jt.Float[jt.Array, " n"]


@base.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class GPStatistics:
  """Sufficient statistics for making GP posterior predictions.

  Attributes:
    L: the lower Cholesky factor of the noisy kernel matrix.
    r: the residual of the observations relative to the prior mean.
    a: the solve `K^-1 r`, i.e. `cho_solve((L, True), r)`.
  """

  L: jt.Float[jt.Array, "n n"]
  r: jt.Float[jt.Array, " n"]
  a: jt.Float[jt.Array, " n"]


class GP(Generic[KernelParams, MeanParams]):
  """A Gaussian process combining a kernel, mean function, and noise level."""

  def __init__(
    self,
    kernel: kernels_base.Kernel[KernelParams],
    mean: means_base.Mean[MeanParams],
    sn2: float = 1.0,
  ) -> None:
    """Initialize the GP with constituent models.

    Args:
      kernel: the kernel modeling the covariance between inputs.
      mean: the mean function modeling the prior expected output.
      sn2: the initial observation noise variance.
    """
    if kernel.shape != mean.shape:
      msg = f"kernel.shape {kernel.shape} != mean.shape {mean.shape}."
      raise checks.CheckError(msg)

    self._kernel = kernel
    self._mean = mean
    self._params = GPParams(
      kernel=kernel.init(),
      mean=mean.init(),
      logsn2=jnp.log(jnp.asarray(sn2)),
    )

    self._data: GPData | None = None

  @base.typed
  def add_data(
    self,
    x: jt.Float[jt.ArrayLike, "n ..."],
    y: jt.Float[jt.ArrayLike, " n"],
  ) -> None:
    """Add observed data, appending to any existing observations.

    Args:
      x: the observed inputs.
      y: the observed outputs.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    checks.check_shape(x, (None, *self._kernel.shape))

    if self._data is None:
      self._data = GPData(x=x, y=y)
    else:
      self._data = GPData(
        x=jnp.concatenate([self._data.x, x]),
        y=jnp.concatenate([self._data.y, y]),
      )

  @base.typed
  def _get_stats(self) -> GPStatistics | None:
    """Compute the sufficient statistics needed for posterior prediction."""
    if self._data is None:
      return None

    sn2 = jnp.exp(self._params.logsn2)
    k = self._kernel(self._params.kernel, self._data.x, self._data.x)
    k = k + sn2 * jnp.eye(self._data.x.shape[0])
    r = self._data.y - self._mean(self._params.mean, self._data.x)

    chol = jla.cholesky(k, lower=True)
    a = jla.cho_solve((chol, True), r)

    return GPStatistics(L=chol, r=r, a=a)

  @base.typed
  def predict(
    self,
    x: jt.Float[jt.ArrayLike, "m ..."],
  ) -> tuple[jt.Float[jt.Array, " m"], jt.Float[jt.Array, " m"]]:
    """Predict the latent function at the given input points.

    Args:
      x: the input points to predict at.

    Returns:
      A tuple `(mu, s2)` of the predicted mean and variance at each input
      point. `s2` is the variance of the latent function itself and does
      not include observation noise.
    """
    x = jnp.asarray(x)
    checks.check_shape(x, (None, *self._kernel.shape))

    mu = self._mean(self._params.mean, x)
    s2 = self._kernel.diag(self._params.kernel, x)

    stats = self._get_stats()
    if stats is None or self._data is None:
      return mu, s2

    k = self._kernel(self._params.kernel, self._data.x, x)
    v = jla.solve_triangular(stats.L, k, lower=True)

    mu = mu + k.T @ stats.a
    s2 = s2 - jnp.sum(v**2, axis=0)

    return mu, s2
