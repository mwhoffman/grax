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
from grax import optimization
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
    logsn2: the log of the observation noise variance in excess of the GP's
      `sn2_min` floor.
  """

  kernel: KernelParams
  mean: MeanParams
  logsn2: jt.Float[jt.Array, ""]


@base.typed
@jax.tree_util.register_dataclass
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
@jax.tree_util.register_dataclass
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

  # Instance params class for convenience so we don't have to repeat the
  # typevars. Bare `Params` only resolves in the signature of a method
  # defined directly in this class body. Everywhere else runs later, in a scope
  # that can't see this namespace, so use `GP.Params` there instead.
  Params = GPParams[KernelParams, MeanParams]

  def __init__(
    self,
    kernel: kernels_base.Kernel[KernelParams],
    mean: means_base.Mean[MeanParams],
    sn2: float = 1.0,
    sn2_min: float = 1e-6,
  ) -> None:
    """Initialize the GP with constituent models.

    Args:
      kernel: the kernel modeling the covariance between inputs.
      mean: the mean function modeling the prior expected output.
      sn2: the initial observation noise variance; must exceed `sn2_min`.
      sn2_min: a floor added to the observation noise variance, which acts as
        regularization and keeps the noisy kernel matrix well conditioned.
    """
    if kernel.shape != mean.shape:
      msg = f"kernel.shape {kernel.shape} != mean.shape {mean.shape}."
      raise checks.CheckError(msg)
    if sn2_min < 0 or sn2 <= sn2_min:
      msg = f"Require 0 <= sn2_min < sn2, got {sn2_min=} and {sn2=}."
      raise checks.CheckError(msg)

    self.__data: GPData | None = None
    self.__data_pending: GPData | None = None
    self.__stats: GPStatistics | None = None
    self.__params: GP.Params = GPParams(
      kernel=kernel.init(),
      mean=mean.init(),
      logsn2=jnp.log(jnp.asarray(sn2 - sn2_min)),
    )

    self._kernel = kernel
    self._mean = mean
    self._sn2_min = sn2_min

    @base.typed
    def compute_stats(
      params: GP.Params,
      data: GPData,
    ) -> GPStatistics:
      sn2_ = jnp.exp(params.logsn2) + sn2_min
      k = kernel(params.kernel, data.x, data.x)
      k = k + sn2_ * jnp.eye(data.x.shape[0])
      r = data.y - mean(params.mean, data.x)

      chol = jla.cholesky(k, lower=True)
      a = jla.cho_solve((chol, True), r)

      return GPStatistics(L=chol, r=r, a=a)

    @base.typed
    def update_stats(
      params: GP.Params,
      data: GPData,
      pending: GPData,
      stats: GPStatistics,
    ) -> GPStatistics:
      # Block Cholesky update: extend the cached factorization of `data`
      # with `pending` rather than refactorizing `data` and `pending`
      # together from scratch. This is the same block partitioning LAPACK's
      # own blocked Cholesky routines use internally, so it's no less
      # numerically stable than a full recompute of the combined data.
      sn2_ = jnp.exp(params.logsn2) + sn2_min
      n_pending = pending.x.shape[0]

      k_np = kernel(params.kernel, data.x, pending.x)
      k_pp = kernel(params.kernel, pending.x, pending.x)
      k_pp = k_pp + sn2_ * jnp.eye(n_pending)
      r_pending = pending.y - mean(params.mean, pending.x)

      w = jla.solve_triangular(stats.L, k_np, lower=True)
      schur = k_pp - w.T @ w
      c = jla.cholesky(schur, lower=True)

      z = jla.solve_triangular(stats.L, stats.r, lower=True)
      z_pending = jla.solve_triangular(c, r_pending - w.T @ z, lower=True)

      a_pending = jla.solve_triangular(c.T, z_pending, lower=False)
      a = jla.solve_triangular(stats.L.T, z - w @ a_pending, lower=False)

      n = data.x.shape[0]
      chol = jnp.block([[stats.L, jnp.zeros((n, n_pending))], [w.T, c]])

      return GPStatistics(
        L=chol,
        r=jnp.concatenate([stats.r, r_pending]),
        a=jnp.concatenate([a, a_pending]),
      )

    @base.typed
    def predict(
      params: GP.Params,
      data: GPData,
      stats: GPStatistics,
      x: jt.Float[jt.Array, "m ..."],
    ) -> tuple[jt.Float[jt.Array, " m"], jt.Float[jt.Array, " m"]]:
      mu = mean(params.mean, x)
      s2 = kernel.diag(params.kernel, x)

      k = kernel(params.kernel, data.x, x)
      v = jla.solve_triangular(stats.L, k, lower=True)

      mu = mu + k.T @ stats.a
      s2 = s2 - jnp.sum(v**2, axis=0)

      return mu, s2

    # jit functions to compute/update stats and predict.
    self._compute_stats = jax.jit(compute_stats)
    self._update_stats = jax.jit(update_stats)
    self._predict = jax.jit(predict)

  @property
  def _params(self) -> Params:
    """The GP's current parameters."""
    return self.__params

  @_params.setter
  def _params(self, params: Params) -> None:
    self.__params = params
    # Any pending data can no longer be incrementally folded into the cache
    # we're about to drop, so fold it into `__data` now -- there's no
    # cached Cholesky left to extend once `__stats` is cleared below anyway.
    if self.__data_pending is not None:
      self.__data = self._data
      self.__data_pending = None
    self.__stats = None

  @property
  def _data(self) -> GPData | None:
    """All observed data."""
    if self.__data is None:
      return self.__data_pending
    if self.__data_pending is None:
      return self.__data
    return GPData(
      x=jnp.concatenate([self.__data.x, self.__data_pending.x]),
      y=jnp.concatenate([self.__data.y, self.__data_pending.y]),
    )

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
    checks.check_shape(x, (None, *self._kernel.shape), name="x")

    if self.__data_pending is None:
      self.__data_pending = GPData(x=x, y=y)
    else:
      self.__data_pending = GPData(
        x=jnp.concatenate([self.__data_pending.x, x]),
        y=jnp.concatenate([self.__data_pending.y, y]),
      )

  @base.typed
  def _statistics(
    self,
    params: Params,
  ) -> GPStatistics | None:
    """Compute the sufficient statistics needed for posterior prediction.

    Cached when `params` is the GP's current parameters (`self._params`); that
    cache is invalidated automatically whenever `self._params` is reassigned.
    Data added since the cache was last computed (via `add_data`) is folded in
    with a cheap block Cholesky update rather than a full recompute. For any
    other `params`, e.g. a candidate value during `fit`'s search, this always
    recomputes fresh over all the data and never reads or writes the cache.

    Args:
      params: the parameters of the GP to compute statistics for.
    """
    data = self._data
    if data is None:
      return None

    if params is not self._params:
      return self._compute_stats(params, data)

    if self.__stats is not None and self.__data_pending is None:
      return self.__stats

    if self.__stats is not None:
      stats = self._update_stats(
        params,
        self.__data,
        self.__data_pending,
        self.__stats,
      )
    else:
      stats = self._compute_stats(params, data)

    self.__data = data
    self.__data_pending = None
    self.__stats = stats
    return stats

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
    checks.check_shape(x, (None, *self._kernel.shape), name="x")

    stats = self._statistics(self._params)
    data = self._data
    if stats is None or data is None:
      mu = self._mean(self._params.mean, x)
      s2 = self._kernel.diag(self._params.kernel, x)
      return mu, s2

    mu, s2 = self._predict(self._params, data, stats, x)

    return mu, s2

  @base.typed
  def _loglikelihood(
    self,
    params: Params,
  ) -> jt.Float[jt.Array, ""]:
    """Compute the log-likelihood of the observed data.

    Args:
      params: the parameters of the GP to evaluate the log-likelihood at.

    Returns:
      The log-likelihood of the observed data, or 0 if there is none.
    """
    stats = self._statistics(params)
    if stats is None:
      return jnp.array(0.0)

    n = stats.r.shape[0]
    misfit = jnp.inner(stats.a, stats.r)
    logdet = jnp.sum(jnp.log(jnp.diagonal(stats.L)))

    return -0.5 * misfit - 0.5 * n * jnp.log(2 * jnp.pi) - logdet

  @base.typed
  def fit(self, max_iter: int = 100, tol: float = 1e-3) -> None:
    """Fit the GP's parameters by maximizing the log-likelihood.

    Uses L-BFGS; see `grax.optimization.minimize` for how it stops.

    Args:
      max_iter: the maximum number of L-BFGS iterations to run.
      tol: stop early once the gradient norm falls below this tolerance.
    """
    result = optimization.minimize(
      lambda params: -self._loglikelihood(params),
      self._params,
      max_iter=max_iter,
      tol=tol,
    )
    self._params = result.params
