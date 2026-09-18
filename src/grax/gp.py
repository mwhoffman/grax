"""Implementation of a GP."""

import dataclasses
from typing import Generic
from typing import TypeVar

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import jaxtyping as jt
import optax

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
    self._data: GPData | None = None
    self._stats: GPStatistics | None = None
    self._params = GPParams(
      kernel=kernel.init(),
      mean=mean.init(),
      logsn2=jnp.log(jnp.asarray(sn2)),
    )

    @base.typed
    def compute_stats(
      params: GPParams[KernelParams, MeanParams],
      data: GPData,
    ) -> GPStatistics:
      sn2_ = jnp.exp(params.logsn2)
      k = kernel(params.kernel, data.x, data.x)
      k = k + sn2_ * jnp.eye(data.x.shape[0])
      r = data.y - mean(params.mean, data.x)

      chol = jla.cholesky(k, lower=True)
      a = jla.cho_solve((chol, True), r)

      return GPStatistics(L=chol, r=r, a=a)

    @base.typed
    def predict(
      params: GPParams[KernelParams, MeanParams],
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

    # jax.jit is the outermost decorator (wrapping the already-@base.typed
    # functions above) so that repeated calls with matching shapes skip
    # straight to the compiled executable -- including skipping the
    # jaxtyping/beartype checks, which then only run once per shape, at
    # trace time.
    self._compute_stats = jax.jit(compute_stats)
    self._predict = jax.jit(predict)

  @property
  def _params(self) -> GPParams[KernelParams, MeanParams]:
    """The GP's current parameters."""
    return self.__params

  @_params.setter
  def _params(self, params: GPParams[KernelParams, MeanParams]) -> None:
    self.__params = params
    self._stats = None

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
    self._stats = None

  @base.typed
  def _statistics(
    self,
    params: GPParams[KernelParams, MeanParams],
  ) -> GPStatistics | None:
    """Compute the sufficient statistics needed for posterior prediction.

    Cached when `params` is the GP's current parameters (`self._params`);
    that cache is invalidated automatically whenever `self._params` is
    reassigned or `add_data` is called. For any other `params` -- e.g. a
    candidate value during `fit`'s search -- this always recomputes fresh
    and never reads or writes the cache.

    Args:
      params: the parameters of the GP to compute statistics for.
    """
    if self._data is None:
      return None

    is_current = params is self._params
    if is_current and self._stats is not None:
      return self._stats

    stats = self._compute_stats(params, self._data)

    if is_current:
      self._stats = stats
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
    checks.check_shape(x, (None, *self._kernel.shape))

    stats = self._statistics(self._params)
    if stats is None or self._data is None:
      mu = self._mean(self._params.mean, x)
      s2 = self._kernel.diag(self._params.kernel, x)
      return mu, s2

    mu, s2 = self._predict(self._params, self._data, stats, x)

    return mu, s2

  @base.typed
  def _loglikelihood(
    self,
    params: GPParams[KernelParams, MeanParams],
  ) -> jt.Float[jt.Array, ""]:
    """Compute the log-likelihood of the observed data.

    Args:
      params: the parameters of the GP to evaluate the log-likelihood at.

    Returns:
      The log-likelihood of the observed data, or 0 if there is none.
    """
    stats = self._statistics(params)
    if stats is None or self._data is None:
      return jnp.array(0.0)

    n = self._data.x.shape[0]
    misfit = jnp.inner(stats.a, stats.r)
    logdet = jnp.sum(jnp.log(jnp.diagonal(stats.L)))

    return -0.5 * misfit - 0.5 * n * jnp.log(2 * jnp.pi) - logdet

  @base.typed
  def fit(self, max_iter: int = 100, tol: float = 1e-3) -> None:
    """Fit the GP's parameters by maximizing the log-likelihood.

    Uses L-BFGS, roughly following the `run_opt` pattern from:
    https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html.

    Args:
      max_iter: the maximum number of L-BFGS iterations to run.
      tol: stop early once the gradient norm falls below this tolerance.
    """
    # This optimizes over a flattened list of `self._params`'s rather `GPParams`
    # directly. This is due to the fact that L-BFGS uses the parameter structure
    # with an extra per-leaf "history" dimension for its internal state and as a
    # result any runtime shape checks will fail.
    flat_params, treedef = jax.tree_util.tree_flatten(self._params)

    def objective(flat_params: list[jax.Array]) -> jt.Float[jt.Array, ""]:
      params = jax.tree_util.tree_unflatten(treedef, flat_params)
      return -self._loglikelihood(params)

    optimizer = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(objective)

    def step(
      carry: tuple[list[jax.Array], optax.OptState],
    ) -> tuple[list[jax.Array], optax.OptState]:
      flat_params, state = carry
      value, grad = value_and_grad_fun(flat_params, state=state)
      updates, state = optimizer.update(
        grad,
        state,
        flat_params,
        value=value,
        grad=grad,
        value_fn=objective,
      )
      flat_params = optax.apply_updates(flat_params, updates)
      # optax's stubs return the same broad Union type regardless of the
      # concrete input type, so this doesn't statically narrow back to
      # `list[Array]` even though it is one at runtime.
      return flat_params, state  # ty: ignore[invalid-return-type]

    def continuing_criterion(
      carry: tuple[list[jax.Array], optax.OptState],
    ) -> jt.Bool[jt.Array, ""]:
      _, state = carry
      iter_num = optax.tree.get(state, "count")
      grad = optax.tree.get(state, "grad")
      err = optax.tree.norm(grad)
      return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (flat_params, optimizer.init(flat_params))
    final_flat_params, _ = jax.lax.while_loop(
      continuing_criterion,
      step,
      init_carry,
    )
    self._params = jax.tree_util.tree_unflatten(treedef, final_flat_params)
