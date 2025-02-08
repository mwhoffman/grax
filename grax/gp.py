"""Implementation of a GP."""

from dataclasses import dataclass

import jax.numpy as jnp
import jax.scipy.linalg as jla
from flax import nnx

from grax import kernels
from grax import likelihoods
from grax import means
from grax import types
from grax.utils import checks


@dataclass
class GPData:
  """Input/output data."""
  X: types.Array
  Y: types.Array

  def __repr__(self) -> str:
    return f"<GPData length={self.X.shape[0]}, dim={self.X.shape[1]}>"


@dataclass
class GPStatistics:
  """Sufficient statistics for making GP posterior predictions."""
  L: types.Array
  r: types.Array
  a: types.Array
  w: types.Array


class GP(nnx.Module):
  """Implementation of a GP."""

  def __init__(
    self,
    kernel: kernels.Kernel,
    likelihood: likelihoods.Gaussian,
    mean: means.Mean | None = None,
    data: tuple[types.ArrayLike, types.ArrayLike] | None = None,
  ):
    """Initialize the GP with constituent models.

    Args:
      kernel: instance of `grax.kernels.Kernel` which models how correlated any
        two inputs are.
      likelihood: an instance of `grax.likelihoods.Gaussian` which models the
        probability of an observed output given its latent function value.
      mean: an instance of `grax.means.Mean` which defines the prior expected
        output value of any input; if not given the mean is assumed to be zero.
      data: initial input data of the form `(X, Y)`; this is equivalent to
        instantiating the GP and calling `add_data(X, Y)`.
    """
    self.kernel = kernel
    self.likelihood = likelihood
    self.mean = mean or means.Zero(dim=kernel.dim)

    if self.kernel.dim != self.mean.dim:
      raise checks.CheckError(
        "The kernel and mean functions must have the same input dimensions"
      )

    # Storage for the data.
    self.data: GPData | None = None

    # Cache the posterior statistics for repeated predictions.
    self._stats : GPStatistics | None = None

    if data is not None:
      self.add_data(*data)

  def add_data(self, X: types.ArrayLike, Y: types.ArrayLike):
    """Add observed data.

    This will update the model's sufficient statistics so that posterior
    predictions can be made conditioned on these observations.

    Args:
      X: observed inputs.
      Y: observed outputs.
    """
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)

    checks.check_type_and_shape(X, types.Float, (None, self.kernel.dim))
    checks.check_type_and_shape(Y, types.Float, (X.shape[0], ))

    if self.data is None:
      # TODO: do I really want to copy here?
      self.data = GPData(X.copy(), Y.copy())

    else:
      self.data.X = jnp.r_[self.data.X, X]
      self.data.Y = jnp.r_[self.data.Y, Y]

    # Get rid of previous stats so that we recompute them during the next
    # prediction.
    self._stats = None

    # TODO: we can also incrementally update the Cholesky when data is added,
    # but this will need to be recomputed from scratch when computing gradients.

  def _get_stats(self) -> GPStatistics | None:
    if self.data is None:
      return None

    # Get the kernel plus any noise.
    K = self.kernel(self.data.X)
    diag = jnp.diag_indices_from(K)
    K = K.at[diag].set(K.at[diag].get() + self.likelihood.sn2)

    # Get the residual of the observations.
    r = self.data.Y - self.mean(self.data.X)

    L = jla.cholesky(K, lower=True)
    a = jla.cho_solve((L, True), r)
    w = jnp.ones_like(a)

    return GPStatistics(L, r, a, w)

  def predict(self, X: types.ArrayLike) -> tuple[types.Array, types.Array]:
    """Make predictions of the latent function at the given input points.

    Args:
      X: input points to make predictions at.

    Returns:
      A tuple of the form `(mu, s2)` consisting of the mean/expected predictions
      as well as the variance in this prediction. Note: this is the variance of
      the prediction itself, e.g. f(x) and does NOT include any observation
      noise.
    """
    X = jnp.asarray(X)

    checks.check_type_and_shape(X, types.Float, (None, self.kernel.dim))

    # Compute the prior mean and variance.
    mu = self.mean(X)
    s2 = self.kernel(X, diag=True)

    # Return early if we have no data (i.e. these are prior predictions).
    if self.data is None:
      return mu, s2

    # Recompute the statistics if we haven't cached them. Since we've already
    # checked that self.data is not None, stats cannot be None. The assert is
    # only here to help with type checking.
    if self._stats is None:
      self._stats = self._get_stats()
      assert self._stats is not None

    # Perform the intermediate computations.
    K = self.kernel(self.data.X, X)
    w = self._stats.w.reshape(-1, 1)
    V = jla.solve_triangular(self._stats.L, w*K, lower=True)

    # Adjust the mean and shrink the variance.
    mu = mu + K.T @ self._stats.a
    s2 = s2 - jnp.sum(V**2, axis=0)

    return mu, s2

  def log_likelihood(self) -> types.Array:
    """Return the log-likelihood of the observed data."""
    if self.data is None:
      return jnp.array(0.0)

    # Get the stats. And again, data is not None, so stats won't be None and the
    # assert is only helping type checking.
    stats = self._get_stats()
    assert stats is not None

    # Get the diagonal of the cholesky.
    L_diag = stats.L.at[jnp.diag_indices_from(stats.L)].get()

    return (
      -0.5 * jnp.inner(stats.a, stats.r)
      - 0.5 * jnp.log(2 * jnp.pi) * self.data.X.shape[0]
      - jnp.sum(jnp.log(L_diag))
    )
