"""Implementation of a GP."""

from dataclasses import dataclass

import jax.numpy as jnp
import jax.scipy.linalg as jla

from grax import kernels
from grax import likelihoods
from grax import means
from grax import typing
from grax.utils import checks


@dataclass
class GPData:
  """Input/output data."""
  X: typing.Array
  Y: typing.Array


@dataclass
class GPStatistics:
  """Sufficient statistics for making GP posterior predictions."""
  L: typing.Array
  a: typing.Array
  w: typing.Array


class GP:
  """Implementation of a GP."""

  def __init__(
    self,
    kernel: kernels.Kernel,
    likelihood: likelihoods.Gaussian,
    mean: means.Mean | None = None,
    data: tuple[typing.ArrayLike, typing.ArrayLike] | None = None,
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

    # Storage for the data and posterior sufficent statistics.
    self.data: GPData | None = None
    self.post: GPStatistics | None = None

    if data is not None:
      self.add_data(*data)

  def add_data(self, X: typing.ArrayLike, Y: typing.ArrayLike):
    """Add observed data.

    This will update the model's sufficient statistics so that posterior
    predictions can be made conditioned on these observations.

    Args:
      X: observed inputs.
      Y: observed outputs.
    """
    X = jnp.asarray(X)
    Y = jnp.asarray(Y)

    checks.check_type_and_shape(X, typing.Float, (None, self.kernel.dim))
    checks.check_type_and_shape(Y, typing.Float, (X.shape[0], ))

    if self.data is None:
      # TODO: do I really want to copy here?
      self.data = GPData(X.copy(), Y.copy())

    else:
      self.data.X = jnp.r_[self.data.X, X]
      self.data.Y = jnp.r_[self.data.Y, Y]

    # Update the sufficient statistics.
    self._update()

  def _update(self):
    if self.data is not None:
      # Get the kernel plus any noise.
      K = self.kernel(self.data.X)
      diag = jnp.diag_indices_from(K)
      K = K.at[diag].set(K.at[diag].get() + self.likelihood.sn2)

      # Get the residual of the observations.
      r = self.data.Y - self.mean(self.data.X)

      L = jla.cholesky(K, lower=True)
      a = jla.cho_solve((L, True), r)
      w = jnp.ones_like(a)

      self.post = GPStatistics(L, a, w)

  def predict(self, X: typing.ArrayLike) -> tuple[typing.Array, typing.Array]:
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

    checks.check_type_and_shape(X, typing.Float, (None, self.kernel.dim))

    # Compute the prior mean and variance.
    mu = self.mean(X)
    s2 = self.kernel(X, diag=True)

    # if we have data compute the posterior
    if self.post is not None:
      # If post is not None we know the data will be as well, this is just here
      # to help with typing.
      assert self.data is not None

      # Perform the intermediate computations.
      K = self.kernel(self.data.X, X)
      w = self.post.w.reshape(-1, 1)
      V = jla.solve_triangular(self.post.L, w*K, lower=True)

      # Adjust the mean and shrink the variance.
      mu = mu + K.T @ self.post.a
      s2 = s2 - jnp.sum(V**2, axis=0)

    return mu, s2
