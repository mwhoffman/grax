"""Implementation of a GP."""

import jax.numpy as jnp
import jax.scipy.linalg as jla

from dataclasses import dataclass

from grax import kernels
from grax import likelihoods
from grax import means
from grax import typing
from grax.utils import checks


@dataclass
class GPData:
  X: typing.Array
  Y: typing.Array


@dataclass
class GPPosterior:
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
    self.kernel = kernel
    self.likelihood = likelihood
    self.mean = mean or means.Zero(dim=kernel.dim)

    # Storage for the data and posterior sufficent statistics.
    self.data: GPData | None = None
    self.post: GPPosterior | None = None

    if data is not None:
      self.add_data(*data)

  def add_data(self, X: typing.ArrayLike, Y: typing.ArrayLike):
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

      self.post = GPPosterior(L, a, w)

  def predict(self, X: typing.ArrayLike) -> tuple[typing.Array, typing.Array]:
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
