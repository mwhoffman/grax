"""Likelihood base class."""

from abc import abstractmethod

from flax import nnx

from grax import types


class Likelihood(nnx.Module):
  """Definition of the interface for marginal log-likelihoods."""

  @abstractmethod
  def __call__(self, y: types.ArrayLike, f: types.ArrayLike) -> types.Array:
    """Evaluate the marginal log-likelihood.

    Args:
      y: an n-array of observations.
      f: an n-array of latent function values.

    Returns:
      a vector of the log-likelihood of observing `y`, conditioned on the given
      value of the latent function `f`.
    """
