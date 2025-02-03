"""Likelihood base class."""

from abc import abstractmethod

from flax import nnx

from grax import typing


class Likelihood(nnx.Module):
  """Definition of the interface for marginal log-likelihoods."""

  @abstractmethod
  def __call__(self, y: typing.ArrayLike, f: typing.ArrayLike) -> typing.Array:
    """Evaluate the marginal log-likelihood.

    Args:
      y: an n-array of observations.
      f: an n-array of latent function values.

    Returns:
      a vector of the log-likelihood of observing `y`, conditioned on the given
      value of the latent function `f`.
    """
