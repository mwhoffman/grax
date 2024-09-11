"""Likelihood base class."""

from abc import ABCMeta, abstractmethod
from grax import typing


class Likelihood(metaclass=ABCMeta):
  """Definition of the interface for marginal log-likelihoods."""

  @abstractmethod
  def __call__(self, y: typing.ArrayLike, f: typing.ArrayLike) -> typing.Array:
    """Evaluate the log-likelihood."""
