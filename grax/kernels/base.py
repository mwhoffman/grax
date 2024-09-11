"""Kernel base class."""

from abc import ABCMeta, abstractmethod, abstractproperty
from grax import typing


class Kernel(metaclass=ABCMeta):
  """Definition of the kernel interface."""

  @abstractmethod
  def __call__(
    self,
    x1: typing.ArrayLike,
    x2: typing.ArrayLike | None = None,
    diag: bool = False,
  ) -> typing.Array:
    """Evaluate the kernel."""
