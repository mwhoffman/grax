"""Kernel base class."""

from abc import ABCMeta, abstractmethod
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

  @property
  @abstractmethod
  def dim(self) -> int:
    """Return the input dimension."""

    # Implicitly this assumes our kernels are defined over a Euclidean space.
    # This is added just so we can check and make sure that the inputs for
    # kernels/means maches up, but we can make this explicit later.
