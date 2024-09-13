"""Kernel base class."""

from abc import ABCMeta
from abc import abstractmethod

from grax import typing


class Kernel(metaclass=ABCMeta):
  """Definition of the kernel interface."""

  @abstractmethod
  def __call__(
    self,
    x1: typing.ArrayLike,
    x2: typing.ArrayLike | None = None,
    *,
    diag: bool = False,
  ) -> typing.Array:
    """Evaluate the kernel on given inputs.

    Args:
      x1: the first set of kernel inputs.
      x2: the second set of kernel inputs (or None).
      diag: whether or not to return just the diagonal of the kernel matrix;
        this is only valid if `x2` is `None`.

    Returns:
      An array K of shape (n, m) where n and m are the batch dimensions of `x1`
      and `x2` respectively. K[i, j] is the kernel function evaluated on
      k(x1[i], x2[j]).

      If `x2` is `None`, this returns the pairwise kernel of `x1` against
      itself, i.e. it is equivalent to letting `x2=x1`. If `diag` is `True`,
      then this returns the diagonal of the resulting self-kernel.
    """

  @property
  @abstractmethod
  def dim(self) -> int:
    """Return the input dimension of the kernel."""

    # Implicitly this assumes our kernels are defined over a Euclidean space.
    # This is added just so we can check and make sure that the inputs for
    # kernels/means maches up, but we can make this explicit later.
