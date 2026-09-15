"""Kernel base class."""

from typing import Protocol
from typing import TypeVar

import jaxtyping as jt


Params = TypeVar("Params")


class Kernel(Protocol[Params]):
  """Definition of the kernel interface."""

  def init(self) -> Params:
    """Construct parameters for the kernel."""

  @property
  def shape(self) -> tuple[int, ...]:
    """The shape of inputs expected by this kernel."""

  def __call__(
    self,
    params: Params,
    x1: jt.Shaped[jt.Array, "n ..."],
    x2: jt.Shaped[jt.Array, "m ..."],
  ) -> jt.Float[jt.Array, "n m"]:
    """Evaluate the kernel on given inputs.

    Args:
      params: the parameters of the kernel.
      x1: the first set of kernel inputs.
      x2: the second set of kernel inputs (or None).

    Returns:
      An array K of shape (n, m) where n and m are the batch dimensions of `x1`
      and `x2` respectively. K[i, j] is the kernel function evaluated on
      k(x1[i], x2[j]).
    """

  def diag(
    self,
    params: Params,
    x: jt.Shaped[jt.Array, "n ..."],
  ) -> jt.Float[jt.Array, " n"]:
    """Evaluate the diagonal kernel matrix on given inputs.

    Args:
      params: the parameters of the kernel.
      x: the set of kernel inputs.

    Returns:
      The pairwise kernel evaluating each element of `x` against itself, i.e.
      k[i] is the kernel function evaluated on k(x1[i], x1[i]).
    """
