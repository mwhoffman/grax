"""Mean function base class."""

from typing import Protocol
from typing import TypeVar

import jaxtyping as jt


Params = TypeVar("Params")


class Mean(Protocol[Params]):
  """Definition of the mean function interface."""

  def init(self) -> Params:
    """Construct parameters for the mean function."""

  @property
  def shape(self) -> tuple[int, ...]:
    """The shape of inputs expected by this mean function."""

  def __call__(
    self,
    params: Params,
    x: jt.Shaped[jt.Array, "n ..."],
  ) -> jt.Float[jt.Array, " n"]:
    """Evaluate the mean function on given inputs.

    Args:
      params: the parameters of the mean function.
      x: the set of input points.

    Returns:
      An n-vector giving the mean function evaluated at each input point.
    """
