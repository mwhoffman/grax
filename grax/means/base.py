"""Mean function base class."""

from abc import abstractmethod

from flax import nnx

from grax import types


class Mean(nnx.Module):
  """Definition of the mean function interface."""

  @abstractmethod
  def __call__(
    self,
    x: types.ArrayLike,
  ) -> types.Array:
    """Evaluate the mean on the given inputs.

    Args:
      x: an n-array containing the input points.

    Returns:
      an n-vector of the function evaluated at the given points.
    """

  @property
  @abstractmethod
  def dim(self) -> int:
    """The input dimension."""

    # Implicitly this assumes mean functions are defined over a Euclidean space.
    # This is added just so we can check and make sure that the inputs for
    # kernels/means maches up, but we can make this explicit later.
