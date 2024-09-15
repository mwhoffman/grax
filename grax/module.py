"""Module base class."""

from abc import ABCMeta
from abc import abstractmethod
from typing import Generic
from typing import TypeVar

Params = TypeVar("Params")


class Module(Generic[Params], metaclass=ABCMeta):
  """Module base class."""

  @abstractmethod
  def get_params(self) -> Params:
    """Return module parameters."""
