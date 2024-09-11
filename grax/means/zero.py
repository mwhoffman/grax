"""Implementation of the zero-mean function."""

import jax.numpy as jnp

from grax import typing
from grax.means import base
from grax.utils import checks
from grax.utils import repr


class Zero(base.Mean):
  """The zero-mean function."""

  def __init__(self, dim: int):
    checks.check_positive(dim)

    self._dim = dim

  @property
  def dim(self):
    return self._dim

  def __repr__(self) -> str:
    kwargs = dict()
    kwargs["dim"] = str(self.dim)
    return f"{self.__class__.__name__}({repr.join_dict(kwargs)})"

  def __call__(self, x: typing.ArrayLike) -> typing.Array:
    x = jnp.asarray(x)
    checks.check_type_and_shape(x, typing.Float, (None, self.dim))
    return jnp.zeros(x.shape[0])
