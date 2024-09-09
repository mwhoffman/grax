"""Implementation of a simple, squared-exponential kernel.
"""

from grax import typing
from grax import checks

import jax.numpy as jnp


class SquaredExponential:
  """The squared-exponential kernel."""

  def __init__(
    self,
    rho: typing.ArrayLike,
    ell: typing.ArrayLike,
    dim: int | None = None,
  ):
    rho = jnp.asarray(rho)
    ell = jnp.asarray(ell)

    checks.check_type_and_rank(rho, typing.Float, 0)
    checks.check_type_and_rank(ell, typing.Float, {0, 1})

    if dim is not None:
      if ell.ndim == 1:
        raise ValueError("dim cannot be specified if ell is non-scalar.")

      elif dim <= 0:
        raise ValueError("dim must be greater than zero.")

    elif ell.ndim == 0:
      raise ValueError("dim must be specified if ell is a scalar.")

    self.rho = rho
    self.ell = ell
    self._dim = dim

  def __repr__(self) -> str:
    kwargs = dict()
    kwargs["rho"] = str(self.rho.tolist())
    kwargs["ell"] = str(self.ell.tolist())

    if self._dim is not None:
      kwargs["dim"] = str(self._dim)

    kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    type_str = f"{self.__class__.__name__}({kwargs_str})"

    return type_str

  @property
  def dim(self) -> int:
    return self.ell.shape[0] if self._dim is None else self._dim

  def __call__(
    self,
    x1: typing.ArrayLike,
    x2: typing.ArrayLike | None = None,
    diag: bool = False,
  ) -> typing.Array:
    x1 = jnp.asarray(x1)
    checks.check_type_and_dimensions(x1, typing.Float, (None, self.dim))

    if diag:
      if x2 is None:
        return jnp.full(x1.shape[0], self.rho)
      else:
        raise ValueError("the diagonal kernel is invalid for two input arrays.")

    x1 = x1 / self.ell
    z1 = jnp.sum(x1**2, axis=1, keepdims=True)

    if x2 is not None:
      x2 = jnp.asarray(x2)
      checks.check_type_and_dimensions(x1, typing.Float, (None, self.dim))
      x2 = x2 / self.ell
      z2 = jnp.sum(x2**2, axis=1, keepdims=True)

    else:
      x2 = x1
      z2 = z1

    D = jnp.clip(z1 - 2 * jnp.matmul(x1, x2.T) + z2.T, 0)
    K = self.rho * jnp.exp(-D / 2)
    return K

