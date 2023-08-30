"""Implementation of a simple, squared-exponential kernel.
"""

from grax.typing import Array, ArrayLike, DTypeLike
from typing import Optional

import chex
import jax
import jax.numpy as jnp


class SquaredExponential:
  """The squared-exponential kernel."""

  def __init__(
    self,
    rho: ArrayLike,
    ell: ArrayLike,
    dim: Optional[int] = None,
    dtype: DTypeLike = jnp.float32,
  ):
    rho = jnp.asarray(rho, dtype=dtype)
    ell = jnp.asarray(ell, dtype=dtype)

    chex.assert_rank(rho, 0)
    chex.assert_rank(ell, {0, 1})

    if dim is None and ell.ndim == 0:
      raise ValueError("dim must be specified if ell is a scalar.")

    if dim is not None and ell.ndim == 1:
      raise ValueError("dim cannot be specified if ell is non-scalar.")

    if dim is not None and dim <= 0:
      raise ValueError("dim must be greater than zero.")

    self.rho = rho
    self.ell = ell
    self.dtype = dtype
    self._dim = dim

  def __repr__(self) -> str:
    kwargs = dict()
    kwargs["rho"] = str(self.rho.tolist())
    kwargs["ell"] = str(self.ell.tolist())

    if self._dim is not None:
      kwargs["dim"] = str(self._dim)

    if self.dtype is not jnp.float32:
      kwargs["dtype"] = jax.dtypes.canonicalize_dtype(self.dtype).name

    kwargs_str = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    type_str = f"{self.__class__.__name__}({kwargs_str})"

    return type_str

  @property
  def dim(self) -> int:
    return self.ell.shape[0] if self._dim is None else self._dim

  def __call__(
    self,
    x1: ArrayLike,
    x2: Optional[ArrayLike] = None,
    diag: bool = False,
  ) -> Array:
    x1 = jnp.asarray(x1, dtype=self.dtype)
    chex.assert_rank(x1, 2)
    chex.assert_axis_dimension(x1, 1, self.dim)

    if diag:
      if x2 is None:
        return jnp.full(x1.shape[0], self.rho)
      else:
        raise ValueError("the diagonal kernel is invalid for two input arrays.")

    x1 = x1 / self.ell
    z1 = jnp.sum(x1**2, axis=1, keepdims=True)

    if x2 is not None:
      x2 = jnp.asarray(x2, dtype=self.dtype)
      chex.assert_rank(x2, 2)
      chex.assert_axis_dimension(x2, 1, self.dim)
      x2 = x2 / self.ell
      z2 = jnp.sum(x2**2, axis=1, keepdims=True)

    else:
      x2 = x1
      z2 = z1

    D = jnp.clip(z1 - 2 * jnp.matmul(x1, x2.T) + z2.T, 0)
    K = self.rho * jnp.exp(-D / 2)
    return K
