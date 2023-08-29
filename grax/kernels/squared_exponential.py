"""Implementation of a simple, squared-exponential kernel.
"""

from grax.typing import Array, ArrayLike, DTypeLike
from typing import Optional

import chex
import jax
import jax.numpy as jnp


class SquaredExponential:
  """The squared-exponential kernel.
  """

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
      raise ValueError('dim must be specified if ell is a scalar.')

    if dim is not None and ell.ndim == 1:
      raise ValueError('dim cannot be specified if ell is non-scalar.')

    if dim is not None and dim <= 0:
      raise ValueError('dim must be greater than zero.')

    self.rho = rho
    self.ell = ell
    self.dtype = dtype
    self._dim = dim

  def __repr__(self) -> str:
    kwargs = dict()
    kwargs['rho'] = str(self.rho.tolist())
    kwargs['ell'] = str(self.ell.tolist())

    if self._dim is not None:
      kwargs['dim'] = str(self._dim)

    if self.dtype is not jnp.float32:
      kwargs['dtype'] = jax.dtypes.canonicalize_dtype(self.dtype).name

    kwargs_str = ', '.join(f'{k}={v}' for k, v in kwargs.items())
    type_str = f'{self.__class__.__name__}({kwargs_str})'

    return type_str

  @property
  def dim(self) -> int:
    return self.ell.shape[0] if self._dim is None else self._dim

  def __call__(
      self,
      X1: ArrayLike,
      X2: Optional[ArrayLike] = None,
      diag: bool = False,
  ) -> Array:
    X1 = jnp.asarray(X1, dtype=self.dtype)
    chex.assert_rank(X1, 2)
    chex.assert_axis_dimension(X1, 1, self.dim)

    if diag:
      if X2 is None:
        return jnp.full(X1.shape[0], self.rho)
      else:
        raise ValueError('the diagonal kernel is invalid for two input arrays.')

    X1 = X1 / self.ell
    Z1 = jnp.sum(X1**2, axis=1, keepdims=True)

    if X2 is not None:
      X2 = jnp.asarray(X2, dtype=self.dtype)
      chex.assert_rank(X2, 2)
      chex.assert_axis_dimension(X2, 1, self.dim)
      X2 = X2 / self.ell
      Z2 = jnp.sum(X2**2, axis=1, keepdims=True)

    else:
      X2 = X1
      Z2 = Z1

    D = jnp.clip(Z1 - 2*jnp.matmul(X1, X2.T) + Z2.T, 0)
    K = self.rho * jnp.exp(-D/2)
    return K

