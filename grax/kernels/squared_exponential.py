"""Implementation of a simple, squared-exponential kernel."""

import jax.numpy as jnp
from flax import nnx

from grax import types
from grax.kernels import base
from grax.utils import checks


class SquaredExponential(base.Kernel):
  """The squared-exponential kernel."""

  def __init__(
    self,
    rho: types.ArrayLike,
    ell: types.ArrayLike,
    dim: int | None = None,
  ):
    """Initialize a squared-exponential kernel.

    Args:
      rho: the output variance.
      ell: the lengthscale of the kernel which determines how far away observing
        one point influences another point (in the input space). ell can either
        be a vector in which case each dimension has its own length scale, or a
        scalar in which case all dimensions share the same lengthscale.
      dim: an integer specifying the dimensionality of the kernel; this must be
        specified if ell is a scalar.
    """
    rho = jnp.asarray(rho)
    ell = jnp.asarray(ell)

    checks.check_type_and_rank(rho, types.Float, 0)
    checks.check_type_and_rank(ell, types.Float, {0, 1})

    if dim is not None:
      if ell.ndim == 1:
        raise checks.CheckError("dim cannot be specified if ell is non-scalar.")

      checks.check_positive(dim)

    elif ell.ndim == 0:
      raise checks.CheckError("dim must be specified if ell is a scalar.")

    self.logrho = nnx.Param(jnp.log(rho))
    self.logell = nnx.Param(jnp.log(ell))
    self._dim = dim

  @property
  def dim(self) -> int:
    """The dimension of the kernel inputs."""
    return self.logell.shape[0] if self._dim is None else self._dim

  def __call__(
    self,
    x1: types.ArrayLike,
    x2: types.ArrayLike | None = None,
    *,
    diag: bool = False,
  ) -> types.Array:
    """Evaluate the kernel on the given inputs."""
    x1 = jnp.asarray(x1)
    checks.check_type_and_shape(x1, types.Float, (None, self.dim))

    # TODO: deal with numerical issues when the log parameters -> -inf.
    ell = jnp.exp(self.logell.value)
    rho = jnp.exp(self.logrho.value)

    if diag:
      if x2 is not None:
        msg = "the diagonal kernel is invalid for two input arrays."
        raise checks.CheckError(msg)
      return jnp.full(x1.shape[0], rho)

    x1 = x1 / ell
    z1 = jnp.sum(x1**2, axis=1, keepdims=True)

    if x2 is not None:
      x2 = jnp.asarray(x2)
      checks.check_type_and_shape(x1, types.Float, (None, self.dim))
      x2 = x2 / ell
      z2 = jnp.sum(x2**2, axis=1, keepdims=True)

    else:
      x2 = x1
      z2 = z1

    D = jnp.clip(z1 - 2 * jnp.matmul(x1, x2.T) + z2.T, 0)
    K = rho * jnp.exp(-D / 2)
    return K
