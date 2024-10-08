"""Implementation of a simple, squared-exponential kernel."""

import jax.numpy as jnp

from grax import typing
from grax.kernels import base
from grax.utils import checks
from grax.utils import repr

type Params = tuple[typing.Array, typing.Array]


class SquaredExponential(base.Kernel):
  """The squared-exponential kernel."""

  def __init__(
    self,
    rho: typing.ArrayLike,
    ell: typing.ArrayLike,
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

    checks.check_type_and_rank(rho, typing.Float, 0)
    checks.check_type_and_rank(ell, typing.Float, {0, 1})

    if dim is not None:
      if ell.ndim == 1:
        raise checks.CheckError("dim cannot be specified if ell is non-scalar.")

      checks.check_positive(dim)

    elif ell.ndim == 0:
      raise checks.CheckError("dim must be specified if ell is a scalar.")

    self.rho = rho
    self.ell = ell
    self._dim = dim

  def __repr__(self) -> str:
    kwargs = dict()
    kwargs["rho"] = str(self.rho.tolist())
    kwargs["ell"] = str(self.ell.tolist())

    if self._dim is not None:
      kwargs["dim"] = str(self._dim)

    return f"{self.__class__.__name__}({repr.join_dict(kwargs)})"

  def get_params(self) -> Params:
    """Return the parameters of the model."""
    return (self.rho, self.ell)

  @property
  def dim(self) -> int:
    """The dimension of the kernel inputs."""
    return self.ell.shape[0] if self._dim is None else self._dim

  def __call__(
    self,
    x1: typing.ArrayLike,
    x2: typing.ArrayLike | None = None,
    *,
    diag: bool = False,
  ) -> typing.Array:
    """Evaluate the kernel on the given inputs."""
    x1 = jnp.asarray(x1)
    checks.check_type_and_shape(x1, typing.Float, (None, self.dim))

    if diag:
      if x2 is not None:
        msg = "the diagonal kernel is invalid for two input arrays."
        raise checks.CheckError(msg)

      return jnp.full(x1.shape[0], self.rho)

    x1 = x1 / self.ell
    z1 = jnp.sum(x1**2, axis=1, keepdims=True)

    if x2 is not None:
      x2 = jnp.asarray(x2)
      checks.check_type_and_shape(x1, typing.Float, (None, self.dim))
      x2 = x2 / self.ell
      z2 = jnp.sum(x2**2, axis=1, keepdims=True)

    else:
      x2 = x1
      z2 = z1

    D = jnp.clip(z1 - 2 * jnp.matmul(x1, x2.T) + z2.T, 0)
    K = self.rho * jnp.exp(-D / 2)
    return K
