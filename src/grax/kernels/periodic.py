"""Implementation of the periodic kernel."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import jaxtyping as jt

from grax import base
from grax import checks
from grax.kernels import base as kernels_base


@base.typed
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class PeriodicParams:
  """Parameters of the periodic kernel, stored in log-space.

  Attributes:
    logell: the log of the lengthscale.
    logperiod: the log of the period.
  """

  logell: jt.Float[jt.Array, ""]
  logperiod: jt.Float[jt.Array, ""]


@base.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class PeriodicKernel(kernels_base.Kernel[PeriodicParams]):
  """The periodic kernel on one-dimensional inputs.

  The kernel is `exp(-2 (sin(pi D / period) / ell)^2)` where `D` is the distance
  between the inputs. It has unit output variance; multiply it by a
  `ConstantKernel` (or a positive float) to scale it.

  This is only defined for one-dimensional inputs. A kernel that is periodic in
  the euclidean distance is not positive semi-definite for higher dimensions.

  `ell` and `period` are the initial values of the kernel's hyperparameters,
  used by `init` to construct its params. They are not updated by fitting: the
  fitted values live in the params held by the GP (i.e. `logell` and
  `logperiod`). Note that fitting the period is prone to local optima, so a
  sensible initial value matters.

  Attributes:
    ell: the initial lengthscale, which must be positive; defaults to 1 if
      unset.
    period: the initial period, which must be positive; defaults to 1 if unset.
  """

  ell: base.ScalarLike | None = None
  period: base.ScalarLike | None = None

  def __post_init__(self) -> None:
    """Check that, if given, ell and period are positive."""
    checks.check_none_or_positive(self.ell, name="ell")
    checks.check_none_or_positive(self.period, name="period")

  @property
  def shape(self) -> tuple[int, ...]:
    """The expected shape of a single kernel input (excluding batch dims)."""
    return (1,)

  @base.typed
  def init(self) -> PeriodicParams:
    """Construct parameters for the kernel from ell and period.

    Falls back to a default lengthscale and period of 1 for whichever of
    `ell`/`period` is unset.

    Returns:
      The parameters of the kernel, taken directly from `ell` and `period`.
    """
    ell = self.ell if self.ell is not None else 1.0
    period = self.period if self.period is not None else 1.0
    return PeriodicParams(
      logell=jnp.log(jnp.asarray(ell, dtype=float)),
      logperiod=jnp.log(jnp.asarray(period, dtype=float)),
    )

  @base.typed
  def __call__(
    self,
    params: PeriodicParams,
    x1: jt.Float[jt.Array, "n d"],
    x2: jt.Float[jt.Array, "m d"],
  ) -> jt.Float[jt.Array, "n m"]:
    """Evaluate the kernel on given inputs.

    Args:
      params: the parameters of the kernel.
      x1: the first set of kernel inputs.
      x2: the second set of kernel inputs.

    Returns:
      An array K of shape (n, m) where n and m are the batch dimensions of
      `x1` and `x2` respectively. K[i, j] is the kernel function evaluated
      on k(x1[i], x2[j]).
    """
    checks.check_shape(x1, (None, 1), name="x1")
    checks.check_shape(x2, (None, 1), name="x2")

    ell = jnp.exp(params.logell)
    period = jnp.exp(params.logperiod)
    diff = x1 - x2.T
    return jnp.exp(-2 * (jnp.sin(jnp.pi * diff / period) / ell) ** 2)

  @base.typed
  def diag(
    self,
    params: PeriodicParams,
    x: jt.Float[jt.Array, "n d"],
  ) -> jt.Float[jt.Array, " n"]:
    """Evaluate the diagonal kernel matrix on given inputs.

    Args:
      params: the parameters of the kernel (unused, since the diagonal is
        always one).
      x: the set of kernel inputs.

    Returns:
      The pairwise kernel evaluating each element of `x` against itself,
      i.e. k[i] is the kernel function evaluated on k(x[i], x[i]).
    """
    del params
    checks.check_shape(x, (None, 1), name="x")

    return jnp.ones(x.shape[0])
