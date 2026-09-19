"""Kernel base class and the basic kernels that combine other kernels."""

from __future__ import annotations

import abc
import dataclasses
from typing import Generic
from typing import TypeVar
from typing import overload

import jax
import jax.numpy as jnp
import jaxtyping as jt

from grax import base
from grax import checks


Params = TypeVar("Params")
Params1 = TypeVar("Params1")
Params2 = TypeVar("Params2")


class Kernel(abc.ABC, Generic[Params]):
  """Definition of the kernel interface."""

  @abc.abstractmethod
  def init(self) -> Params:
    """Construct parameters for the kernel."""

  @property
  @abc.abstractmethod
  def shape(self) -> tuple[int, ...]:
    """The shape of inputs expected by this kernel."""

  @abc.abstractmethod
  def __call__(
    self,
    params: Params,
    x1: jt.Shaped[jt.Array, "n ..."],
    x2: jt.Shaped[jt.Array, "m ..."],
  ) -> jt.Float[jt.Array, "n m"]:
    """Evaluate the kernel on given inputs.

    Args:
      params: the parameters of the kernel.
      x1: the first set of kernel inputs.
      x2: the second set of kernel inputs (or None).

    Returns:
      An array K of shape (n, m) where n and m are the batch dimensions of `x1`
      and `x2` respectively. K[i, j] is the kernel function evaluated on
      k(x1[i], x2[j]).
    """

  @abc.abstractmethod
  def diag(
    self,
    params: Params,
    x: jt.Shaped[jt.Array, "n ..."],
  ) -> jt.Float[jt.Array, " n"]:
    """Evaluate the diagonal kernel matrix on given inputs.

    Args:
      params: the parameters of the kernel.
      x: the set of kernel inputs.

    Returns:
      The pairwise kernel evaluating each element of `x` against itself, i.e.
      k[i] is the kernel function evaluated on k(x1[i], x1[i]).
    """

  @overload
  def __mul__(
    self, other: Kernel[Params2]
  ) -> ProductKernel[Params, Params2]: ...

  @overload
  def __mul__(self, other: float) -> ProductKernel[Params, ConstantParams]: ...

  def __mul__(self, other: object) -> ProductKernel:
    """Multiply this kernel by another kernel or a positive float.

    Args:
      other: the kernel, or positive float, to multiply by.

    Returns:
      The product kernel, with this kernel as `k1` and `other` as `k2`.

    Raises:
      TypeError: if `other` is neither a kernel nor a float.
    """
    return ProductKernel(k1=self, k2=as_kernel(other, self.shape))

  def __rmul__(self, other: float) -> ProductKernel[ConstantParams, Params]:
    """Multiply a positive float by this kernel.

    Args:
      other: the positive float to multiply by.

    Returns:
      The product kernel, with a `ConstantKernel` as `k1` and this kernel as
      `k2`.
    """
    return ProductKernel(k1=as_kernel(other, self.shape), k2=self)

  @overload
  def __add__(self, other: Kernel[Params2]) -> SumKernel[Params, Params2]: ...

  @overload
  def __add__(self, other: float) -> SumKernel[Params, ConstantParams]: ...

  def __add__(self, other: object) -> SumKernel:
    """Add another kernel or a positive float to this kernel.

    Args:
      other: the kernel, or positive float, to add.

    Returns:
      The sum kernel, with this kernel as `k1` and `other` as `k2`.

    Raises:
      TypeError: if `other` is neither a kernel nor a float.
    """
    return SumKernel(k1=self, k2=as_kernel(other, self.shape))

  def __radd__(self, other: float) -> SumKernel[ConstantParams, Params]:
    """Add this kernel to a positive float.

    Args:
      other: the positive float to add to.

    Returns:
      The sum kernel, with a `ConstantKernel` as `k1` and this kernel as `k2`.
    """
    return SumKernel(k1=as_kernel(other, self.shape), k2=self)


@base.typed
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class ConstantParams:
  """Parameters of the constant kernel, stored in log-space.

  Attributes:
    logrho: the log of the constant covariance.
  """

  logrho: jt.Float[jt.Array, ""]


@base.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class ConstantKernel(Kernel[ConstantParams]):
  """The constant kernel, i.e. k(x, x') = rho for all inputs.

  `rho` is the initial value of the kernel's hyperparameter, used by `init` to
  construct its params. It is not updated by fitting: the fitted value lives
  in the params held by the GP (i.e. `logrho`).

  Attributes:
    input_shape: the shape of a single kernel input (excluding batch dims).
    rho: the initial constant covariance, which must be positive; defaults to 1
      if unset.
  """

  input_shape: tuple[int, ...]
  rho: base.ScalarLike | None = None

  def __post_init__(self) -> None:
    """Check that, if given, rho is positive."""
    checks.check_none_or_positive(self.rho, name="rho")

  @property
  def shape(self) -> tuple[int, ...]:
    """The expected shape of a single kernel input (excluding batch dims)."""
    return self.input_shape

  @base.typed
  def init(self) -> ConstantParams:
    """Construct parameters for the kernel from rho.

    Falls back to a default constant covariance of 1 if `rho` is unset.

    Returns:
      The parameters of the kernel, taken directly from `rho`.
    """
    rho = self.rho if self.rho is not None else 1.0
    return ConstantParams(logrho=jnp.log(jnp.asarray(rho, dtype=float)))

  @base.typed
  def __call__(
    self,
    params: ConstantParams,
    x1: jt.Float[jt.Array, "n ..."],
    x2: jt.Float[jt.Array, "m ..."],
  ) -> jt.Float[jt.Array, "n m"]:
    """Evaluate the kernel on given inputs.

    Args:
      params: the parameters of the kernel.
      x1: the first set of kernel inputs.
      x2: the second set of kernel inputs.

    Returns:
      An array K of shape (n, m) where n and m are the batch dimensions of
      `x1` and `x2` respectively, with every entry equal to `rho`.
    """
    checks.check_shape(x1, (None, *self.input_shape), name="x1")
    checks.check_shape(x2, (None, *self.input_shape), name="x2")

    rho = jnp.exp(params.logrho)
    return jnp.full((x1.shape[0], x2.shape[0]), rho)

  @base.typed
  def diag(
    self,
    params: ConstantParams,
    x: jt.Float[jt.Array, "n ..."],
  ) -> jt.Float[jt.Array, " n"]:
    """Evaluate the diagonal kernel matrix on given inputs.

    Args:
      params: the parameters of the kernel.
      x: the set of kernel inputs.

    Returns:
      An n-vector with every entry equal to `rho`.
    """
    checks.check_shape(x, (None, *self.input_shape), name="x")

    rho = jnp.exp(params.logrho)
    return jnp.full(x.shape[0], rho)


def as_kernel(x: object, shape: tuple[int, ...]) -> Kernel:
  """Convert a kernel or a positive float to a kernel.

  A float is replaced by a `ConstantKernel`, which introduces a trainable
  parameter initialized to the float.

  Args:
    x: the kernel or positive float to convert.
    shape: the input shape of the kernel that `x` will be combined with, used
      for the `ConstantKernel`.

  Returns:
    `x` if it is already a kernel, and otherwise a `ConstantKernel` of the
    given input shape.

  Raises:
    TypeError: if `x` is neither a kernel nor a float.
  """
  if isinstance(x, Kernel):
    return x
  if isinstance(x, float):
    return ConstantKernel(input_shape=shape, rho=x)
  msg = f"Expected a Kernel or a positive float, got {type(x).__name__}."
  raise TypeError(msg)


@base.typed
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class ProductParams(Generic[Params1, Params2]):
  """Parameters of the product kernel.

  Attributes:
    k1: the parameters of the first kernel.
    k2: the parameters of the second kernel.
  """

  k1: Params1
  k2: Params2


@base.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class ProductKernel(Kernel[ProductParams[Params1, Params2]]):
  """The elementwise product of two kernels.

  Attributes:
    k1: the first kernel.
    k2: the second kernel; must expect the same input shape as `k1`.
  """

  k1: Kernel[Params1]
  k2: Kernel[Params2]

  def __post_init__(self) -> None:
    """Check that the two kernels expect the same input shape."""
    if self.k1.shape != self.k2.shape:
      msg = f"k1.shape {self.k1.shape} != k2.shape {self.k2.shape}."
      raise checks.CheckError(msg)

  @property
  def shape(self) -> tuple[int, ...]:
    """The expected shape of a single kernel input (excluding batch dims)."""
    return self.k1.shape

  @base.typed
  def init(self) -> ProductParams[Params1, Params2]:
    """Construct parameters for the kernel from those of `k1` and `k2`.

    Returns:
      The parameters of the kernel, holding the params of each sub-kernel.
    """
    return ProductParams(k1=self.k1.init(), k2=self.k2.init())

  @base.typed
  def __call__(
    self,
    params: ProductParams[Params1, Params2],
    x1: jt.Float[jt.Array, " n *d"],
    x2: jt.Float[jt.Array, " m *d"],
  ) -> jt.Float[jt.Array, "n m"]:
    """Evaluate the kernel on given inputs.

    Args:
      params: the parameters of the kernel.
      x1: the first set of kernel inputs.
      x2: the second set of kernel inputs.

    Returns:
      An array K of shape (n, m) that is the elementwise product of the
      sub-kernels evaluated on `x1` and `x2`.
    """
    return self.k1(params.k1, x1, x2) * self.k2(params.k2, x1, x2)

  @base.typed
  def diag(
    self,
    params: ProductParams[Params1, Params2],
    x: jt.Float[jt.Array, " n *d"],
  ) -> jt.Float[jt.Array, " n"]:
    """Evaluate the diagonal kernel matrix on given inputs.

    Args:
      params: the parameters of the kernel.
      x: the set of kernel inputs.

    Returns:
      The elementwise product of the sub-kernels' diagonals.
    """
    return self.k1.diag(params.k1, x) * self.k2.diag(params.k2, x)


@base.typed
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class SumParams(Generic[Params1, Params2]):
  """Parameters of the sum kernel.

  Attributes:
    k1: the parameters of the first kernel.
    k2: the parameters of the second kernel.
  """

  k1: Params1
  k2: Params2


@base.typed
@dataclasses.dataclass(frozen=True, kw_only=True)
class SumKernel(Kernel[SumParams[Params1, Params2]]):
  """The sum of two kernels.

  Attributes:
    k1: the first kernel.
    k2: the second kernel; must expect the same input shape as `k1`.
  """

  k1: Kernel[Params1]
  k2: Kernel[Params2]

  def __post_init__(self) -> None:
    """Check that the two kernels expect the same input shape."""
    if self.k1.shape != self.k2.shape:
      msg = f"k1.shape {self.k1.shape} != k2.shape {self.k2.shape}."
      raise checks.CheckError(msg)

  @property
  def shape(self) -> tuple[int, ...]:
    """The expected shape of a single kernel input (excluding batch dims)."""
    return self.k1.shape

  @base.typed
  def init(self) -> SumParams[Params1, Params2]:
    """Construct parameters for the kernel from those of `k1` and `k2`.

    Returns:
      The parameters of the kernel, holding the params of each sub-kernel.
    """
    return SumParams(k1=self.k1.init(), k2=self.k2.init())

  @base.typed
  def __call__(
    self,
    params: SumParams[Params1, Params2],
    x1: jt.Float[jt.Array, " n *d"],
    x2: jt.Float[jt.Array, " m *d"],
  ) -> jt.Float[jt.Array, "n m"]:
    """Evaluate the kernel on given inputs.

    Args:
      params: the parameters of the kernel.
      x1: the first set of kernel inputs.
      x2: the second set of kernel inputs.

    Returns:
      An array K of shape (n, m) that is the elementwise sum of the sub-kernels
      evaluated on `x1` and `x2`.
    """
    return self.k1(params.k1, x1, x2) + self.k2(params.k2, x1, x2)

  @base.typed
  def diag(
    self,
    params: SumParams[Params1, Params2],
    x: jt.Float[jt.Array, " n *d"],
  ) -> jt.Float[jt.Array, " n"]:
    """Evaluate the diagonal kernel matrix on given inputs.

    Args:
      params: the parameters of the kernel.
      x: the set of kernel inputs.

    Returns:
      The elementwise sum of the sub-kernels' diagonals.
    """
    return self.k1.diag(params.k1, x) + self.k2.diag(params.k2, x)
