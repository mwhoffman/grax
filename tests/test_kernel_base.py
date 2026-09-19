"""Tests for the kernels defined in grax.kernels.base."""

import jax
import jax.numpy as jnp
import jaxtyping as jt
import pytest

from grax import checks
from grax import kernels
from grax.kernels import base


@pytest.fixture
def constant() -> base.ConstantKernel:
  return base.ConstantKernel(input_shape=(3,), rho=2.0)


@pytest.fixture
def product() -> base.ProductKernel:
  return base.ProductKernel(
    k1=base.ConstantKernel(input_shape=(3,), rho=2.0),
    k2=kernels.SEKernel(dim=3, ell=jnp.array([0.5, 1.0, 2.0])),
  )


@pytest.fixture
def sum_kernel() -> base.SumKernel:
  return base.SumKernel(
    k1=base.ConstantKernel(input_shape=(3,), rho=2.0),
    k2=kernels.SEKernel(dim=3, ell=jnp.array([0.5, 1.0, 2.0])),
  )


def test_constant_shape(constant: base.ConstantKernel):
  assert constant.shape == (3,)


def test_constant_init(constant: base.ConstantKernel):
  assert jnp.allclose(constant.init().logrho, jnp.log(2.0))


def test_constant_default_rho():
  params = base.ConstantKernel(input_shape=(3,)).init()
  assert jnp.allclose(params.logrho, jnp.log(1.0))


def test_constant_rejects_non_scalar_rho():
  with pytest.raises(jt.TypeCheckError):
    base.ConstantKernel(input_shape=(3,), rho=jnp.array([1.0, 2.0]))


def test_constant_rejects_int_rho():
  with pytest.raises(jt.TypeCheckError):
    base.ConstantKernel(input_shape=(3,), rho=2)


@pytest.mark.parametrize("rho", [0.0, -1.0, float("nan")])
def test_constant_rejects_non_positive_rho(rho: float):
  with pytest.raises(ValueError, match="rho must be positive"):
    base.ConstantKernel(input_shape=(3,), rho=rho)


def test_constant_call_returns_constant_matrix(constant: base.ConstantKernel):
  params = constant.init()
  x1 = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
  x2 = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
  assert jnp.allclose(constant(params, x1, x2), jnp.full((2, 4), 2.0))


def test_constant_diag_returns_constant_vector(constant: base.ConstantKernel):
  params = constant.init()
  assert jnp.allclose(constant.diag(params, jnp.zeros((5, 3))), 2.0)


def test_constant_supports_multi_dimensional_inputs():
  kernel = base.ConstantKernel(input_shape=(2, 3), rho=4.0)
  x = jnp.zeros((5, 2, 3))
  assert kernel.shape == (2, 3)
  assert jnp.allclose(kernel(kernel.init(), x, x), jnp.full((5, 5), 4.0))
  assert jnp.allclose(kernel.diag(kernel.init(), x), jnp.full(5, 4.0))


def test_constant_call_is_differentiable_in_logrho(
  constant: base.ConstantKernel,
):
  x = jnp.zeros((2, 3))
  grad = jax.grad(lambda p: jnp.sum(constant(p, x, x)))(constant.init())
  # d/dlogrho of 4 * exp(logrho) = 4 * rho.
  assert jnp.allclose(grad.logrho, 4 * 2.0)


def test_constant_call_rejects_wrong_input_shape(
  constant: base.ConstantKernel,
):
  params = constant.init()
  with pytest.raises(checks.CheckError, match=r"expected \(None, 3\)"):
    constant(params, jnp.zeros((2, 2)), jnp.zeros((2, 2)))
  with pytest.raises(checks.CheckError, match=r"expected \(None, 3\)"):
    constant.diag(params, jnp.zeros((2, 2)))


def test_product_shape(product: base.ProductKernel):
  assert product.shape == (3,)


def test_product_init_holds_sub_kernel_params(product: base.ProductKernel):
  params = product.init()
  assert jnp.allclose(params.k1.logrho, jnp.log(2.0))
  assert jnp.allclose(params.k2.logell, jnp.log(jnp.array([0.5, 1.0, 2.0])))


def test_product_rejects_mismatched_shapes():
  with pytest.raises(checks.CheckError, match=r"k1\.shape"):
    base.ProductKernel(
      k1=kernels.SEKernel(dim=2),
      k2=kernels.SEKernel(dim=3),
    )


def test_product_rejects_non_kernel():
  with pytest.raises(jt.TypeCheckError):
    base.ProductKernel(k1=kernels.SEKernel(dim=3), k2="not a kernel")  # ty: ignore[invalid-argument-type]


def test_product_call_is_elementwise_product(product: base.ProductKernel):
  params = product.init()
  x1 = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 5
  x2 = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 5
  expected = product.k1(params.k1, x1, x2) * product.k2(params.k2, x1, x2)
  assert jnp.allclose(product(params, x1, x2), expected)


def test_product_diag_matches_call_diagonal(product: base.ProductKernel):
  params = product.init()
  x = jnp.arange(9, dtype=jnp.float32).reshape(3, 3) / 5
  assert jnp.allclose(
    product.diag(params, x),
    jnp.diag(product(params, x, x)),
    atol=1e-5,
  )


def test_product_rejects_mismatched_trailing_input_dims(
  product: base.ProductKernel,
):
  with pytest.raises(jt.TypeCheckError):
    product(product.init(), jnp.zeros((2, 3)), jnp.zeros((2, 4)))


def test_mul_of_kernels_builds_product_preserving_order():
  k1 = kernels.SEKernel(dim=3)
  k2 = base.ConstantKernel(input_shape=(3,), rho=2.0)
  result = k1 * k2
  assert isinstance(result, base.ProductKernel)
  assert result.k1 is k1
  assert result.k2 is k2


def test_mul_rejects_mismatched_shapes():
  with pytest.raises(checks.CheckError, match=r"k1\.shape"):
    _ = kernels.SEKernel(dim=2) * kernels.SEKernel(dim=3)


def test_mul_by_float_on_right_adds_trainable_constant_kernel():
  k = kernels.SEKernel(dim=3)
  result = k * 2.0
  assert isinstance(result, base.ProductKernel)
  assert result.k1 is k
  assert isinstance(result.k2, base.ConstantKernel)
  assert result.k2.shape == (3,)
  assert jnp.allclose(result.init().k2.logrho, jnp.log(2.0))


def test_mul_by_float_on_left_adds_trainable_constant_kernel():
  k = kernels.SEKernel(dim=3)
  result = 2.0 * k
  assert isinstance(result, base.ProductKernel)
  assert isinstance(result.k1, base.ConstantKernel)
  assert result.k1.shape == (3,)
  assert result.k2 is k
  assert jnp.allclose(result.init().k1.logrho, jnp.log(2.0))


@pytest.mark.parametrize("scalar", [0.0, -1.0])
def test_mul_by_non_positive_float_raises_value_error(scalar: float):
  k = kernels.SEKernel(dim=3)
  with pytest.raises(ValueError, match="rho must be positive"):
    _ = scalar * k
  with pytest.raises(ValueError, match="rho must be positive"):
    _ = k * scalar


@pytest.mark.parametrize("other", [2, "a", None, jnp.ones(3)])
def test_mul_by_unsupported_type_raises_type_error(other: object):
  k = kernels.SEKernel(dim=3)
  with pytest.raises(TypeError):
    _ = k * other  # ty: ignore[unsupported-operator]


@pytest.mark.parametrize("other", [2, "a", None])
def test_rmul_by_unsupported_type_raises_type_error(other: object):
  k = kernels.SEKernel(dim=3)
  with pytest.raises(TypeError):
    _ = other * k  # ty: ignore[unsupported-operator]


def test_constant_call_error_names_the_offending_input(
  constant: base.ConstantKernel,
):
  params = constant.init()
  good = jnp.zeros((2, 3))
  bad = jnp.zeros((2, 2))
  with pytest.raises(checks.CheckError, match=r"^x1 has shape"):
    constant(params, bad, good)
  with pytest.raises(checks.CheckError, match=r"^x2 has shape"):
    constant(params, good, bad)


def test_sum_shape(sum_kernel: base.SumKernel):
  assert sum_kernel.shape == (3,)


def test_sum_init_holds_sub_kernel_params(sum_kernel: base.SumKernel):
  params = sum_kernel.init()
  assert jnp.allclose(params.k1.logrho, jnp.log(2.0))
  assert jnp.allclose(params.k2.logell, jnp.log(jnp.array([0.5, 1.0, 2.0])))


def test_sum_rejects_mismatched_shapes():
  with pytest.raises(checks.CheckError, match=r"k1\.shape"):
    base.SumKernel(k1=kernels.SEKernel(dim=2), k2=kernels.SEKernel(dim=3))


def test_sum_rejects_non_kernel():
  with pytest.raises(jt.TypeCheckError):
    base.SumKernel(
      k1=kernels.SEKernel(dim=3),
      k2="not a kernel",  # ty: ignore[invalid-argument-type]
    )


def test_sum_call_is_elementwise_sum(sum_kernel: base.SumKernel):
  params = sum_kernel.init()
  x1 = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 5
  x2 = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 5
  expected = sum_kernel.k1(params.k1, x1, x2) + sum_kernel.k2(params.k2, x1, x2)
  assert jnp.allclose(sum_kernel(params, x1, x2), expected)


def test_sum_diag_is_sum_of_diags_and_matches_call_diagonal(
  sum_kernel: base.SumKernel,
):
  params = sum_kernel.init()
  x = jnp.arange(9, dtype=jnp.float32).reshape(3, 3) / 5
  assert jnp.allclose(sum_kernel.diag(params, x), 2.0 + 1.0)
  assert jnp.allclose(
    sum_kernel.diag(params, x),
    jnp.diagonal(sum_kernel(params, x, x)),
    atol=1e-5,
  )


def test_sum_rejects_mismatched_trailing_input_dims(
  sum_kernel: base.SumKernel,
):
  with pytest.raises(jt.TypeCheckError):
    sum_kernel(sum_kernel.init(), jnp.zeros((2, 3)), jnp.zeros((2, 4)))


def test_add_of_kernels_builds_sum_preserving_order():
  k1 = kernels.SEKernel(dim=3)
  k2 = base.ConstantKernel(input_shape=(3,), rho=2.0)
  result = k1 + k2
  assert isinstance(result, base.SumKernel)
  assert result.k1 is k1
  assert result.k2 is k2


def test_add_rejects_mismatched_shapes():
  with pytest.raises(checks.CheckError, match=r"k1\.shape"):
    _ = kernels.SEKernel(dim=2) + kernels.SEKernel(dim=3)


def test_add_float_on_right_adds_trainable_constant_kernel():
  k = kernels.SEKernel(dim=3)
  result = k + 2.0
  assert isinstance(result, base.SumKernel)
  assert result.k1 is k
  assert isinstance(result.k2, base.ConstantKernel)
  assert result.k2.shape == (3,)
  assert jnp.allclose(result.init().k2.logrho, jnp.log(2.0))


def test_add_float_on_left_adds_trainable_constant_kernel():
  k = kernels.SEKernel(dim=3)
  result = 2.0 + k
  assert isinstance(result, base.SumKernel)
  assert isinstance(result.k1, base.ConstantKernel)
  assert result.k2 is k
  assert jnp.allclose(result.init().k1.logrho, jnp.log(2.0))


def test_adding_a_float_shifts_the_kernel_by_that_constant():
  k = kernels.SEKernel(dim=1)
  result = k + 2.0
  x = jnp.array([[0.0], [1.0]])
  assert jnp.allclose(
    result(result.init(), x, x),
    k(k.init(), x, x) + 2.0,
  )


@pytest.mark.parametrize("scalar", [0.0, -1.0])
def test_add_non_positive_float_raises_value_error(scalar: float):
  k = kernels.SEKernel(dim=3)
  with pytest.raises(ValueError, match="rho must be positive"):
    _ = scalar + k
  with pytest.raises(ValueError, match="rho must be positive"):
    _ = k + scalar


@pytest.mark.parametrize("other", [2, "a", None])
def test_add_unsupported_types_raise_type_error(other: object):
  k = kernels.SEKernel(dim=3)
  with pytest.raises(TypeError):
    _ = k + other  # ty: ignore[unsupported-operator]
  with pytest.raises(TypeError):
    _ = other + k  # ty: ignore[unsupported-operator]


def test_sums_and_products_compose():
  k = (kernels.SEKernel(dim=1) + 1.0) * kernels.SEKernel(dim=1, ell=2.0)
  x = jnp.linspace(0.0, 2.0, 4)[:, None]
  assert k(k.init(), x, x).shape == (4, 4)


def test_as_kernel_returns_kernels_unchanged():
  k = kernels.SEKernel(dim=3)
  assert base.as_kernel(k, (3,)) is k


def test_as_kernel_turns_a_float_into_a_constant_kernel_of_the_given_shape():
  result = base.as_kernel(2.0, (2, 3))
  assert isinstance(result, base.ConstantKernel)
  assert result.shape == (2, 3)
  assert jnp.allclose(result.init().logrho, jnp.log(2.0))


@pytest.mark.parametrize("other", [2, "a", None, [1.0]])
def test_as_kernel_rejects_other_types(other: object):
  with pytest.raises(TypeError, match="Expected a Kernel or a positive float"):
    base.as_kernel(other, (3,))
