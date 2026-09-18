"""Tests for grax.kernels.constant."""

import jax
import jax.numpy as jnp
import jaxtyping as jt
import pytest

from grax import checks
from grax.kernels import constant


@pytest.fixture
def kernel() -> constant.ConstantKernel:
  return constant.ConstantKernel(dim=3, rho=2.0)


def test_shape(kernel: constant.ConstantKernel):
  assert kernel.shape == (3,)


def test_init(kernel: constant.ConstantKernel):
  assert jnp.allclose(kernel.init().logrho, jnp.log(2.0))


def test_default_rho():
  params = constant.ConstantKernel(dim=3).init()
  assert jnp.allclose(params.logrho, jnp.log(1.0))


def test_construction_rejects_non_scalar_rho():
  with pytest.raises(jt.TypeCheckError):
    constant.ConstantKernel(dim=3, rho=jnp.array([1.0, 2.0]))


def test_call_returns_constant_matrix(kernel: constant.ConstantKernel):
  params = kernel.init()
  x1 = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
  x2 = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
  assert jnp.allclose(kernel(params, x1, x2), jnp.full((2, 4), 2.0))


def test_diag_returns_constant_vector(kernel: constant.ConstantKernel):
  params = kernel.init()
  x = jnp.zeros((5, 3))
  assert jnp.allclose(kernel.diag(params, x), jnp.full(5, 2.0))


def test_diag_matches_call_diagonal(kernel: constant.ConstantKernel):
  params = kernel.init()
  x = jnp.arange(9, dtype=jnp.float32).reshape(3, 3)
  assert jnp.allclose(kernel.diag(params, x), jnp.diag(kernel(params, x, x)))


def test_call_is_differentiable_in_logrho(kernel: constant.ConstantKernel):
  x = jnp.zeros((2, 3))
  grad = jax.grad(lambda p: jnp.sum(kernel(p, x, x)))(kernel.init())
  # d/dlogrho of 4 * exp(logrho) = 4 * rho.
  assert jnp.allclose(grad.logrho, 4 * 2.0)


def test_call_rejects_wrong_dim(kernel: constant.ConstantKernel):
  params = kernel.init()
  with pytest.raises(checks.CheckError, match=r"expected \(None, 3\)"):
    kernel(params, jnp.zeros((2, 2)), jnp.zeros((2, 2)))


def test_diag_rejects_wrong_dim(kernel: constant.ConstantKernel):
  params = kernel.init()
  with pytest.raises(checks.CheckError, match=r"expected \(None, 3\)"):
    kernel.diag(params, jnp.zeros((2, 2)))
