"""Tests for grax.kernels.rational_quadratic."""

import jax
import jax.numpy as jnp
import jaxtyping as jt
import pytest

from grax import checks
from grax.kernels import rational_quadratic as rq
from grax.kernels import squared_exponential as se


@pytest.fixture
def kernel() -> rq.RQKernel:
  return rq.RQKernel(dim=3, ell=jnp.array([0.5, 1.0, 2.0]), alpha=2.0)


def test_shape(kernel: rq.RQKernel):
  assert kernel.shape == (3,)


def test_init(kernel: rq.RQKernel):
  params = kernel.init()
  assert jnp.allclose(params.logell, jnp.log(jnp.array([0.5, 1.0, 2.0])))
  assert jnp.allclose(params.logalpha, jnp.log(2.0))


def test_default_ell_and_alpha():
  params = rq.RQKernel(dim=3).init()
  assert jnp.allclose(params.logell, jnp.log(jnp.ones(3)))
  assert jnp.allclose(params.logalpha, jnp.log(1.0))


def test_init_accepts_a_scalar_ell_when_dim_is_one():
  params = rq.RQKernel(dim=1, ell=0.8).init()
  assert jnp.allclose(params.logell, jnp.log(jnp.array([0.8])))


@pytest.mark.parametrize("ell", [[0.5, 1.0], (0.5, 1.0)])
def test_init_accepts_a_sequence_ell(ell: list[float] | tuple[float, ...]):
  params = rq.RQKernel(dim=2, ell=ell).init()
  assert jnp.allclose(params.logell, jnp.log(jnp.array([0.5, 1.0])))


def test_construction_rejects_a_scalar_ell_when_dim_is_above_one():
  with pytest.raises(checks.CheckError, match=r"shape \(\), expected \(3,\)"):
    rq.RQKernel(dim=3, ell=0.8)


def test_construction_rejects_wrong_ell_length():
  with pytest.raises(checks.CheckError, match=r"expected \(3,\)"):
    rq.RQKernel(dim=3, ell=jnp.array([0.5, 1.0]))


def test_construction_rejects_non_scalar_alpha():
  with pytest.raises(jt.TypeCheckError):
    rq.RQKernel(dim=3, alpha=jnp.array([1.0, 2.0]))


def test_construction_rejects_int_alpha():
  with pytest.raises(jt.TypeCheckError):
    rq.RQKernel(dim=3, alpha=2)


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("nan")])
def test_construction_rejects_non_positive_alpha(alpha: float):
  with pytest.raises(ValueError, match="alpha must be positive"):
    rq.RQKernel(dim=3, alpha=alpha)


def test_construction_rejects_non_positive_ell():
  with pytest.raises(ValueError, match="ell must be positive"):
    rq.RQKernel(dim=2, ell=[1.0, 0.0])


def test_call_rejects_params_with_wrong_dim(kernel: rq.RQKernel):
  bad_params = rq.RQParams(logell=jnp.zeros(2), logalpha=jnp.array(0.0))
  x = jnp.zeros((1, 3))
  with pytest.raises(checks.CheckError, match=r"expected \(3,\)"):
    kernel(bad_params, x, x)
  with pytest.raises(checks.CheckError, match=r"expected \(3,\)"):
    kernel.diag(bad_params, x)


def test_call_rejects_x_with_wrong_dim(kernel: rq.RQKernel):
  params = kernel.init()
  x = jnp.zeros((2, 2))
  with pytest.raises(checks.CheckError, match=r"expected \(None, 3\)"):
    kernel(params, x, x)
  with pytest.raises(checks.CheckError, match=r"expected \(None, 3\)"):
    kernel.diag(params, x)


def test_call_at_zero_distance_equals_one(kernel: rq.RQKernel):
  x = jnp.zeros((1, 3))
  assert jnp.allclose(kernel(kernel.init(), x, x), 1.0)


def test_call_matches_closed_form_in_1d_with_unit_alpha():
  kernel = rq.RQKernel(dim=1, ell=1.0, alpha=1.0)
  x1 = jnp.array([[0.0], [1.0]])
  x2 = jnp.array([[0.0], [2.0]])
  expected = 1.0 / (1.0 + 0.5 * (x1 - x2.T) ** 2)
  assert jnp.allclose(kernel(kernel.init(), x1, x2), expected)


def test_call_matches_closed_form_with_lengthscales_and_alpha():
  kernel = rq.RQKernel(dim=2, ell=[0.5, 2.0], alpha=3.0)
  x1 = jnp.array([[0.0, 0.0], [1.0, 2.0]])
  x2 = jnp.array([[1.0, 1.0]])
  d = jnp.sum(((x1 - x2) / jnp.array([0.5, 2.0])) ** 2, axis=-1, keepdims=True)
  expected = (1.0 + d / (2 * 3.0)) ** -3.0
  assert jnp.allclose(kernel(kernel.init(), x1, x2), expected, atol=1e-6)


def test_large_alpha_matches_the_squared_exponential_kernel():
  # In float32 the naive form of the kernel is off by ~1e-2 at this alpha.
  ell = jnp.array([0.7])
  x = jnp.linspace(-2.0, 2.0, 7)[:, None]
  rq_kernel = rq.RQKernel(dim=1, ell=ell, alpha=1e6)
  se_kernel = se.SEKernel(dim=1, ell=ell)
  assert jnp.allclose(
    rq_kernel(rq_kernel.init(), x, x),
    se_kernel(se_kernel.init(), x, x),
    atol=1e-5,
  )


def test_call_is_symmetric(kernel: rq.RQKernel):
  params = kernel.init()
  x1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
  x2 = jnp.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
  assert jnp.allclose(kernel(params, x1, x2), kernel(params, x2, x1).T)


def test_call_shape(kernel: rq.RQKernel):
  assert kernel(kernel.init(), jnp.zeros((4, 3)), jnp.zeros((5, 3))).shape == (
    4,
    5,
  )


def test_diag_equals_one(kernel: rq.RQKernel):
  assert jnp.allclose(kernel.diag(kernel.init(), jnp.zeros((5, 3))), 1.0)


def test_diag_matches_call_diagonal(kernel: rq.RQKernel):
  params = kernel.init()
  x = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
  assert jnp.allclose(
    kernel.diag(params, x),
    jnp.diagonal(kernel(params, x, x)),
  )


def test_params_is_a_valid_pytree(kernel: rq.RQKernel):
  assert len(jax.tree_util.tree_leaves(kernel.init())) == 2


def test_grad_is_finite_including_for_large_alpha():
  x = jnp.array([[0.0, 0.0], [1.0, 0.0]])
  for alpha in (1.0, 1e6):
    kernel = rq.RQKernel(dim=2, alpha=alpha)

    def loss(p: rq.RQParams, kernel: rq.RQKernel = kernel) -> jax.Array:
      return jnp.sum(kernel(p, x, x))

    grad = jax.grad(loss)(kernel.init())
    assert jnp.all(jnp.isfinite(grad.logell))
    assert jnp.isfinite(grad.logalpha)
