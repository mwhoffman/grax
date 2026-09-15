"""Tests for grax.kernels.squared_exponential."""

import jax
import jax.numpy as jnp
import jaxtyping as jt
import pytest

from grax.checks import CheckError
from grax.kernels.squared_exponential import SEKernel
from grax.kernels.squared_exponential import SEParams


@pytest.fixture
def kernel() -> SEKernel:
  return SEKernel(
    dim=3,
    init_rho=2.0,
    init_ell=jnp.array([0.5, 1.0, 2.0]),
  )


def test_shape(kernel: SEKernel):
  assert kernel.shape == (3,)


def test_init(kernel: SEKernel):
  params = kernel.init()
  assert jnp.allclose(params.logrho, jnp.log(2.0))
  assert jnp.allclose(params.logell, jnp.log(jnp.array([0.5, 1.0, 2.0])))


def test_default_init_rho_and_ell():
  kernel = SEKernel(dim=3)
  params = kernel.init()
  assert jnp.allclose(params.logrho, jnp.log(1.0))
  assert jnp.allclose(params.logell, jnp.log(jnp.ones(3)))


def test_construction_rejects_wrong_ell_length():
  with pytest.raises(CheckError, match=r"expected \(3,\)"):
    SEKernel(dim=3, init_ell=jnp.array([0.5, 1.0]))


def test_construction_rejects_non_scalar_rho():
  with pytest.raises(jt.TypeCheckError):
    SEKernel(dim=3, init_rho=jnp.array([1.0, 2.0]))


def test_construction_rejects_rank_two_ell():
  with pytest.raises(jt.TypeCheckError):
    SEKernel(dim=3, init_ell=jnp.zeros((3, 1)))


def test_call_rejects_params_with_wrong_dim(kernel: SEKernel):
  bad_params = SEParams(logrho=jnp.array(0.0), logell=jnp.zeros(2))
  x = jnp.zeros((1, 3))
  with pytest.raises(CheckError, match=r"expected \(3,\)"):
    kernel(bad_params, x, x)


def test_diag_rejects_params_with_wrong_dim(kernel: SEKernel):
  bad_params = SEParams(logrho=jnp.array(0.0), logell=jnp.zeros(2))
  x = jnp.zeros((1, 3))
  with pytest.raises(CheckError, match=r"expected \(3,\)"):
    kernel.diag(bad_params, x)


def test_call_rejects_x1_and_x2_with_wrong_dim(kernel: SEKernel):
  # x1 and x2 agree with each other (satisfying jaxtyping's own "n d"/"m d"
  # cross-consistency check), but not with the kernel's own dim.
  params = kernel.init()
  x1 = jnp.zeros((1, 2))
  x2 = jnp.zeros((4, 2))
  with pytest.raises(CheckError, match=r"expected \(None, 3\)"):
    kernel(params, x1, x2)


def test_diag_rejects_x_with_wrong_dim(kernel: SEKernel):
  params = kernel.init()
  x = jnp.zeros((1, 1))
  with pytest.raises(CheckError, match=r"expected \(None, 3\)"):
    kernel.diag(params, x)


def test_call_at_zero_distance_equals_rho(kernel: SEKernel):
  params = kernel.init()
  x = jnp.zeros((1, 3))
  k = kernel(params, x, x)
  assert jnp.allclose(k, 2.0)


def test_call_matches_closed_form_in_1d():
  kernel = SEKernel(dim=1, init_ell=jnp.array([1.0]))
  params = kernel.init()

  x1 = jnp.array([[0.0], [1.0]])
  x2 = jnp.array([[0.0], [2.0]])
  k = kernel(params, x1, x2)

  expected = jnp.exp(-0.5 * (x1 - x2.T) ** 2)
  assert jnp.allclose(k, expected)


def test_call_is_symmetric(kernel: SEKernel):
  params = kernel.init()
  x1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
  x2 = jnp.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
  assert jnp.allclose(kernel(params, x1, x2), kernel(params, x2, x1).T)


def test_call_shape(kernel: SEKernel):
  params = kernel.init()
  x1 = jnp.zeros((4, 3))
  x2 = jnp.zeros((5, 3))
  assert kernel(params, x1, x2).shape == (4, 5)


def test_call_rejects_mismatched_input_dims(kernel: SEKernel):
  params = kernel.init()
  x1 = jnp.zeros((4, 3))
  x2 = jnp.zeros((5, 2))
  with pytest.raises(jt.TypeCheckError):
    kernel(params, x1, x2)


def test_diag_matches_call_diagonal(kernel: SEKernel):
  params = kernel.init()
  x = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
  full = kernel(params, x, x)
  assert jnp.allclose(kernel.diag(params, x), jnp.diagonal(full))


def test_diag_equals_rho(kernel: SEKernel):
  params = kernel.init()
  x = jnp.zeros((5, 3))
  assert jnp.allclose(kernel.diag(params, x), 2.0)


def test_params_is_a_valid_pytree(kernel: SEKernel):
  params = kernel.init()
  leaves = jax.tree_util.tree_leaves(params)
  assert len(leaves) == 2


def test_grad_flows_through_call(kernel: SEKernel):
  params = kernel.init()
  x = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

  def loss(p: SEParams) -> jt.Float[jt.Array, ""]:
    return jnp.sum(kernel(p, x, x))

  grad = jax.grad(loss)(params)
  assert jnp.all(jnp.isfinite(grad.logrho))
  assert jnp.all(jnp.isfinite(grad.logell))
