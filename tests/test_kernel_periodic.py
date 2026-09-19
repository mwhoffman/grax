"""Tests for grax.kernels.periodic."""

import jax
import jax.numpy as jnp
import jaxtyping as jt
import pytest

from grax import checks
from grax import kernels
from grax.kernels import periodic


@pytest.fixture
def kernel() -> periodic.PeriodicKernel:
  return periodic.PeriodicKernel(ell=0.7, period=2.0)


def test_shape(kernel: periodic.PeriodicKernel):
  assert kernel.shape == (1,)


def test_init(kernel: periodic.PeriodicKernel):
  params = kernel.init()
  assert jnp.allclose(params.logell, jnp.log(0.7))
  assert jnp.allclose(params.logperiod, jnp.log(2.0))


def test_default_ell_and_period():
  params = periodic.PeriodicKernel().init()
  assert jnp.allclose(params.logell, jnp.log(1.0))
  assert jnp.allclose(params.logperiod, jnp.log(1.0))


@pytest.mark.parametrize("name", ["ell", "period"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_construction_rejects_non_positive_values(name: str, value: float):
  with pytest.raises(ValueError, match=f"^{name} must be positive"):
    periodic.PeriodicKernel(**{name: value})


@pytest.mark.parametrize("name", ["ell", "period"])
def test_construction_rejects_ints_and_vectors(name: str):
  with pytest.raises(jt.TypeCheckError):
    periodic.PeriodicKernel(**{name: 2})
  with pytest.raises(jt.TypeCheckError):
    periodic.PeriodicKernel(**{name: [1.0, 2.0]})  # ty: ignore[invalid-argument-type]


def test_call_rejects_x_that_is_not_one_dimensional(
  kernel: periodic.PeriodicKernel,
):
  params = kernel.init()
  x = jnp.zeros((2, 2))
  with pytest.raises(checks.CheckError, match=r"^x1 has shape"):
    kernel(params, x, x)
  with pytest.raises(checks.CheckError, match=r"^x has shape"):
    kernel.diag(params, x)


def test_call_matches_closed_form(kernel: periodic.PeriodicKernel):
  x1 = jnp.array([[0.0], [0.3], [1.1]])
  x2 = jnp.array([[0.5], [2.0]])
  dist = jnp.abs(x1 - x2.T)
  expected = jnp.exp(-2 * (jnp.sin(jnp.pi * dist / 2.0) / 0.7) ** 2)
  assert jnp.allclose(kernel(kernel.init(), x1, x2), expected, atol=1e-6)


def test_call_at_zero_distance_equals_one(kernel: periodic.PeriodicKernel):
  x = jnp.array([[0.4]])
  assert jnp.allclose(kernel(kernel.init(), x, x), 1.0)


def test_call_is_periodic(kernel: periodic.PeriodicKernel):
  # Inputs a whole number of periods apart are perfectly correlated, and
  # shifting one set of inputs by whole periods changes nothing.
  params = kernel.init()
  x = jnp.array([[0.3], [1.1]])
  assert jnp.allclose(jnp.diagonal(kernel(params, x, x + 2.0)), 1.0, atol=1e-5)
  assert jnp.allclose(
    kernel(params, x, x + 6.0),
    kernel(params, x, x),
    atol=1e-4,
  )


def test_call_is_symmetric(kernel: periodic.PeriodicKernel):
  params = kernel.init()
  x1 = jnp.array([[0.0], [0.4], [1.3]])
  x2 = jnp.array([[0.9], [2.2]])
  assert jnp.allclose(kernel(params, x1, x2), kernel(params, x2, x1).T)


def test_call_shape(kernel: periodic.PeriodicKernel):
  k = kernel(kernel.init(), jnp.zeros((4, 1)), jnp.zeros((5, 1)))
  assert k.shape == (4, 5)


def test_diag_equals_one(kernel: periodic.PeriodicKernel):
  assert jnp.allclose(kernel.diag(kernel.init(), jnp.zeros((5, 1))), 1.0)


def test_diag_matches_call_diagonal(kernel: periodic.PeriodicKernel):
  params = kernel.init()
  x = jnp.array([[0.0], [0.4], [1.3]])
  assert jnp.allclose(
    kernel.diag(params, x), jnp.diagonal(kernel(params, x, x))
  )


def test_params_is_a_valid_pytree(kernel: periodic.PeriodicKernel):
  assert len(jax.tree_util.tree_leaves(kernel.init())) == 2


def test_grads_are_finite_including_at_coincident_inputs(
  kernel: periodic.PeriodicKernel,
):
  x = jnp.array([[0.0], [0.4], [1.3]])
  params = kernel.init()

  def loss(p: periodic.PeriodicParams, x: jax.Array) -> jax.Array:
    return jnp.sum(kernel(p, x, x))

  grad_params, grad_x = jax.grad(loss, argnums=(0, 1))(params, x)
  assert jnp.isfinite(grad_params.logell)
  assert jnp.isfinite(grad_params.logperiod)
  assert jnp.all(jnp.isfinite(grad_x))


def test_is_positive_semi_definite_in_one_dimension(
  kernel: periodic.PeriodicKernel,
):
  x = jnp.linspace(0.0, 5.0, 30)[:, None]
  k = kernel(kernel.init(), x, x)
  assert jnp.linalg.eigvalsh(k).min() > -1e-4


def test_can_be_multiplied_with_a_one_dimensional_kernel(
  kernel: periodic.PeriodicKernel,
):
  product = kernel * kernels.SEKernel(dim=1)
  x = jnp.linspace(0.0, 3.0, 5)[:, None]
  assert product(product.init(), x, x).shape == (5, 5)
