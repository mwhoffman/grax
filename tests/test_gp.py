"""Tests for grax.gp."""

import jax
import jax.numpy as jnp
import jaxtyping as jt
import pytest

from grax import checks
from grax import gp as gp_module
from grax.kernels import squared_exponential as se
from grax.means import zero


@pytest.fixture
def gp() -> gp_module.GP:
  kernel = se.SEKernel(dim=1, init_ell=jnp.array([1.0]))
  mean = zero.ZeroMean(dim=1)
  return gp_module.GP(kernel, mean, sn2=0.01)


def test_init_rejects_mismatched_kernel_and_mean_shape():
  kernel = se.SEKernel(dim=2, init_ell=jnp.ones(2))
  mean = zero.ZeroMean(dim=3)
  with pytest.raises(checks.CheckError, match=r"kernel\.shape"):
    gp_module.GP(kernel, mean)


def test_init_builds_params_from_kernel_and_mean_init(gp: gp_module.GP):
  assert jnp.allclose(gp._params.kernel.logrho, jnp.log(1.0))
  assert jnp.allclose(gp._params.kernel.logell, jnp.log(jnp.array([1.0])))
  assert gp._params.mean is None
  assert jnp.allclose(gp._params.logsn2, jnp.log(0.01))


def test_predict_with_no_data_returns_prior(gp: gp_module.GP):
  x = jnp.array([[0.0], [1.0], [2.0]])
  mu, s2 = gp.predict(x)
  assert jnp.allclose(mu, 0.0)
  assert jnp.allclose(s2, 1.0)


def test_predict_after_data_moves_toward_observations(gp: gp_module.GP):
  xd = jnp.array([[0.0], [1.0], [2.0], [3.0]])
  yd = jnp.array([0.0, 1.0, 2.0, 3.0])
  gp.add_data(xd, yd)

  mu, s2 = gp.predict(xd)
  assert jnp.allclose(mu, yd, atol=0.2)
  assert jnp.all(s2 < 1.0)


def test_add_data_accumulates(gp: gp_module.GP):
  gp.add_data(jnp.zeros((3, 1)), jnp.zeros(3))
  gp.add_data(jnp.zeros((2, 1)), jnp.zeros(2))
  assert gp._data is not None
  assert gp._data.x.shape[0] == 5
  assert gp._data.y.shape[0] == 5


def test_add_data_rejects_wrong_input_dim(gp: gp_module.GP):
  with pytest.raises(checks.CheckError, match=r"expected \(None, 1\)"):
    gp.add_data(jnp.zeros((3, 2)), jnp.zeros(3))


def test_add_data_rejects_mismatched_x_y_length(gp: gp_module.GP):
  # x and y share a jaxtyping-checked "n" dim, so this is caught by
  # jaxtyping itself rather than a manual checks.check_shape call.
  with pytest.raises(jt.TypeCheckError):
    gp.add_data(jnp.zeros((3, 1)), jnp.zeros(2))


def test_predict_rejects_wrong_input_dim(gp: gp_module.GP):
  with pytest.raises(checks.CheckError, match=r"expected \(None, 1\)"):
    gp.predict(jnp.zeros((3, 2)))


def test_grad_flows_through_predict(gp: gp_module.GP):
  gp.add_data(jnp.array([[0.0], [1.0]]), jnp.array([0.0, 1.0]))
  x = jnp.array([[0.5]])

  def loss(params: gp_module.GPParams) -> jax.Array:
    gp._params = params
    mu, _ = gp.predict(x)
    return jnp.sum(mu**2)

  grad = jax.grad(loss)(gp._params)
  assert jnp.isfinite(grad.kernel.logrho)
  assert jnp.all(jnp.isfinite(grad.kernel.logell))
  assert jnp.isfinite(grad.logsn2)
