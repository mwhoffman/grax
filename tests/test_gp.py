"""Tests for grax.gp."""

import dataclasses

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


def test_loglikelihood_with_no_data_is_zero(gp: gp_module.GP):
  assert jnp.allclose(gp._loglikelihood(gp._params), 0.0)


def _reference_loglikelihood(
  gp: gp_module.GP,
  x: jax.Array,
  y: jax.Array,
  params: gp_module.GPParams,
) -> jax.Array:
  """Independent multivariate-normal log-likelihood for cross-checking."""
  sn2 = jnp.exp(params.logsn2)
  k = gp._kernel(params.kernel, x, x) + sn2 * jnp.eye(x.shape[0])
  r = y - gp._mean(params.mean, x)
  _, logdet = jnp.linalg.slogdet(k)
  n = x.shape[0]
  return (
    -0.5 * r @ jnp.linalg.solve(k, r)
    - 0.5 * logdet
    - 0.5 * n * jnp.log(2 * jnp.pi)
  )


def test_loglikelihood_matches_reference_value(gp: gp_module.GP):
  x = jnp.array([[0.0], [1.0], [2.0], [3.5]])
  y = jnp.array([0.1, 0.9, 0.3, -0.5])
  gp.add_data(x, y)

  ll = gp._loglikelihood(gp._params)
  ll_ref = _reference_loglikelihood(gp, x, y, gp._params)
  assert jnp.allclose(ll, ll_ref, atol=1e-4)


def test_loglikelihood_grad_matches_reference(gp: gp_module.GP):
  x = jnp.array([[0.0], [1.0], [2.0], [3.5]])
  y = jnp.array([0.1, 0.9, 0.3, -0.5])
  gp.add_data(x, y)

  grad = jax.grad(gp._loglikelihood)(gp._params)
  grad_ref = jax.grad(lambda p: _reference_loglikelihood(gp, x, y, p))(
    gp._params,
  )

  assert jnp.allclose(grad.logsn2, grad_ref.logsn2, atol=1e-3)
  assert jnp.allclose(grad.kernel.logrho, grad_ref.kernel.logrho, atol=1e-3)
  assert jnp.allclose(grad.kernel.logell, grad_ref.kernel.logell, atol=1e-3)


def test_fit_with_no_data_is_a_noop(gp: gp_module.GP):
  params_before = gp._params
  gp.fit(max_iter=10)
  assert jnp.allclose(gp._params.logsn2, params_before.logsn2)
  assert jnp.allclose(gp._params.kernel.logrho, params_before.kernel.logrho)


def test_fit(gp: gp_module.GP):
  # Noisy (rather than exact/noise-free) data, so sn2 doesn't get pushed
  # toward a slow-to-converge, near-degenerate optimum close to zero.
  key = jax.random.key(0)
  x = jnp.linspace(0.0, 10.0, 20)[:, None]
  y = jnp.sin(x[:, 0]) + 0.1 * jax.random.normal(key, (20,))
  gp.add_data(x, y)

  ll_before = gp._loglikelihood(gp._params)
  gp.fit(max_iter=200, tol=1e-4)
  ll_after = gp._loglikelihood(gp._params)

  # Ensure the log-likelihood improves.
  assert ll_after > ll_before

  # Perturb the parameters and ensure none are meaningfully better in
  # log-likelihood than the point we've found.
  flat, treedef = jax.tree_util.tree_flatten(gp._params)
  for trial_key in jax.random.split(jax.random.key(1), 10):
    leaf_keys = jax.random.split(trial_key, len(flat))
    perturbed = jax.tree_util.tree_unflatten(
      treedef,
      [
        leaf + 1e-3 * jax.random.normal(k, leaf.shape)
        for k, leaf in zip(leaf_keys, flat, strict=True)
      ],
    )
    assert gp._loglikelihood(perturbed) <= ll_after + 1e-2


def test_statistics_caches_for_current_params(gp: gp_module.GP):
  gp.add_data(jnp.array([[0.0], [1.0]]), jnp.array([0.0, 1.0]))
  stats1 = gp._statistics(gp._params)
  stats2 = gp._statistics(gp._params)
  assert stats1 is stats2


def test_statistics_cache_invalidated_by_add_data(gp: gp_module.GP):
  gp.add_data(jnp.array([[0.0], [1.0]]), jnp.array([0.0, 1.0]))
  stats1 = gp._statistics(gp._params)
  gp.add_data(jnp.array([[2.0]]), jnp.array([2.0]))
  stats2 = gp._statistics(gp._params)
  assert stats1 is not stats2


def test_statistics_cache_invalidated_by_params_assignment(gp: gp_module.GP):
  gp.add_data(jnp.array([[0.0], [1.0]]), jnp.array([0.0, 1.0]))
  stats1 = gp._statistics(gp._params)
  # A new (but value-equal) params object still invalidates the cache --
  # invalidation is triggered by assignment, not by a value comparison.
  gp._params = dataclasses.replace(gp._params)
  stats2 = gp._statistics(gp._params)
  assert stats1 is not stats2


def test_statistics_for_other_params_does_not_use_or_pollute_cache(
  gp: gp_module.GP,
):
  gp.add_data(jnp.array([[0.0], [1.0]]), jnp.array([0.0, 1.0]))
  cached = gp._statistics(gp._params)

  other_params = dataclasses.replace(gp._params)
  other_stats = gp._statistics(other_params)
  assert other_stats is not cached

  # The real cache (for self._params) must be untouched by that call.
  assert gp._statistics(gp._params) is cached
