"""Tests for grax.optimization."""

import dataclasses
import warnings

import jax
import jax.numpy as jnp
import jaxtyping as jt
import optax
import pytest

from grax import base
from grax import optimization


@base.typed
@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, kw_only=True)
class ShapeCheckedParams:
  """Params whose fields have runtime-checked shapes, like a GP's."""

  a: jt.Float[jt.Array, ""]
  b: jt.Float[jt.Array, " d"]


def quadratic(x: jax.Array) -> jax.Array:
  return jnp.sum((x - jnp.array([1.0, -2.0, 3.0])) ** 2)


def test_minimize_finds_the_minimum_of_a_quadratic():
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    result = optimization.minimize(quadratic, jnp.zeros(3))

  assert result.status is optimization.Status.CONVERGED
  assert jnp.allclose(result.params, jnp.array([1.0, -2.0, 3.0]), atol=1e-3)
  assert result.grad_norm < 1e-3
  assert result.n_iter > 0


def test_minimize_supports_shape_checked_pytree_params():
  # L-BFGS stacks a history dimension onto its internal copy of the params,
  # which would break shape checks if `minimize` optimized over them directly.
  def objective(p: ShapeCheckedParams) -> jax.Array:
    return (p.a - 1.0) ** 2 + jnp.sum((p.b - jnp.array([1.0, 2.0])) ** 2)

  init = ShapeCheckedParams(a=jnp.array(0.0), b=jnp.zeros(2))
  result = optimization.minimize(objective, init)

  assert isinstance(result.params, ShapeCheckedParams)
  assert result.status is optimization.Status.CONVERGED
  assert jnp.allclose(result.params.a, 1.0, atol=1e-3)
  assert jnp.allclose(result.params.b, jnp.array([1.0, 2.0]), atol=1e-3)


def test_minimize_respects_max_iter():
  def rosenbrock(x: jax.Array) -> jax.Array:
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2

  init = jnp.array([-1.2, 1.0])
  result = optimization.minimize(rosenbrock, init, max_iter=2)

  assert result.status is optimization.Status.MAX_ITER
  assert result.n_iter == 2
  assert rosenbrock(result.params) < rosenbrock(init)


def test_minimize_stops_and_warns_when_the_line_search_fails(
  monkeypatch: pytest.MonkeyPatch,
):
  # With a single line search step, the first unit step overshoots this steep
  # quadratic, so the search fails and optax would take an unsafe step that
  # increases the objective.
  lbfgs = optax.lbfgs
  monkeypatch.setattr(
    optax,
    "lbfgs",
    lambda: lbfgs(
      linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=1),
    ),
  )

  def steep(x: jax.Array) -> jax.Array:
    return 100.0 * jnp.sum(x**2)

  init = jnp.array([0.3])
  with pytest.warns(RuntimeWarning, match="line search"):
    result = optimization.minimize(steep, init)

  # The failed steps are discarded, leaving the parameters untouched.
  assert result.status is optimization.Status.LINE_SEARCH_FAILED
  assert jnp.array_equal(result.params, init)
  assert result.grad_norm >= 1e-3
