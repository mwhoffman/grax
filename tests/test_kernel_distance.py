"""Tests for grax.kernels.distance."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jaxtyping as jt
import numpy as np
import pytest

from grax.kernels import distance


# Two clusters far apart with tiny separations within each. In float32 the
# expanded form rounds to -0.0625 for the pairs within a cluster (which is then
# clipped to 0), while the true squared distance is about 9.5e-7.
CLUSTERS = jnp.array([[1000.0], [1000.001], [-1000.0], [-1000.001]])
CLUSTER_PAIR_SQDIST = 0.0009765625**2

Algorithm = Callable[[jax.Array, jax.Array], jax.Array]
ALGORITHMS = [distance._direct, distance._expanded]


def _reference(x1: jax.Array, x2: jax.Array) -> np.ndarray:
  a = np.asarray(x1, dtype=np.float64)
  b = np.asarray(x2, dtype=np.float64)
  return np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithms_match_the_reference(algorithm: Algorithm):
  x1 = jnp.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]])
  x2 = jnp.array([[1.0, 1.0], [0.0, -2.0]])
  assert jnp.allclose(algorithm(x1, x2), _reference(x1, x2), atol=1e-5)


def test_sqdist_matches_the_reference_in_low_and_high_dimensions():
  key = jax.random.key(0)
  for dim in (2, distance._MAX_DIRECT_DIM + 4):
    x1 = jax.random.normal(key, (7, dim))
    x2 = jax.random.normal(jax.random.key(1), (5, dim))
    assert jnp.allclose(distance.sqdist(x1, x2), _reference(x1, x2), atol=1e-4)


def test_direct_is_exact_where_the_expanded_form_is_not():
  assert jnp.allclose(
    distance._direct(CLUSTERS, CLUSTERS)[0, 1],
    CLUSTER_PAIR_SQDIST,
    rtol=1e-3,
  )


def test_expanded_is_accurate_for_inputs_far_from_the_origin():
  # Without shifting by the mean, the expanded form's rounding error at this
  # offset (about 0.06 in float32) is larger than the distances themselves.
  x = 1000.0 + 0.1 * jnp.arange(6.0)[:, None]
  assert jnp.allclose(
    distance._expanded(x, x),
    _reference(x, x),
    atol=1e-4,
  )


def test_expanded_is_invariant_to_shifting_the_inputs():
  x1 = jnp.array([[0.0, 1.0], [2.0, -1.0], [0.5, 0.25]])
  x2 = jnp.array([[1.0, 0.0], [-2.0, 3.0]])
  shift = jnp.array([1500.0, 2000.0])
  assert jnp.allclose(
    distance._expanded(x1 + shift, x2 + shift),
    distance._expanded(x1, x2),
    atol=1e-3,
  )


def test_expanded_is_non_negative():
  assert jnp.all(distance._expanded(CLUSTERS, CLUSTERS) >= 0.0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_algorithms_have_the_same_gradient_as_the_reference(
  algorithm: Algorithm,
):
  x1 = jnp.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]])
  x2 = jnp.array([[1.0, 1.0], [0.0, -2.0]])

  def reference(a: jax.Array, b: jax.Array) -> jax.Array:
    return jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)

  weights = jnp.arange(6.0).reshape(3, 2) + 1.0
  grad = jax.grad(lambda a: jnp.sum(weights * algorithm(a, x2)))(x1)
  expected = jax.grad(lambda a: jnp.sum(weights * reference(a, x2)))(x1)
  assert jnp.allclose(grad, expected, atol=1e-4)


@pytest.mark.skipif(
  jax.default_backend() != "cpu",
  reason="The exact form is only used on CPU.",
)
def test_sqdist_uses_the_exact_form_for_low_dimensions_on_cpu():
  assert jnp.allclose(
    distance.sqdist(CLUSTERS, CLUSTERS)[0, 1],
    CLUSTER_PAIR_SQDIST,
    rtol=1e-3,
  )


def _clusters_with_dim(dim: int) -> jax.Array:
  # Padding with zeros leaves the distances unchanged.
  return jnp.concatenate([CLUSTERS, jnp.zeros((4, dim - 1))], axis=-1)


@pytest.mark.skipif(
  jax.default_backend() != "cpu",
  reason="The exact form is only used on CPU.",
)
def test_sqdist_uses_the_exact_form_up_to_the_threshold_on_cpu():
  x = _clusters_with_dim(distance._MAX_DIRECT_DIM)
  assert jnp.allclose(
    distance.sqdist(x, x)[0, 1],
    CLUSTER_PAIR_SQDIST,
    rtol=1e-3,
  )


def test_sqdist_uses_the_expanded_form_beyond_the_threshold():
  # The pairs within a cluster are clipped to zero by the expanded form. The
  # exact form isn't appropriate for many dimensions, so this pins that 64
  # counts as many, as well as the boundary just past the threshold.
  for dim in (distance._MAX_DIRECT_DIM + 1, 64):
    x = _clusters_with_dim(dim)
    assert distance.sqdist(x, x)[0, 1] == 0.0


def test_sqdist_is_differentiable_and_batchable():
  x = jnp.array([[0.0], [1.0], [3.0]])
  grad = jax.grad(lambda a: jnp.sum(distance.sqdist(a, x)))(x)
  # d/da_i of sum_j (a_i - x_j)^2 = 2 (3 a_i - sum_j x_j).
  assert jnp.allclose(grad, 2 * (3 * x - jnp.sum(x)))

  batched = jax.vmap(lambda a: distance.sqdist(a, x))(jnp.stack([x, x + 1.0]))
  assert batched.shape == (2, 3, 3)


def _reference_scaled(
  x1: jax.Array,
  x2: jax.Array,
  ell: jax.Array,
) -> np.ndarray:
  a = np.asarray(x1, dtype=np.float64)
  b = np.asarray(x2, dtype=np.float64)
  scale = np.asarray(ell, dtype=np.float64)
  return np.sum(((a[:, None, :] - b[None, :, :]) / scale) ** 2, axis=-1)


@pytest.mark.parametrize("dim", [2, 64])
@pytest.mark.parametrize("ell", [0.7, [0.5, 2.0]])
def test_sqdist_divides_by_a_scalar_or_per_dimension_ell(
  dim: int,
  ell: float | list[float],
):
  # Both the exact form (low dimensions, on CPU) and the expanded form (high).
  x1 = jax.random.normal(jax.random.key(0), (6, dim))
  x2 = jax.random.normal(jax.random.key(1), (4, dim))
  ell_array = jnp.asarray(ell)
  if ell_array.ndim == 1:
    ell_array = jnp.resize(ell_array, (dim,))
  assert jnp.allclose(
    distance.sqdist(x1, x2, ell_array),
    _reference_scaled(x1, x2, ell_array),
    rtol=1e-4,
    atol=1e-4,
  )


def test_sqdist_without_ell_is_the_standard_squared_distance():
  x1 = jnp.array([[0.0, 0.0], [1.0, 2.0]])
  x2 = jnp.array([[1.0, 1.0]])
  assert jnp.allclose(
    distance.sqdist(x1, x2),
    distance.sqdist(x1, x2, jnp.ones(2)),
  )
  assert jnp.allclose(distance.sqdist(x1, x2), _reference(x1, x2))


def test_sqdist_rejects_ell_with_the_wrong_length():
  with pytest.raises(jt.TypeCheckError):
    distance.sqdist(jnp.zeros((3, 2)), jnp.zeros((4, 2)), jnp.ones(3))


@pytest.mark.skipif(
  jax.default_backend() != "cpu",
  reason="The exact form is only used on CPU.",
)
def test_sqdist_with_ell_does_not_depend_on_the_offset_of_the_inputs():
  # Taking exact differences before dividing by ell makes the result identical
  # for offset inputs. Dividing the inputs first would round differently.
  x = 0.25 * jnp.arange(6.0)[:, None]
  ell = jnp.array([0.3])
  assert jnp.array_equal(
    distance.sqdist(x + 1000.0, x + 1000.0, ell),
    distance.sqdist(x, x, ell),
  )


@pytest.mark.parametrize("dim", [2, 64])
def test_sqdist_gradient_with_respect_to_ell_matches_the_reference(dim: int):
  x1 = jax.random.normal(jax.random.key(0), (5, dim))
  x2 = jax.random.normal(jax.random.key(1), (3, dim))
  ell = jnp.linspace(0.5, 2.0, dim)

  def reference(e: jax.Array) -> jax.Array:
    diff = (x1[:, None, :] - x2[None, :, :]) / e
    return jnp.sum(diff**2)

  grad = jax.grad(lambda e: jnp.sum(distance.sqdist(x1, x2, e)))(ell)
  assert jnp.allclose(grad, jax.grad(reference)(ell), rtol=1e-3, atol=1e-3)
