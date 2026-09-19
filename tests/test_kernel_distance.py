"""Tests for grax.kernels.distance."""

import jax.numpy as jnp

from grax.kernels import distance


def test_sqdist_matches_direct_computation():
  x1 = jnp.array([[0.0, 0.0], [1.0, 2.0], [-1.0, 0.5]])
  x2 = jnp.array([[1.0, 1.0], [0.0, -2.0]])

  expected = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
  assert jnp.allclose(distance.sqdist(x1, x2), expected, atol=1e-5)


def test_sqdist_is_non_negative():
  # Nearly-identical points with large offsets, for which the expanded form
  # (|a|^2 - 2 a.b + |b|^2) rounds to about -0.25 in float32 without clipping.
  x = jnp.array(
    [
      [1000.0016479492188, 1000.0020141601562],
      [999.9995727539062, 999.9999389648438],
      [1000.0001831054688, 999.9990234375],
      [999.99951171875, 1000.00048828125],
      [1000.0006713867188, 999.9990234375],
      [1000.002197265625, 999.998046875],
    ]
  )
  assert jnp.all(distance.sqdist(x, x) >= 0.0)
