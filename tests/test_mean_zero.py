"""Tests for grax.means.zero."""

import jax.numpy as jnp
import pytest

from grax import checks
from grax.means import zero


@pytest.fixture
def mean() -> zero.ZeroMean:
  return zero.ZeroMean(dim=3)


def test_shape(mean: zero.ZeroMean):
  assert mean.shape == (3,)


def test_init(mean: zero.ZeroMean):
  assert mean.init() is None


def test_call_returns_zeros(mean: zero.ZeroMean):
  params = mean.init()
  x = jnp.arange(15, dtype=jnp.float32).reshape(5, 3)
  assert jnp.array_equal(mean(params, x), jnp.zeros(5))


def test_call_rejects_wrong_dim(mean: zero.ZeroMean):
  params = mean.init()
  x = jnp.zeros((5, 2))
  with pytest.raises(checks.CheckError, match=r"expected \(None, 3\)"):
    mean(params, x)
