"""Tests for grax.means.constant."""

import jax
import jax.numpy as jnp
import pytest

from grax import checks
from grax.means import constant


@pytest.fixture
def mean() -> constant.ConstantMean:
  return constant.ConstantMean(dim=3, init_offset=2.5)


def test_shape(mean: constant.ConstantMean):
  assert mean.shape == (3,)


def test_init_uses_init_offset(mean: constant.ConstantMean):
  assert jnp.allclose(mean.init().offset, 2.5)


def test_init_defaults_to_zero_offset():
  assert jnp.allclose(constant.ConstantMean(dim=3).init().offset, 0.0)


def test_call_returns_offset(mean: constant.ConstantMean):
  params = mean.init()
  x = jnp.arange(15, dtype=jnp.float32).reshape(5, 3)
  assert jnp.allclose(mean(params, x), jnp.full(5, 2.5))


def test_call_is_differentiable_in_offset(mean: constant.ConstantMean):
  x = jnp.zeros((5, 3))
  grad = jax.grad(lambda p: jnp.sum(mean(p, x)))(mean.init())
  assert jnp.allclose(grad.offset, 5.0)


def test_call_rejects_wrong_dim(mean: constant.ConstantMean):
  params = mean.init()
  with pytest.raises(checks.CheckError, match=r"expected \(None, 3\)"):
    mean(params, jnp.zeros((5, 2)))
