"""Tests for grax.checks."""

import jax.numpy as jnp
import pytest

from grax import checks


def test_check_shape_accepts_matching_shape():
  checks.check_shape(jnp.zeros((2, 3)), (2, 3))


def test_check_shape_accepts_wildcard_dims():
  checks.check_shape(jnp.zeros((2, 3)), (None, 3))


def test_check_shape_rejects_wrong_rank():
  with pytest.raises(checks.CheckError, match=r"expected \(2, 3\)"):
    checks.check_shape(jnp.zeros((2, 3, 4)), (2, 3))


def test_check_shape_rejects_wrong_size():
  with pytest.raises(checks.CheckError, match=r"expected \(2, 3\)"):
    checks.check_shape(jnp.zeros((2, 4)), (2, 3))


def test_check_none_or_shape_accepts_none():
  checks.check_none_or_shape(None, (2, 3))


def test_check_none_or_shape_accepts_matching_shape():
  checks.check_none_or_shape(jnp.zeros((2, 3)), (2, 3))


def test_check_none_or_shape_rejects_wrong_size():
  with pytest.raises(checks.CheckError, match=r"expected \(2, 3\)"):
    checks.check_none_or_shape(jnp.zeros((2, 4)), (2, 3))
