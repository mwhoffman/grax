"""Tests for grax.checks."""

import warnings

import jax.numpy as jnp
import pytest

from grax import checks


def test_check_shape_accepts_matching_shape():
  checks.check_shape(jnp.zeros((2, 3)), (2, 3))


def test_check_shape_accepts_sequences_and_scalars():
  # Converting first avoids `jnp.shape`'s deprecation warning for sequences.
  with warnings.catch_warnings():
    warnings.simplefilter("error")
    checks.check_shape([0.5, 1.0], (2,))
    checks.check_shape((0.5, 1.0), (2,))
    checks.check_shape(0.5, ())


def test_check_shape_does_not_promote_by_default():
  # A single point without a batch axis must not pass as a batch of one.
  with pytest.raises(checks.CheckError, match=r"shape \(3,\), expected"):
    checks.check_shape(jnp.zeros(3), (None, 3))
  with pytest.raises(checks.CheckError, match=r"shape \(\), expected \(1,\)"):
    checks.check_shape(0.5, (1,))


def test_check_shape_promotes_when_asked():
  checks.check_shape(0.5, (1,), promote=True)
  checks.check_shape([0.5, 1.0], (2,), promote=True)
  checks.check_shape(jnp.zeros(3), (1, 3), promote=True)


def test_check_shape_error_uses_the_name():
  with pytest.raises(
    checks.CheckError, match=r"^ell has shape \(2,\), expected"
  ):
    checks.check_shape([0.5, 1.0], (3,), name="ell")


def test_check_shape_name_defaults_to_array():
  with pytest.raises(checks.CheckError, match=r"^array has shape"):
    checks.check_shape([0.5, 1.0], (3,))


def test_check_none_or_shape_passes_the_name_through():
  with pytest.raises(checks.CheckError, match=r"^ell has shape"):
    checks.check_none_or_shape([0.5, 1.0], (3,), name="ell")


def test_check_positive_name_defaults_to_array():
  with pytest.raises(checks.CheckError, match=r"^array must be positive"):
    checks.check_positive(-1.0)


def test_check_shape_promote_reports_the_original_shape():
  with pytest.raises(checks.CheckError, match=r"shape \(\), expected \(3,\)"):
    checks.check_shape(0.5, (3,), promote=True)


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


def test_check_none_or_shape_passes_promote_through():
  checks.check_none_or_shape(0.5, (1,), promote=True)
  with pytest.raises(checks.CheckError):
    checks.check_none_or_shape(0.5, (1,))


def test_check_positive_accepts_positive_values():
  checks.check_positive(0.5, name="x")
  checks.check_positive([0.5, 2.0], name="x")
  checks.check_positive(jnp.array([[1.0, 2.0]]), name="x")


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), [1.0, 0.0]])
def test_check_positive_rejects_non_positive_values(value: float):
  with pytest.raises(checks.CheckError, match="x must be positive"):
    checks.check_positive(value, name="x")


def test_check_none_or_positive():
  checks.check_none_or_positive(None, name="x")
  checks.check_none_or_positive(1.0, name="x")
  with pytest.raises(checks.CheckError, match="x must be positive"):
    checks.check_none_or_positive(-1.0, name="x")
