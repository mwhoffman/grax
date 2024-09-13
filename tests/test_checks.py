"""Tests for assertion checks."""

import numpy as np
import pytest

from grax.utils import checks


def test_check_type():
  checks.check_type(1, int)
  checks.check_type(1.0, [float, int])

  with pytest.raises(checks.CheckError):
    checks.check_type(1, float)


def test_check_rank():
  checks.check_rank(1, 0)
  checks.check_rank(np.array([1, 2]), 1)

  with pytest.raises(checks.CheckError):
    checks.check_rank(1, 1)


def test_check_shape():
  checks.check_shape(np.zeros((4, 3, 2)), (4, None, 2))

  with pytest.raises(checks.CheckError):
    checks.check_shape(np.zeros((2, 2, 2)), (2, 2))

  with pytest.raises(checks.CheckError):
    checks.check_shape(np.zeros((2, 2, 2)), (1, 1, 1))


def test_check_equal_shape():
  checks.check_equal_shape()
  checks.check_equal_shape(np.zeros((4, 3, 2)), np.ones((4, 3, 2)))

  with pytest.raises(checks.CheckError):
    checks.check_equal_shape(np.zeros((4, 3, 2)), np.ones((4, 3)))


def test_check_type_and_rank():
  checks.check_type_and_rank(np.zeros((2, 2)), float, 2)


def test_check_type_and_shape():
  checks.check_type_and_shape(np.zeros((2, 2)), float, (2, 2))


def test_check_positive():
  checks.check_positive(1)
  checks.check_positive(np.ones((2, 2)))

  with pytest.raises(checks.CheckError):
    checks.check_positive(-1)

  with pytest.raises(checks.CheckError):
    checks.check_positive(-1 * np.ones((2, 2)))
