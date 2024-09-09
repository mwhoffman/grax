"""Tests for assertion checks.
"""

import numpy as np
import pytest

from grax import checks


def test_check_type() -> None:
  checks.check_type(1, int)
  checks.check_type(1., [float, int])

  with pytest.raises(ValueError):
    checks.check_type(1, float)


def test_check_rank() -> None:
  checks.check_rank(1, 0)
  checks.check_rank(np.array([1, 2]), 1)

  with pytest.raises(ValueError):
    checks.check_rank(1, 1)


def test_check_shape() -> None:
  checks.check_shape(np.zeros((4, 3, 2)), (4, None, 2))

  with pytest.raises(ValueError):
    checks.check_shape(np.zeros((2, 2, 2)), (2, 2))

  with pytest.raises(ValueError):
    checks.check_shape(np.zeros((2, 2, 2)), (1, 1, 1))


def test_check_type_and_rank() -> None:
  checks.check_type_and_rank(np.zeros((2, 2)), float, 2)


def test_check_type_and_shape() -> None:
  checks.check_type_and_shape(np.zeros((2, 2)), float, (2, 2))
