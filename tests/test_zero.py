"""Tests for the zero mean."""

import numpy as np
import numpy.testing as nt
import pytest

from grax.means.zero import Zero


def test_init():
  Zero(1)

  with pytest.raises(ValueError):
    Zero(-1)


def test_repr():
  result = repr(Zero(1))
  expected = "Zero(dim=1)"
  assert result == expected


def test_call():
  mean = Zero(3)

  zeros0 = mean(np.zeros((10, 3)))
  zeros1 = mean(np.ones((10, 3)))

  nt.assert_equal(zeros0, np.zeros(10))
  nt.assert_equal(zeros1, np.zeros(10))

  with pytest.raises(ValueError):
    mean(np.ones((10, 2)))
