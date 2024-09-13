"""Tests for the Gaussian likelihood."""

import numpy as np
import pytest
import testing

from grax.likelihoods.gaussian import Gaussian
from grax.typing import Array
from grax.utils import checks


def test_init():
  Gaussian(1.0)

  with pytest.raises(checks.CheckError):
    Gaussian(1)

  with pytest.raises(checks.CheckError):
    Gaussian(-1.0)

  with pytest.raises(checks.CheckError):
    Gaussian(np.array([1.0, 1.0]))


def test_repr():
  result = repr(Gaussian(1.0))
  expected = "Gaussian(sn2=1.0)"
  assert result == expected


def test_call():
  likelihood = Gaussian(1.0)
  likelihood(np.ones(10), np.zeros(10))

  with pytest.raises(checks.CheckError):
    likelihood(1.0, 1.0)

  with pytest.raises(checks.CheckError):
    likelihood(np.ones(10, dtype=int), np.zeros(10))

  with pytest.raises(checks.CheckError):
    likelihood(np.ones(10), np.zeros(10, dtype=int))

  with pytest.raises(checks.CheckError):
    likelihood(np.ones(10), np.zeros(5))


@testing.parameterize_goldens(
  dict(
    sn2=1.0,
    y=np.linspace(0, 1, 10),
    f=np.zeros(10),
  ),
)
def test_goldens(sn2, y, f) -> Array:
  return Gaussian(sn2)(y, f)
