"""Tests for the Gaussian likelihood."""

import numpy as np
import numpy.testing as nt
import pytest
import testing
from jax import tree

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


def test_get_params():
  expected = (1.0,)
  params = Gaussian(*expected).get_params()

  # Make sure the structures match.
  vals1, struct1 = tree.flatten(expected)
  vals2, struct2 = tree.flatten(params)

  # Make sure the leaves match.
  assert struct1 == struct2
  for val1, val2 in zip(vals1, vals2):
    nt.assert_allclose(val1, val2)


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
