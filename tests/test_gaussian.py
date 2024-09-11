"""Tests for the Gaussian likelihood.
"""

import numpy as np
import pytest
import testing

from grax.likelihoods.gaussian import Gaussian
from grax.typing import Array


def test_init() -> None:
  Gaussian(1.)

  with pytest.raises(ValueError):
    Gaussian(1)

  with pytest.raises(ValueError):
    Gaussian(-1.)

  with pytest.raises(ValueError):
    Gaussian(np.array([1., 1.]))


def test_call() -> None:
  likelihood = Gaussian(1.)
  likelihood(np.ones(10), np.zeros(10))

  with pytest.raises(ValueError):
    likelihood(1.0, 1.0)

  with pytest.raises(ValueError):
    likelihood(np.ones(10, dtype=int), np.zeros(10))

  with pytest.raises(ValueError):
    likelihood(np.ones(10), np.zeros(10, dtype=int))

  with pytest.raises(ValueError):
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

