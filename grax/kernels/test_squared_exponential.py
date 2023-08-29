"""Tests for the squared exponential kernel.
"""

import numpy as np
import pickle
import pytest

from grax.kernels.squared_exponential import SquaredExponential
from os import path


def test_init():
  with pytest.raises(ValueError):
    SquaredExponential(1, 1)

  with pytest.raises(ValueError):
    SquaredExponential(1, 1, dim=-1)
    
  with pytest.raises(ValueError):
    SquaredExponential(1, [1, 1], dim=1)


def test_repr():
  result = repr(SquaredExponential(1, 1, dim=1, dtype='float16'))
  desired = 'SquaredExponential(rho=1.0, ell=1.0, dim=1, dtype=float16)'
  assert result == desired

  result = repr(SquaredExponential(1, [1, 1]))
  desired = 'SquaredExponential(rho=1.0, ell=[1.0, 1.0])'
  assert result == desired


def test_dim():
  assert SquaredExponential(1, 1, dim=1).dim == 1
  assert SquaredExponential(1, [1, 1]).dim == 2


def test_call():
  n, m = 8, 5
  kernels = [
      SquaredExponential(1, 1, dim=1),
      SquaredExponential(1, 1, dim=2),
      SquaredExponential(1, [1, 2])
  ]

  for kernel in kernels:
    x1 = np.linspace(0, 1, n*kernel.dim).reshape(n, -1)
    x2 = np.linspace(0, 1, m*kernel.dim).reshape(m, -1)

    assert kernel(x1, x2).shape == (n, m)
    assert kernel(x2, x1).shape == (m, n)
    assert kernel(x1).shape == (n, n)
    assert kernel(x1, diag=True).shape == (n,)

    with pytest.raises(ValueError):
      kernel(x1, x2, diag=True)


def test_goldens():
  with open(path.splitext(__file__)[0] + '.golden', 'rb') as f:
    goldens = pickle.load(f)

  for class_kwargs, call_kwargs, desired in goldens:
    result = SquaredExponential(**class_kwargs)(**call_kwargs)
    assert np.allclose(result, desired)
