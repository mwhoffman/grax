"""Tests for the squared exponential kernel.
"""

import numpy as np
import pytest

from grax import testing
from grax.kernels.squared_exponential import SquaredExponential


def test_init():
  with pytest.raises(ValueError):
    SquaredExponential(1, 1)

  with pytest.raises(ValueError):
    SquaredExponential(1, 1, dim=-1)
    
  with pytest.raises(ValueError):
    SquaredExponential(1, [1, 1], dim=1)


def test_repr():
  result = repr(SquaredExponential(1, 1, dim=1, dtype='float16'))
  expected = 'SquaredExponential(rho=1.0, ell=1.0, dim=1, dtype=float16)'
  assert result == expected

  result = repr(SquaredExponential(1, [1, 1]))
  expected = 'SquaredExponential(rho=1.0, ell=[1.0, 1.0])'
  assert result == expected


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


@testing.parameterize_goldens('test_squared_exponential.pkl')
def test_goldens(class_kwargs, call_kwargs):
  return SquaredExponential(**class_kwargs)(**call_kwargs)
