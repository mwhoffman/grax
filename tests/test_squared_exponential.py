"""Tests for the squared exponential kernel."""

import numpy as np
import pytest
import testing

from grax.kernels.squared_exponential import SquaredExponential
from grax.typing import Array


def test_init() -> None:
  with pytest.raises(ValueError):
    SquaredExponential(1.0, 1.0)

  with pytest.raises(ValueError):
    SquaredExponential(1.0, 1.0, dim=-1)

  with pytest.raises(ValueError):
    SquaredExponential(1.0, np.array([1.0, 1.0]), dim=1)


def test_repr() -> None:
  result = repr(SquaredExponential(1.0, 1.0, dim=1))
  expected = "SquaredExponential(rho=1.0, ell=1.0, dim=1)"
  assert result == expected

  result = repr(SquaredExponential(1.0, np.array([1.0, 1.0])))
  expected = "SquaredExponential(rho=1.0, ell=[1.0, 1.0])"
  assert result == expected


def test_dim() -> None:
  assert SquaredExponential(1.0, 1.0, dim=1).dim == 1
  assert SquaredExponential(1.0, np.array([1.0, 1.0])).dim == 2


def test_call() -> None:
  n, m = 8, 5
  kernels = [
    SquaredExponential(1.0, 1.0, dim=1),
    SquaredExponential(1.0, 1.0, dim=2),
    SquaredExponential(1.0, np.array([1.0, 2.0])),
  ]

  for kernel in kernels:
    x1 = np.linspace(0, 1, n * kernel.dim).reshape(n, -1)
    x2 = np.linspace(0, 1, m * kernel.dim).reshape(m, -1)

    assert kernel(x1, x2).shape == (n, m)
    assert kernel(x2, x1).shape == (m, n)
    assert kernel(x1).shape == (n, n)
    assert kernel(x1, diag=True).shape == (n,)

    with pytest.raises(ValueError):
      kernel(x1, x2, diag=True)


@testing.parameterize_goldens(
  dict(
    rho=1.0,
    ell=1.0,
    dim=1,
    x1=np.linspace(0, 1, 10)[:, None],
    x2=np.linspace(0, 1, 20)[:, None],
    diag=False,
  ),
  dict(
    rho=1.0,
    ell=1.0,
    dim=1,
    x1=np.linspace(0, 1, 10)[:, None],
    x2=None,
    diag=True,
  ),
)
def test_goldens(rho, ell, dim, x1, x2, diag) -> Array:
  return SquaredExponential(rho, ell, dim)(x1, x2, diag)
