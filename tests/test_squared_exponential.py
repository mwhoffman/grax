"""Tests for the squared exponential kernel."""

import numpy as np
import numpy.testing as nt
import pytest
import testing
from jax import tree

from grax.kernels.squared_exponential import SquaredExponential
from grax.typing import Array
from grax.utils import checks


def test_init():
  SquaredExponential(1.0, 1.0, dim=1)

  with pytest.raises(checks.CheckError):
    SquaredExponential(1.0, 1.0)

  with pytest.raises(checks.CheckError):
    SquaredExponential(1.0, 1.0, dim=-1)

  with pytest.raises(checks.CheckError):
    SquaredExponential(1.0, np.array([1.0, 1.0]), dim=1)


def test_repr():
  result = repr(SquaredExponential(1.0, 1.0, dim=1))
  expected = "SquaredExponential(rho=1.0, ell=1.0, dim=1)"
  assert result == expected

  result = repr(SquaredExponential(1.0, np.array([1.0, 1.0])))
  expected = "SquaredExponential(rho=1.0, ell=[1.0, 1.0])"
  assert result == expected


def test_get_params():
  expected = (1.0, np.array([1.0, 1.0]))
  params = SquaredExponential(*expected).get_params()

  # Make sure the structures match.
  vals1, struct1 = tree.flatten(expected)
  vals2, struct2 = tree.flatten(params)

  # Make sure the leaves match.
  assert struct1 == struct2
  for val1, val2 in zip(vals1, vals2):
    nt.assert_allclose(val1, val2)


def test_dim():
  assert SquaredExponential(1.0, 1.0, dim=1).dim == 1
  assert SquaredExponential(1.0, np.array([1.0, 1.0])).dim == 2


def test_call():
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

    with pytest.raises(checks.CheckError):
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
  return SquaredExponential(rho, ell, dim)(x1, x2, diag=diag)
