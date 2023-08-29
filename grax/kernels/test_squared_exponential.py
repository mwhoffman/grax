"""Tests for the squared exponential kernel.
"""

import numpy as np
import unittest

from grax.kernels.squared_exponential import SquaredExponential


class TestSquaredExponential(unittest.TestCase):

  def test_init(self):
    with self.assertRaises(ValueError):
      SquaredExponential(1, 1)

    with self.assertRaises(ValueError):
      SquaredExponential(1, 1, dim=-1)
      
    with self.assertRaises(ValueError):
      SquaredExponential(1, [1, 1], dim=1)

  def test_repr(self):
    self.assertEqual(
      repr(SquaredExponential(1, 1, dim=1, dtype='float16')),
      'SquaredExponential(rho=1.0, ell=1.0, dim=1, dtype=float16)')

    self.assertEqual(
      repr(SquaredExponential(1, [1, 1])),
      'SquaredExponential(rho=1.0, ell=[1.0, 1.0])')

  def test_dim(self):
    kernel1 = SquaredExponential(1, 1, dim=1)
    kernel2 = SquaredExponential(1, [1, 1])

    self.assertEqual(kernel1.dim, 1)
    self.assertEqual(kernel2.dim, 2)

  def test_call(self):
    n, m = 8, 5
    kernels = [
        SquaredExponential(1, 1, dim=1),
        SquaredExponential(1, 1, dim=2),
        SquaredExponential(1, [1, 2])
    ]

    for kernel in kernels:
      x1 = np.linspace(0, 1, n*kernel.dim).reshape(n, -1)
      x2 = np.linspace(0, 1, m*kernel.dim).reshape(m, -1)

      self.assertEqual(kernel(x1, x2).shape, (n, m))
      self.assertEqual(kernel(x2, x1).shape, (m, n))
      self.assertEqual(kernel(x1).shape, (n, n))
      self.assertEqual(kernel(x1, diag=True).shape, (n,))

      with self.assertRaises(ValueError):
        kernel(x1, x2, diag=True)
