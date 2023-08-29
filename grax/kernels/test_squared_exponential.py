"""Tests for the squared exponential kernel.
"""

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

