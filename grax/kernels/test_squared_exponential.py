"""Tests for the squared exponential kernel.
"""

from grax.kernels import squared_exponential


def test_kernel():
  kernel = squared_exponential.SquaredExponential(1, 1, dim=1)
