"""Implementation of the standard Gaussian likelihood."""

import jax.numpy as jnp

from grax import typing
from grax.likelihoods import base
from grax.utils import checks


class Gaussian(base.Likelihood):
  """The Gaussian likelihood, i.e. for standard GP regression."""

  def __init__(self, sn2: typing.ArrayLike):
    """Initialize a Gaussian likelihood model.

    Args:
      sn2: the noise variance.
    """
    sn2 = jnp.asarray(sn2)

    checks.check_type_and_rank(sn2, typing.Float, 0)
    checks.check_positive(sn2)

    self.sn2 = sn2

  def __call__(self, y: typing.ArrayLike, f: typing.ArrayLike) -> typing.Array:
    """Evaluate the marginal log-likelihood."""
    y = jnp.asarray(y)
    f = jnp.asarray(f)

    checks.check_type_and_rank(y, typing.Float, 1)
    checks.check_type_and_rank(f, typing.Float, 1)
    checks.check_equal_shape(y, f)

    return -0.5 * ((y - f) ** 2 / self.sn2 + jnp.log(2 * jnp.pi * self.sn2))
