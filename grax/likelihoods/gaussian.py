"""Implementation of the standard Gaussian likelihood."""

import jax.numpy as jnp
from flax import nnx

from grax import types
from grax.likelihoods import base
from grax.utils import checks


LOG2PI = jnp.log(2*jnp.pi)


class Gaussian(base.Likelihood):
  """The Gaussian likelihood, i.e. for standard GP regression."""

  def __init__(self, sn2: types.ArrayLike):
    """Initialize a Gaussian likelihood model.

    Args:
      sn2: the noise variance.
    """
    sn2 = jnp.asarray(sn2)

    checks.check_type_and_rank(sn2, types.Float, 0)
    checks.check_positive(sn2)

    self.logsn2 = nnx.Param(jnp.log(sn2))

  @property
  def sn2(self):
    """Return the variance of the model."""

    # TODO: deal with numerical issues when the log parameters -> -inf.
    return jnp.exp(self.logsn2.value)

  def __call__(self, y: types.ArrayLike, f: types.ArrayLike) -> types.Array:
    """Evaluate the marginal log-likelihood."""
    y = jnp.asarray(y)
    f = jnp.asarray(f)

    checks.check_type_and_rank(y, types.Float, 1)
    checks.check_type_and_rank(f, types.Float, 1)
    checks.check_equal_shape(y, f)

    return -0.5 * ((y - f) ** 2 / self.sn2 + LOG2PI + self.logsn2)
