"""Distance computations shared between kernels."""

import jax.numpy as jnp
import jaxtyping as jt

from grax import base


@base.typed
def sqdist(
  x1: jt.Float[jt.Array, "n d"],
  x2: jt.Float[jt.Array, "m d"],
) -> jt.Float[jt.Array, "n m"]:
  """Compute the squared euclidean distance between two sets of inputs.

  Args:
    x1: the first set of inputs.
    x2: the second set of inputs.

  Returns:
    An array D of shape (n, m) where D[i, j] is the squared euclidean distance
    between `x1[i]` and `x2[j]`. It is clipped to be non-negative, since the
    expanded form used here can round to slightly below zero.
  """
  sqdist1 = jnp.sum(x1**2, axis=-1, keepdims=True)
  sqdist2 = jnp.sum(x2**2, axis=-1)
  return jnp.clip(sqdist1 - 2 * jnp.matmul(x1, x2.T) + sqdist2, min=0)
