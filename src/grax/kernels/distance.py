"""Distance computations shared between kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxtyping as jt

from grax import base


# On CPU, inputs with at most this many dimensions use exact differences, which
# is both more accurate and faster than the expanded form in that case. For
# more dimensions the expanded form's matrix multiplication is much cheaper
# than the differences, which don't stay fused (and materialize an array with
# a value per pair and dimension).
_MAX_DIRECT_DIM = 8

# Lengthscales dividing the inputs: one per dimension, or a single one.
Lengthscale = jt.Float[jt.Array, " d"] | jt.Float[jt.Array, ""] | None


def _direct(
  x1: jt.Float[jt.Array, "n d"],
  x2: jt.Float[jt.Array, "m d"],
  ell: Lengthscale = None,
) -> jt.Float[jt.Array, "n m"]:
  """Compute squared distances by summing squared differences."""
  # The differences are taken before dividing by the lengthscale, which is
  # exact for the differences, unlike differences of rounded scaled inputs.
  diff = x1[:, None, :] - x2[None, :, :]
  if ell is not None:
    diff = diff / ell
  return jnp.sum(diff**2, axis=-1)


def _expanded(
  x1: jt.Float[jt.Array, "n d"],
  x2: jt.Float[jt.Array, "m d"],
  ell: Lengthscale = None,
) -> jt.Float[jt.Array, "n m"]:
  """Compute squared distances using the expansion `|a|^2 - 2 a.b + |b|^2`.

  This is cheap for many dimensions but loses precision when the inputs are far
  from the origin relative to the distances between them. To avoid that for
  inputs that are merely offset (e.g. times given as years like 2000), the
  inputs are first shifted by their common mean, which leaves the distances
  unchanged. It does not help if the inputs are spread far apart with small
  distances within clusters.
  """
  # The distances don't depend on the center, so it is not differentiated.
  center = jax.lax.stop_gradient(jnp.mean(jnp.concatenate([x1, x2]), axis=0))
  x1 = x1 - center
  x2 = x2 - center
  if ell is not None:
    x1 = x1 / ell
    x2 = x2 / ell

  sqdist1 = jnp.sum(x1**2, axis=-1, keepdims=True)
  sqdist2 = jnp.sum(x2**2, axis=-1)
  # By default float32 matrix multiplications can use lower precision (such as
  # TensorFloat32 on some GPUs), which is far too inaccurate here.
  cross = jnp.matmul(x1, x2.T, precision=jax.lax.Precision.HIGHEST)
  # The expanded form can round to slightly below zero.
  return jnp.clip(sqdist1 - 2 * cross + sqdist2, min=0)


@base.typed
def sqdist(
  x1: jt.Float[jt.Array, "n d"],
  x2: jt.Float[jt.Array, "m d"],
  ell: Lengthscale = None,
) -> jt.Float[jt.Array, "n m"]:
  """Compute the squared euclidean distance between two sets of inputs.

  Args:
    x1: the first set of inputs.
    x2: the second set of inputs.
    ell: the lengthscales to divide the inputs by before computing distances,
      either one per dimension or a single one for all of them. If not given,
      this is the standard squared distance.

  Returns:
    An array D of shape (n, m) where D[i, j] is the squared euclidean distance
    between `x1[i] / ell` and `x2[j] / ell`.
  """
  if x1.shape[-1] > _MAX_DIRECT_DIM:
    return _expanded(x1, x2, ell)

  # This is decided when the computation is lowered for the platform it runs
  # on, which a check of the default backend would not get right.
  return jax.lax.platform_dependent(x1, x2, ell, cpu=_direct, default=_expanded)
