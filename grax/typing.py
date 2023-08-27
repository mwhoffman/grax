"""Type definitions.
"""

import jax
import jax.numpy as jnp


Array = jax.Array
ArrayLike = jax.typing.ArrayLike
DType = type(jnp.float32)

del jax
del jnp

