"""Type definitions.
"""

import jax


Array = jax.Array
ArrayLike = jax.typing.ArrayLike
DType = jax._src.typing.DType
DTypeLike = jax._src.typing.DTypeLike

del jax
