"""Type definitions."""

import jax
import jax.typing

Array = jax.Array
ArrayLike = jax.typing.ArrayLike
DTypeLike = jax.typing.DTypeLike

Float = ("bfloat16", "float16", "float32", "float64")
