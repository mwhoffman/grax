"""Type definitions.
"""

import jax
from typing import Union


Array = jax.Array
ArrayLike = Union[jax.typing.ArrayLike, list[bool], list[int], list[float]]
DType = jax._src.typing.DType
DTypeLike = jax._src.typing.DTypeLike

del jax
del Union
