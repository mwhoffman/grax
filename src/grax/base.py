"""Common type definitions shared across grax."""

from collections.abc import Sequence

import beartype as bt
import jaxtyping as jt


PRNGKey = jt.Key[jt.Array, ""]

# Types for user-supplied initial values, which are stored as given and only
# converted to float arrays later, e.g. in a kernel's `init`. Only floats are
# accepted, not ints, so that they match `Float` arrays.
ScalarLike = float | jt.Float[jt.ArrayLike, ""]
VectorLike = ScalarLike | Sequence[float] | jt.Float[jt.ArrayLike, " d"]

# Decorator to runtime-check jaxtyping annotations.
typed = jt.jaxtyped(typechecker=bt.beartype)
