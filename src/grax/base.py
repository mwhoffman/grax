"""Common type definitions shared across grax."""

import beartype as bt
import jaxtyping as jt


PRNGKey = jt.Key[jt.Array, ""]

# Decorator to runtime-check jaxtyping annotations.
typed = jt.jaxtyped(typechecker=bt.beartype)
