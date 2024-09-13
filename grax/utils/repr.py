"""Small utilities for writing repr functions."""

from collections.abc import Iterable
from typing import Any


def join(seq: Iterable[Any]) -> str:
  """Return a string representing any iterable as a comma separated list."""
  return ", ".join(str(x) for x in seq)


def join_dict(adict: dict[str, Any]) -> str:
  """Return a string representing key/value inputs (e.g. kwargs)."""
  return join(f"{k}={v}" for k, v in adict.items())
