"""Tests that annotations can be resolved by runtime type checking."""

import importlib
import inspect
import pkgutil
import typing
from collections.abc import Callable
from collections.abc import Iterator
from types import ModuleType

import pytest

import grax


def _modules() -> Iterator[ModuleType]:
  for info in pkgutil.walk_packages(grax.__path__, "grax."):
    yield importlib.import_module(info.name)


def _callables(
  module: ModuleType,
) -> Iterator[tuple[str, Callable[..., object]]]:
  for name, obj in vars(module).items():
    if getattr(obj, "__module__", None) != module.__name__:
      continue
    if inspect.isfunction(obj):
      yield f"{module.__name__}.{name}", obj
    elif inspect.isclass(obj):
      for member_name, member in vars(obj).items():
        qualname = f"{module.__name__}.{name}.{member_name}"
        if isinstance(member, property):
          for accessor in (member.fget, member.fset):
            if accessor is not None:
              yield qualname, accessor
        elif inspect.isfunction(member):
          yield qualname, member


def _unresolved_name_error(function: Callable[..., object]) -> str | None:
  try:
    typing.get_type_hints(inspect.unwrap(function), include_extras=True)
  except NameError as e:
    return str(e)
  return None


@pytest.mark.parametrize("module", list(_modules()), ids=lambda m: m.__name__)
def test_annotations_resolve_from_module_globals(module: ModuleType):
  # Modules use `from __future__ import annotations`, so annotations are
  # strings resolved lazily in module scope by runtime type checking. A name
  # that isn't defined there (e.g. one that is scoped to a class) would make
  # the check fail or be skipped.
  failures = [
    f"{qualname}: {error}"
    for qualname, function in _callables(module)
    if (error := _unresolved_name_error(function)) is not None
  ]
  assert not failures, failures
