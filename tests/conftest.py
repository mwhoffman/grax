"""Configuration for pytest."""

import pytest


def pytest_addoption(parser: pytest.Parser):
  """Add options for pytest."""
  parser.addoption(
    "--save-goldens",
    action="store_true",
    default=False,
    help="Save golden outputs for tests with empty goldens",
  )

  parser.addoption(
    "--update-goldens",
    action="store_true",
    default=False,
    help="Update golden outputs for failing tests",
  )
