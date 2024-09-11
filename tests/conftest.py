"""Configuration for pytest.
"""


def pytest_addoption(parser):
  parser.addoption(
    "--save-goldens",
    action="store_true",
    default=False,
    help="Save golden outputs for tests with empty goldens"
  )

  parser.addoption(
    "--update-goldens",
    action="store_true",
    default=False,
    help="Update golden outputs for failing tests"
  )

