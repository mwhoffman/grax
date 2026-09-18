# Run the linter and type checker.
[group("testing")]
check:
  uv run ruff check src tests examples
  uv run ty check src tests examples

# Run the test suite.
[group("testing")]
[arg("html", long, value="true", help="Generate HTML coverage.")]
test html="":
  uv run pytest --cov --cov-report=term {{ if html == "true" { "--cov-report=html" } else { "" } }}

# Sync dependencies.
[group("dependencies")]
sync:
  uv sync --group=examples

# Upgrade dependencies.
[group("dependencies")]
upgrade:
  uv sync --upgrade --group=examples
