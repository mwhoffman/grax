# Run the linter and type checker.
check:
  uv run ruff check src tests examples
  uv run ty check src tests examples

# Run the lint fixer and formatter.
fix:
  uv run ruff check src tests examples --fix
  uv run ruff format

# Run the test suite.
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
