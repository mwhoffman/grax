# Run the linter and type checker.
check:
    uv run ruff check src tests
    uv run ty check src tests

# Run the test suite.
test:
    uv run pytest --cov

# Sync dependencies, including the examples group.
sync:
    uv sync --group examples

# Upgrade dependencies, including the examples group.
upgrade:
    uv sync --upgrade --group examples
