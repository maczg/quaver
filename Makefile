.PHONY: lint format-check format typecheck test check fix

check: lint format-check typecheck test

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

format:
	uv run ruff format .

typecheck:
	uv run mypy quaver/

test:
	uv run pytest

fix:
	uv run ruff check --fix .
	$(MAKE) format
