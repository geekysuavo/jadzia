.PHONY: all clean doc lint ruff test ty

all: lint test

lint: ruff ty

test:
	uv run --group test pytest -v --cov=jadzia --cov-report=term:skip-covered test

ruff:
	uv run --group test ruff check jadzia test

ty:
	uv run --group test ty check jadzia

doc:
	uv run --group doc mkdocs serve

clean:
	find jadzia test -type d -name __pycache__ -exec rm -r {} \; || true
	rm -rf jadzia.egg-info htmlcov
