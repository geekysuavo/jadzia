.PHONY: all clean lint test ruff ty

all: lint test

lint: ruff ty

test:
	uv run --group test pytest -v --cov=jadzia --cov-report term:skip-covered test

ruff:
	uv run --group test ruff check jadzia test

ty:
	uv run --group test ty check jadzia

clean:
	rm -rf jadzia.egg-info
