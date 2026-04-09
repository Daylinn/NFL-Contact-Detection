.PHONY: install dev test coverage lint clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest

coverage:
	pytest --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

lint-fix:
	ruff check --fix src/ tests/

clean:
	rm -rf .pytest_cache .coverage htmlcov *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
