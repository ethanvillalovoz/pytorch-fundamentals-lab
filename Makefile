.PHONY: install notebooks reference test lint check

install:
	python -m pip install -e ".[notebooks,dev]"

notebooks:
	python scripts/build_notebooks.py

reference:
	torch-lab iris --output-dir artifacts/iris-reference

test:
	pytest

lint:
	ruff check .

check: lint test
	python scripts/build_notebooks.py
	git diff --exit-code -- notebooks
