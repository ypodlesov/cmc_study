PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

.PHONY: help venv install test typecheck lint format figures bench tensor-figures tensor-bench docs clean

help:
	@echo "Targets:"
	@echo "  venv             Create the virtual environment in $(VENV)"
	@echo "  install          Editable install with dev + docs extras"
	@echo "  test             Run the pytest suite"
	@echo "  typecheck        Run mypy --strict on src/"
	@echo "  lint             Run ruff (check)"
	@echo "  format           Run ruff (format)"
	@echo "  figures          Run every experiments/{matrix,tensor}/run_*.py"
	@echo "  bench            Run experiments/matrix/run_matrix_benchmarks.py"
	@echo "  tensor-bench     Run experiments/tensor/run_tensor_benchmarks.py"
	@echo "  docs             Build Sphinx HTML docs"
	@echo "  clean            Remove build artefacts and caches"

venv: $(VENV)/bin/python

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	@echo "Activate with: source $(VENV)/bin/activate"

install: venv
	$(ACTIVATE) && pip install -U pip && pip install -e ".[dev,docs]"

test:
	pytest

typecheck:
	mypy src/

lint:
	ruff check src/ tests/ experiments/

format:
	ruff format src/ tests/ experiments/

figures:
	@for script in experiments/matrix/run_*.py experiments/tensor/run_*.py; do \
	    echo "==> $$script"; \
	    PYTHONPATH=src $(PYTHON) $$script || exit 1; \
	done

bench:
	PYTHONPATH=src $(PYTHON) experiments/matrix/run_matrix_benchmarks.py

tensor-bench:
	PYTHONPATH=src $(PYTHON) experiments/tensor/run_tensor_benchmarks.py

docs:
	sphinx-build -b html docs docs/_build/html

clean:
	rm -rf build dist .pytest_cache .mypy_cache .ruff_cache **/__pycache__ docs/_build
