# One command reproduces everything: `make all` (data download -> report).
# Recipes only call `uv run`, so they behave the same on Linux, macOS and Windows.
# Without make: `uv run conso-forecast all` does the same thing.

.PHONY: help install data download clean-data eda train evaluate analyze report all lint format test check mlflow-ui

help:  ## List the available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install:  ## Create the virtual environment and install pinned dependencies
	uv sync --locked
	uv run pre-commit install

download:  ## Download raw data (skips files already present)
	uv run conso-forecast download

clean-data:  ## Build the clean hourly table (data/processed/hourly.parquet)
	uv run conso-forecast clean

data: download clean-data  ## Download and clean the data

eda:  ## Exploratory figures and summary statistics
	uv run conso-forecast eda

train:  ## Select models on the validation year, refit, save to models/
	uv run conso-forecast train

evaluate:  ## Score all models once on the held-out test year
	uv run conso-forecast evaluate

analyze:  ## Error analysis and feature importance on the test year
	uv run conso-forecast analyze

report:  ## Write reports/report.md and the README results section
	uv run conso-forecast report

all: data eda train evaluate analyze report  ## Full pipeline, from download to report

lint:  ## Static checks (ruff)
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

test:  ## Unit tests
	uv run pytest

check: lint test  ## What CI runs

mlflow-ui:  ## Browse experiment runs at http://127.0.0.1:5000
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
