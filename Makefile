.DEFAULT_GOAL := help

.PHONY: help setup run test lint format scan

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-8s %s\n", $$1, $$2}'

setup: ## Install Python and project dependencies into .venv
	uv sync

run: ## Serve the web UI on 127.0.0.1:8000
	uv run uvicorn --factory doppel.app:build --host 127.0.0.1 --port 8000

test: ## Run the test suite
	uv run pytest

lint: ## Lint and check formatting
	uv run ruff check src tests
	uv run ruff format --check src tests

format: ## Auto-format and fix lint issues
	uv run ruff format src tests
	uv run ruff check --fix src tests

scan: ## Sync the Drive photo inventory
	uv run python -m doppel.scan
