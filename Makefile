.PHONY: help test test-unit test-integration lint typecheck compose verify verify-integration

help:
	@echo "Available commands:"
	@echo "  make test              - run the fast suite, excluding integration/e2e/slow (pytest)"
	@echo "  make test-unit         - alias for 'make test'"
	@echo "  make test-integration  - run the Testcontainers-based integration suite (needs Docker)"
	@echo "  make lint              - lint the codebase (ruff check .)"
	@echo "  make typecheck         - type-check the app package (mypy app)"
	@echo "  make compose           - validate docker-compose.yml (docker compose config)"
	@echo "  make verify            - run test, lint, typecheck, compose, in order, stopping at"
	@echo "                          the first failure (the canonical pre-commit/pre-PR check;"
	@echo "                          fast, does not require Docker beyond compose-config validation)"
	@echo "  make verify-integration - run the integration suite plus its own checks"
	@echo "  make help              - show this message"
	@echo ""
	@echo "Install the pre-commit hook that runs 'make verify' automatically:"
	@echo "  ./scripts/install-git-hooks.sh"

test:
	pytest -m "not integration and not e2e and not slow" -q

test-unit: test

test-integration:
	pytest -m integration -q

lint:
	ruff check .

typecheck:
	mypy app

compose:
	docker compose config

verify: test lint typecheck compose
	@echo "All quality gates passed."

verify-integration: test-integration
	@echo "Integration quality gates passed."
