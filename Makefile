.PHONY: help test lint typecheck compose verify

help:
	@echo "Available commands:"
	@echo "  make test        - run the test suite (pytest -q)"
	@echo "  make lint        - lint the codebase (ruff check .)"
	@echo "  make typecheck   - type-check the app package (mypy app)"
	@echo "  make compose     - validate docker-compose.yml (docker compose config)"
	@echo "  make verify      - run all of the above, in order, stopping at the first failure"
	@echo "                     (the canonical pre-commit/pre-PR check)"
	@echo "  make help        - show this message"
	@echo ""
	@echo "Install the pre-commit hook that runs 'make verify' automatically:"
	@echo "  ./scripts/install-git-hooks.sh"

test:
	pytest -q

lint:
	ruff check .

typecheck:
	mypy app

compose:
	docker compose config

verify: test lint typecheck compose
	@echo "All quality gates passed."
