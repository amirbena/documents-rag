.PHONY: test lint typecheck compose verify

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
