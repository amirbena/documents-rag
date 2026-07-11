.PHONY: help test test-unit test-integration test-e2e-backend test-rag-engines lint typecheck compose verify verify-integration verify-e2e-backend verify-rag-engines

help:
	@echo "Available commands:"
	@echo "  make test              - run the fast suite, excluding integration/e2e/slow (pytest)"
	@echo "  make test-unit         - alias for 'make test'"
	@echo "  make test-integration  - run the Testcontainers-based integration suite (needs Docker)"
	@echo "  make test-e2e-backend  - run the Testcontainers-based backend E2E suite (needs Docker)"
	@echo "  make test-rag-engines  - run only the RAG engine (custom/LangChain) unit, integration,"
	@echo "                          and E2E parity tests (needs Docker for the latter two)"
	@echo "  make lint              - lint the codebase (ruff check .)"
	@echo "  make typecheck         - type-check the app package (mypy app)"
	@echo "  make compose           - validate docker-compose.yml (docker compose config)"
	@echo "  make verify            - run test, lint, typecheck, compose, in order, stopping at"
	@echo "                          the first failure (the canonical pre-commit/pre-PR check;"
	@echo "                          fast, does not require Docker beyond compose-config validation)"
	@echo "  make verify-integration - run the integration suite plus its own checks"
	@echo "  make verify-e2e-backend - run the backend E2E suite plus its own checks"
	@echo "  make verify-rag-engines - run the RAG engine tests plus their own checks"
	@echo "  make help              - show this message"
	@echo ""
	@echo "Install the pre-commit hook that runs 'make verify' automatically:"
	@echo "  ./scripts/install-git-hooks.sh"

test:
	pytest -m "not integration and not e2e and not slow" -q

test-unit: test

test-integration:
	pytest -m integration -q

test-e2e-backend:
	pytest -m e2e tests/e2e/backend -q

test-rag-engines:
	pytest tests/test_rag_engine_factory.py tests/test_custom_rag_engine.py tests/test_langchain_rag_engine.py tests/test_langchain_adapters.py tests/test_rag_responses.py -q
	pytest -m integration tests/integration/test_langchain_rag_engine_integration.py -q
	pytest -m e2e tests/e2e/backend/test_rag_engine_parity.py -q

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

verify-e2e-backend: test-e2e-backend
	@echo "Backend E2E quality gates passed."

verify-rag-engines: test-rag-engines
	@echo "RAG engine compatibility layer quality gates passed."
