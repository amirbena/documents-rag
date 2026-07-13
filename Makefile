.PHONY: help test test-unit test-integration test-e2e-backend test-e2e-backend-minio test-rag-engines test-multilingual-rag test-storage test-storage-integration test-minio test-document-read test-document-read-integration test-ingestion-retry test-ingestion-retry-integration test-document-deletion test-document-deletion-integration lint typecheck compose verify verify-integration verify-e2e-backend verify-e2e-backend-minio verify-rag-engines verify-multilingual-rag verify-storage verify-storage-integration verify-minio verify-document-read verify-document-read-integration verify-ingestion-retry verify-ingestion-retry-integration verify-document-deletion verify-document-deletion-integration smoke-multilingual-real recover-stale-ingestion-jobs process-pending-document-deletions

help:
	@echo "Available commands:"
	@echo "  make test              - run the fast suite, excluding integration/e2e/slow (pytest)"
	@echo "  make test-unit         - alias for 'make test'"
	@echo "  make test-integration  - run the Testcontainers-based integration suite (needs Docker)"
	@echo "  make test-e2e-backend  - run the Testcontainers-based backend E2E suite (needs Docker)"
	@echo "  make test-e2e-backend-minio - run the backend E2E suite's focused MinIO coverage"
	@echo "                          (upload -> real MinIO -> ingestion -> streaming chat, needs Docker)"
	@echo "  make test-rag-engines  - run only the RAG engine (custom/LangChain) unit, integration,"
	@echo "                          and E2E parity tests (needs Docker for the latter two)"
	@echo "  make test-multilingual-rag - run only the Phase 2.5 multilingual RAG unit,"
	@echo "                          integration, and E2E matrix tests (needs Docker for the latter two)"
	@echo "  make test-storage      - run only the Phase 2.6/2.7 storage-abstraction unit tests"
	@echo "                          (FileStorage contract, LocalFileStorage, factory, upload/"
	@echo "                          ingestion wiring) — no Docker required"
	@echo "  make test-storage-integration - run the Testcontainers-based MinIO integration suite"
	@echo "                          (needs Docker)"
	@echo "  make test-minio        - run only the MinIO-specific unit + integration tests"
	@echo "                          (needs Docker for the integration half)"
	@echo "  make test-document-read - run the Phase 2.8.2 document read/download API unit tests"
	@echo "                          (list/detail/ingestion/failure/download, local storage) —"
	@echo "                          no Docker required"
	@echo "  make test-document-read-integration - run the Testcontainers-based Postgres + MinIO"
	@echo "                          coverage for the document read/download APIs (needs Docker)"
	@echo "  make lint              - lint the codebase (ruff check .)"
	@echo "  make typecheck         - type-check the app package (mypy app)"
	@echo "  make compose           - validate docker-compose.yml (docker compose config)"
	@echo "  make verify            - run test, lint, typecheck, compose, in order, stopping at"
	@echo "                          the first failure (the canonical pre-commit/pre-PR check;"
	@echo "                          fast, does not require Docker beyond compose-config validation)"
	@echo "  make verify-integration - run the integration suite plus its own checks"
	@echo "  make verify-e2e-backend - run the backend E2E suite plus its own checks"
	@echo "  make verify-e2e-backend-minio - run the backend E2E suite's MinIO coverage plus its"
	@echo "                          own checks"
	@echo "  make verify-rag-engines - run the RAG engine tests plus their own checks"
	@echo "  make verify-multilingual-rag - run the multilingual RAG tests plus their own checks"
	@echo "  make verify-storage     - run the storage-abstraction unit tests plus their own checks"
	@echo "  make verify-storage-integration - run the MinIO integration suite plus its own checks"
	@echo "  make verify-minio       - run the MinIO-specific tests plus their own checks"
	@echo "  make verify-document-read - run the document read/download unit tests plus their own"
	@echo "                          checks"
	@echo "  make verify-document-read-integration - run the document read/download Postgres +"
	@echo "                          MinIO integration/E2E coverage plus its own checks"
	@echo "  make smoke-multilingual-real - OPTIONAL, MANUAL, non-blocking: exercise the real"
	@echo "                          configured embedding model (default bge-m3) against 5"
	@echo "                          Hebrew/English scenarios. Needs a local Ollama with the"
	@echo "                          model already pulled; never run by make verify/test/CI."
	@echo "  make test-ingestion-retry - run the Phase 2.8.3 retry/stale-recovery unit tests —"
	@echo "                          no Docker required"
	@echo "  make test-ingestion-retry-integration - run the Phase 2.8.3 retry/stale-recovery"
	@echo "                          Postgres integration + Backend E2E coverage (needs Docker)"
	@echo "  make verify-ingestion-retry - run the retry/stale-recovery unit tests plus their own"
	@echo "                          checks"
	@echo "  make verify-ingestion-retry-integration - run the retry/stale-recovery integration"
	@echo "                          coverage plus its own checks"
	@echo "  make recover-stale-ingestion-jobs - OPTIONAL, MANUAL: run one stale-PROCESSING-job"
	@echo "                          recovery batch against the configured database and print a"
	@echo "                          summary; never run by make verify/test/CI."
	@echo "  make test-document-deletion - run the Phase 2.8.4 full-document-deletion unit tests —"
	@echo "                          no Docker required"
	@echo "  make test-document-deletion-integration - run the Phase 2.8.4 deletion Postgres +"
	@echo "                          Qdrant + storage integration and Backend E2E coverage"
	@echo "                          (needs Docker)"
	@echo "  make verify-document-deletion - run the document-deletion unit tests plus their own"
	@echo "                          checks"
	@echo "  make verify-document-deletion-integration - run the document-deletion integration"
	@echo "                          coverage plus its own checks"
	@echo "  make process-pending-document-deletions - OPTIONAL, MANUAL: process pending"
	@echo "                          DocumentDeletionJob rows against the configured database and"
	@echo "                          print a summary; never run by make verify/test/CI."
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

test-e2e-backend-minio:
	pytest -m e2e tests/e2e/backend/test_minio_e2e.py -q

test-rag-engines:
	pytest tests/test_rag_engine_factory.py tests/test_custom_rag_engine.py tests/test_langchain_rag_engine.py tests/test_langchain_adapters.py tests/test_prompt_provider_engine_parity.py -q
	pytest -m integration tests/integration/test_langchain_rag_engine_integration.py -q
	pytest -m e2e tests/e2e/backend/test_rag_engine_parity.py -q

test-multilingual-rag:
	pytest tests/test_embedding_config.py tests/unit/services/indexing/test_collection_registry.py tests/unit/services/indexing/test_vector_deletion_service.py tests/unit/services/indexing/test_cleanup_job_service.py tests/unit/services/indexing/test_reindex_service.py tests/test_language_detector.py tests/test_prompt_catalog.py tests/test_prompt_provider_engine_parity.py -q
	pytest -m integration tests/integration/test_multilingual_indexing.py -q
	pytest -m e2e tests/e2e/backend/test_multilingual_matrix.py -q

test-storage:
	pytest tests/unit/storage/test_storage_contract.py tests/unit/storage/test_local_file_storage.py tests/unit/storage/test_storage_factory.py tests/unit/services/documents/test_text_extractor.py tests/unit/api/test_document_upload.py tests/unit/services/ingestion/test_worker.py -q

test-storage-integration:
	pytest -m integration tests/integration/test_minio_storage.py tests/integration/ingestion/test_worker_minio.py -q

test-minio:
	pytest tests/unit/storage/test_minio_file_storage.py -q
	pytest -m integration tests/integration/test_minio_storage.py tests/integration/ingestion/test_worker_minio.py -q

test-document-read:
	pytest tests/unit/services/documents/test_query_service.py tests/unit/services/documents/test_download_service.py tests/unit/api/test_document_read_routes.py tests/unit/services/documents/test_download_service_local_storage.py -q

test-document-read-integration:
	pytest -m integration tests/integration/documents/read/test_postgres.py tests/integration/documents/download/test_minio.py -q
	pytest -m e2e tests/e2e/backend/documents/read/test_local.py tests/e2e/backend/documents/read/test_minio.py -q

test-ingestion-retry:
	pytest tests/unit/services/ingestion/test_retry_service.py tests/unit/services/ingestion/test_stale_recovery_service.py tests/unit/api/test_ingestion_retry_routes.py -q

test-ingestion-retry-integration:
	pytest -m integration tests/integration/ingestion/test_retry_postgres.py tests/integration/ingestion/test_concurrency.py -q
	pytest -m e2e tests/e2e/backend/ingestion/test_retry_recovery.py -q

test-document-deletion:
	pytest tests/unit/services/documents tests/unit/api/test_document_deletion_routes.py -q

test-document-deletion-integration:
	pytest -m integration tests/integration/documents/deletion -q
	pytest -m e2e tests/e2e/backend/documents/deletion -q

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

verify-e2e-backend-minio: test-e2e-backend-minio
	@echo "Backend E2E MinIO quality gates passed."

verify-rag-engines: test-rag-engines
	@echo "RAG engine compatibility layer quality gates passed."

verify-multilingual-rag: test-multilingual-rag
	@echo "Multilingual RAG quality gates passed."

verify-storage: test-storage
	@echo "Storage abstraction quality gates passed."

verify-storage-integration: test-storage-integration
	@echo "MinIO storage integration quality gates passed."

verify-minio: test-minio
	@echo "MinIO quality gates passed."

verify-document-read: test-document-read
	@echo "Document read/download quality gates passed."

verify-document-read-integration: test-document-read-integration
	@echo "Document read/download integration/E2E quality gates passed."

verify-ingestion-retry: test-ingestion-retry
	@echo "Ingestion retry/stale-recovery quality gates passed."

verify-ingestion-retry-integration: test-ingestion-retry-integration
	@echo "Ingestion retry/stale-recovery integration quality gates passed."

verify-document-deletion: test-document-deletion
	@echo "Document deletion quality gates passed."

verify-document-deletion-integration: test-document-deletion-integration
	@echo "Document deletion integration/E2E quality gates passed."

smoke-multilingual-real:
	python scripts/smoke_multilingual_real.py

recover-stale-ingestion-jobs:
	python scripts/recover_stale_ingestion_jobs.py

process-pending-document-deletions:
	python scripts/process_pending_document_deletions.py
