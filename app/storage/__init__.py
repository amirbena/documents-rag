"""Provider-neutral file storage abstraction (Phase 2.6/2.7).

Ingestion, extraction, and upload code depends only on the `FileStorage` contract in
`app.storage.contract` — never on `LocalFileStorage`/`MinioFileStorage` directly, and never on a
filesystem path or a MinIO SDK type. Concrete providers are resolved exclusively through
`app.storage.factory.create_file_storage`, mirroring `app/rag/providers/provider_factory.py`.
"""
