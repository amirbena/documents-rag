# Storage

Three storage systems, three distinct roles, never conflated. See
[docs/architecture/](../architecture/README.md) for how this fits the overall system, and
[docs/document-lifecycle/](../document-lifecycle/README.md) for how each is touched across a
document's lifecycle.

## Ownership boundaries

| System | Role | Owning module | Ever the source of truth for lifecycle? |
|---|---|---|---|
| **PostgreSQL** | Lifecycle authority | `app/models/`, `app/services/documents/`, `app/services/indexing/` | **Yes** |
| **Object storage** (local disk or MinIO) | Original-content authority | `app/storage/` | No — only holds bytes |
| **Qdrant** | Rebuildable derived index | `app/rag/providers/qdrant_vector_store.py`, `app/services/indexing/` | **Never** — always rebuildable from Postgres + object storage |

## Relational storage (PostgreSQL)

Async SQLAlchemy + Alembic. Key tables: `documents`, `ingestion_jobs`, `document_deletion_jobs`,
`reindex_jobs`, `vector_cleanup_jobs`, `index_collections`. Every job table is append-only with a
Postgres partial unique index enforcing at most one active row per document — see
[docs/document-lifecycle/](../document-lifecycle/README.md) for the full per-table state machines.

`Document` carries both the storage identity (`storage_provider`/`storage_bucket`/`storage_key`/
`storage_etag`) and the indexing identity (`collection_name`/`embedding_*`/`chunking_version`/
`indexed_at`/`content_hash`) — the row that ties all three storage systems together for one
logical document.

## Object storage

`app/storage/` is the provider-neutral contract every upload/ingestion/extraction/re-index code
path depends on — never a filesystem path or a MinIO SDK type directly.

| Piece | File | Role |
|---|---|---|
| `FileStorage` (contract) | `app/storage/contract.py` | `save`/`read`/`delete`/`exists`/`get_metadata`/`generate_download_url` |
| `LocalFileStorage` | `app/storage/local_storage.py` | Local-disk implementation, keys resolved under `LOCAL_STORAGE_ROOT` |
| `MinioFileStorage` | `app/storage/minio_storage.py` | S3-compatible via the official `minio` SDK (used directly — S3 request signing makes a raw-HTTP reimplementation not worth it, unlike the Ollama/Qdrant providers) |
| `create_file_storage(settings)` | `app/storage/factory.py` | The *only* place a concrete storage class is constructed |
| Object keys | `app/storage/keys.py` | `generate_object_key()`; `resolve_document_storage_key()` (the backward-compat fallback path) |
| Error hierarchy | `app/storage/errors.py` | Every SDK/`urllib3` exception translated to a `StorageError` subclass before leaving the adapter |

**`GET /documents/{id}/download` streams bytes through the application** (`FileStorage.read()`),
never a redirect to a presigned URL — a MinIO endpoint/bucket/credential is never exposed to a
client. `generate_download_url()` exists on the contract but this endpoint deliberately does not
call it.

## Vector storage (Qdrant)

`QdrantVectorStore` (`app/rag/providers/qdrant_vector_store.py`) talks to Qdrant's HTTP API
directly via async `httpx` — no official Qdrant SDK, matching the Ollama providers' pattern.
Collection naming is **versioned and derived, never a literal setting**:
`EmbeddingIndexConfig.collection_name` (`app/rag/embedding_config.py`) combines
`collection_prefix`/`provider`/`model`/`dimension`/`embedding_version`/`chunking_version` — changing
any one field always produces a different collection name, so incompatible vectors can never share
a collection. `QDRANT_COLLECTION_NAME` is a **prefix**, not the literal collection name.

An existing collection with the wrong vector dimension is rejected explicitly
(`IncompatibleIndexConfigurationError`), never silently reused, recreated, or deleted.

## Naming and collection behavior

See [docs/multilingual/](../multilingual/README.md) for why/when a new collection is created
(embedding model/version/chunking changes), and
[docs/document-lifecycle/](../document-lifecycle/README.md) for the build-ahead re-index cycle that
migrates a document from one collection to another without downtime.

## Cleanup implications

A vacated collection's vectors are **never** deleted automatically at activation time — a
`VectorCleanupJob` records the obligation, and a separate, explicit, bounded operational command
executes it later. See [docs/operations/](../operations/README.md) for the command and
[docs/document-lifecycle/README.md#6-vector-cleanup-job-lifecycle](../document-lifecycle/README.md#6-vector-cleanup-job-lifecycle)
for the state machine.

Full document deletion resolves vectors from **three** sources (current collection, historical
cleanup-pending collections, completed-but-not-yet-activated re-index targets), deduplicated, each
attempted independently — see
[docs/document-lifecycle/README.md#3-deletion-job-lifecycle](../document-lifecycle/README.md#3-deletion-job-lifecycle).

## Consistency expectations

- **Postgres and object storage are not one atomic transaction.** Upload: save object → persist
  rows → commit. A commit failure after the object was saved triggers a best-effort delete
  (failure there is logged, never raised) before the original DB exception re-raises — a partially
  completed attempt is not indistinguishable from one that never ran; no orphan-cleanup worker
  exists.
- **Postgres and Qdrant are not one atomic transaction either** — a Qdrant write can succeed while
  the Postgres commit fails. This is documented, never glossed over.
- **Content-hash deduplication is DB-enforced, not merely application-checked**: a normal
  (non-partial) unique index on `content_hash` is the actual race-safety mechanism (Postgres never
  treats two `NULL`s as equal, so unhashed/legacy rows coexist freely); the application-level
  lookup is a fast-path optimization only.

## Test ownership

```bash
make test-storage                # FileStorage contract, Local implementation, factory — no Docker
make test-storage-integration     # real MinIO container (needs Docker)
make test-minio                   # unit + integration MinIO coverage
```

## Current Limitations

- Download and upload both buffer the full object in memory — not streamed. A known, accepted
  limitation, not a new risk introduced by any single feature.
- No AWS S3 or Cloudflare R2-specific implementation — `MinioFileStorage` covers any
  S3-compatible endpoint generically, but there is no dedicated S3/R2 adapter or presigned
  multipart upload support.
- No orphan-object cleanup worker for the upload-then-commit-fails race described above.

## Deferred Behavior

- A dedicated AWS S3 / Cloudflare R2 `FileStorage` implementation and presigned multipart uploads.
- An orphan-object reconciliation/cleanup worker for the storage/Postgres commit-ordering race.
- Content-hash-based storage-drift detection (verifying stored bytes still match the recorded
  hash) — see [docs/document-lifecycle/](../document-lifecycle/README.md)'s Deferred Behavior.
