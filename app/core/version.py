"""Single source of truth for this service's name/version.

Shared by the FastAPI app's own title/version metadata and by the unversioned platform health
endpoints, so both always report the same value.
"""

SERVICE_NAME = "documents-rag"
SERVICE_VERSION = "0.1.0"
