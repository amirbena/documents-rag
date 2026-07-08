"""Local filesystem-backed storage for uploaded documents.

Saves file content under storage/documents/ using a generated, filesystem-safe stored
filename — never the raw original filename (which may contain Unicode, path separators, or
other unsafe characters). No S3 or other remote backend yet.
"""

import asyncio
import uuid
from pathlib import Path

DEFAULT_STORAGE_ROOT = Path("storage/documents")

_MAX_SAFE_SUFFIX_LENGTH = 10


class LocalFileStorage:
    """Saves uploaded file content to local disk under a generated safe filename."""

    def __init__(self, root: Path | None = None) -> None:
        self._root = root or DEFAULT_STORAGE_ROOT
        self._root.mkdir(parents=True, exist_ok=True)

    async def save(self, content: bytes, original_filename: str) -> tuple[str, str]:
        """Save file content under a generated filename; return (stored_filename, stored_path)."""
        suffix = Path(original_filename).suffix
        safe_suffix = suffix if 1 < len(suffix) <= _MAX_SAFE_SUFFIX_LENGTH and suffix[1:].isalnum() else ""
        stored_filename = f"{uuid.uuid4().hex}{safe_suffix}"
        stored_path = self._root / stored_filename
        await asyncio.to_thread(stored_path.write_bytes, content)
        return stored_filename, str(stored_path)
