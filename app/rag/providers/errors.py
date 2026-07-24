"""Shared exception types for provider resolution and stub implementations."""

# Category (Phase 2.10, see app/core/errors.py): ConfigurationError — a stub provider being
# configured is a config problem, not a call-time provider failure.


class ProviderNotImplementedError(NotImplementedError):
    """Raised by a provider stub (or the factory) whose implementation doesn't exist yet."""
