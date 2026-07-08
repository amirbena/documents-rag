"""Shared exception types for provider resolution and stub implementations."""


class ProviderNotImplementedError(NotImplementedError):
    """Raised by a provider stub (or the factory) whose implementation doesn't exist yet."""
