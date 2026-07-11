"""Shared, language-aware prompt resolution: PromptType, ResolvedPrompt, PromptCatalog, PromptProvider.

Both CustomRagEngine and LangChainRagEngine consume this package's PromptProvider — neither owns
a private prompt catalog or performs its own language detection.
"""
