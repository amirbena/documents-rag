"""Shared, fixed RAG response/prompt text — framework-neutral, no engine-specific logic.

Every RagEngine implementation (CustomRagEngine via RagOrchestrator, LangChainRagEngine) must
emit byte-identical text for the CLARIFICATION_NEEDED/OUT_OF_SCOPE decisions and the same
direct-LLM system instruction. This module is the single source of truth for that text, so no
engine implementation imports it from another engine's implementation module. Contains no FastAPI
dependency, no LangChain dependency, no provider-client construction, and no orchestration logic —
just the fixed strings themselves.

This is a temporary compatibility boundary, not the final design. Phase 2.5 will move fixed
response resolution into a shared, language-aware PromptProvider/PromptCatalog (multilingual
catalogs, language detection, per-request response selection). Until that milestone, this module
intentionally stays a flat set of English-only constants — do not add language variants,
detection, or catalog abstractions here ahead of that work.
"""

CLARIFICATION_NEEDED_RESPONSE = (
    "Could you rephrase or add more detail? Your question is too short or unclear to act on."
)

OUT_OF_SCOPE_RESPONSE = "I can't help with that request."

DIRECT_LLM_SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question directly."
