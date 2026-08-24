"""Typed Stage 0 output contract and run-result shapes.

Deliberately flat and Stage-0-specific: no typed subject/predicate/object relationships, no
cross-document records, no graph structure — matching analysis/llm-wiki-research.md's Stage 1
target shape exactly. These types are not production domain models and must never be imported by
anything under app/ — Stage 0 exists specifically to answer whether that production work is
justified, not to pre-build it.
"""

from dataclasses import dataclass

from app.rag.language import SupportedLanguage
from app.services.documents.chunker import DocumentChunk


@dataclass(frozen=True)
class GeneratedSummary:
    """A concise, per-document summary grounded in one or more source chunks."""

    text: str
    source_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedEntity:
    """One extracted entity/topic — a name and a one-line description, never a relationship."""

    name: str
    description: str
    source_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedClaim:
    """One grounded natural-language claim — a sentence, never a typed triple."""

    text: str
    source_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class GeneratedKnowledge:
    """The full Stage 0 output for one document: summary + entities + claims, all grounded."""

    document_id: str
    summary: GeneratedSummary
    entities: tuple[GeneratedEntity, ...]
    claims: tuple[GeneratedClaim, ...]


@dataclass(frozen=True)
class Stage0Document:
    """One sample document's chunks, as fed into a single Stage 0 generation call."""

    document_id: str
    chunks: tuple[DocumentChunk, ...]


@dataclass
class Stage0RunResult:
    """One document's Stage 0 run outcome — recorded metrics, never a quality judgment.

    `language_fidelity_ok`/`language_issues` are only meaningful when `parse_succeeded` is True —
    there is no generated prose to check the language of otherwise, so `language_fidelity_ok`
    stays `None` (not `False`) when parsing failed, distinguishing "not checked" from "checked and
    failed".
    """

    document_id: str
    detected_source_language: SupportedLanguage
    generation_succeeded: bool
    parse_succeeded: bool
    language_fidelity_ok: bool | None
    generation_error: str | None
    parse_issues: list[str]
    language_issues: list[str]
    raw_output: str | None
    knowledge: GeneratedKnowledge | None
    duration_seconds: float
