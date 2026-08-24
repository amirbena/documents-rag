"""Tests for scripts/generated_knowledge_stage0/run.py's execution-flow and failure-isolation
contract — all against fake/mock LLM output, never real Ollama. Real-model execution is an
explicit, separately-run experiment (see analysis/generated-knowledge-stage0-results.md), never a
normal unit-test dependency, matching this repository's existing AI-provider testing policy.
"""

import json

from app.core.config import Settings
from app.rag.language import ScriptBasedLanguageDetector, SupportedLanguage
from app.services.documents.chunker import DocumentChunk
from scripts.generated_knowledge_stage0.models import Stage0Document
from scripts.generated_knowledge_stage0.run import SAMPLE_DOCUMENTS, run_document
from tests.e2e.backend.fakes import FakeFailingLLMProvider, FakeStreamingLLMProvider

_ENGLISH_PAYLOAD = {
    "summary": {
        "text": "Payment Service publishes events consumed by Reconciliation Service.",
        "source_chunk_ids": ["doc-0", "doc-1"],
    },
    "entities": [
        {"name": "Payment Service", "description": "Processes payments.", "source_chunk_ids": ["doc-0"]},
        {
            "name": "Reconciliation Service",
            "description": "Matches invoices.",
            "source_chunk_ids": ["doc-1"],
        },
    ],
    "claims": [
        {
            "text": "Reconciliation Service consumes PaymentCompleted events published by "
            "Payment Service.",
            "source_chunk_ids": ["doc-0", "doc-1"],
        },
    ],
}


def _detector() -> ScriptBasedLanguageDetector:
    return ScriptBasedLanguageDetector(Settings(DEFAULT_RESPONSE_LANGUAGE="en"))


def _english_document() -> Stage0Document:
    return Stage0Document(
        document_id="doc",
        chunks=(
            DocumentChunk(document_id="doc", chunk_id="doc-0", text="Payment text.", chunk_index=0),
            DocumentChunk(document_id="doc", chunk_id="doc-1", text="Reconciliation text.", chunk_index=1),
        ),
    )


class TestRunDocumentFailureIsolation:
    async def test_generation_failure_is_recorded_not_raised(self) -> None:
        document = Stage0Document(
            document_id="doc",
            chunks=(DocumentChunk(document_id="doc", chunk_id="doc-0", text="x", chunk_index=0),),
        )

        result = await run_document(FakeFailingLLMProvider("boom"), _detector(), document)

        assert result.generation_succeeded is False
        assert result.parse_succeeded is False
        assert result.language_fidelity_ok is None
        assert result.generation_error == "boom"
        assert result.knowledge is None

    async def test_parse_failure_is_recorded_not_raised(self) -> None:
        document = Stage0Document(
            document_id="doc",
            chunks=(DocumentChunk(document_id="doc", chunk_id="doc-0", text="x", chunk_index=0),),
        )
        llm = FakeStreamingLLMProvider(chunks=("not valid json",))

        result = await run_document(llm, _detector(), document)

        assert result.generation_succeeded is True
        assert result.parse_succeeded is False
        assert result.language_fidelity_ok is None  # nothing to check when parsing failed
        assert result.parse_issues
        assert result.knowledge is None

    async def test_valid_english_output_for_english_document_produces_populated_result(self) -> None:
        llm = FakeStreamingLLMProvider(chunks=(json.dumps(_ENGLISH_PAYLOAD),))

        result = await run_document(llm, _detector(), _english_document())

        assert result.detected_source_language == SupportedLanguage.EN
        assert result.generation_succeeded is True
        assert result.parse_succeeded is True
        assert result.language_fidelity_ok is True
        assert result.language_issues == []
        assert result.knowledge is not None
        assert result.knowledge.document_id == "doc"
        assert len(result.knowledge.entities) == 2
        assert len(result.knowledge.claims) == 1

    async def test_unknown_chunk_id_from_real_document_scope_is_rejected(self) -> None:
        """A model citing a chunk id that isn't one of this specific document's chunks must be
        caught even when generation itself succeeds cleanly."""
        document = Stage0Document(
            document_id="doc",
            chunks=(DocumentChunk(document_id="doc", chunk_id="doc-0", text="a", chunk_index=0),),
        )
        payload = json.loads(json.dumps(_ENGLISH_PAYLOAD))
        payload["summary"]["source_chunk_ids"] = ["doc-0"]
        payload["entities"] = []
        payload["claims"][0]["source_chunk_ids"] = ["doc-1"]  # not part of this one-chunk document
        llm = FakeStreamingLLMProvider(chunks=(json.dumps(payload),))

        result = await run_document(llm, _detector(), document)

        assert result.parse_succeeded is False
        assert any("unknown chunk_id" in issue for issue in result.parse_issues)

    async def test_valid_output_that_fails_language_fidelity_is_flagged_not_silently_accepted(
        self,
    ) -> None:
        """Exercises the run_document() -> language-fidelity path end to end: structurally valid,
        well-grounded English output returned for a Hebrew document must be flagged, not accepted
        as a successful run — this is the exact Stage 0 failure mode this pipeline exists to catch."""
        document = Stage0Document(
            document_id="doc",
            chunks=(
                DocumentChunk(document_id="doc", chunk_id="doc-0", text="שלום עולם", chunk_index=0),
            ),
        )
        payload = {
            "summary": {
                "text": "This is an English summary of a Hebrew document.",
                "source_chunk_ids": ["doc-0"],
            },
            "entities": [],
            "claims": [],
        }
        llm = FakeStreamingLLMProvider(chunks=(json.dumps(payload),))

        result = await run_document(llm, _detector(), document)

        assert result.detected_source_language == SupportedLanguage.HE
        assert result.parse_succeeded is True
        assert result.language_fidelity_ok is False
        assert result.language_issues


class TestSampleDocumentsCoverNegativeRelationshipCases:
    """Deterministic coverage of the adversarial sample documents' shape through the full
    run_document() pipeline — actual hallucination-avoidance is a real-model concern (see
    analysis/generated-knowledge-stage0-results.md) and cannot be asserted deterministically, but
    the documents themselves and the pipeline that processes them are exercised here."""

    def test_sample_documents_include_both_adversarial_relationship_cases(self) -> None:
        document_ids = {document.document_id for document in SAMPLE_DOCUMENTS}

        assert "en-unrelated-services" in document_ids
        assert "en-cooccurring-teams" in document_ids

    async def test_well_grounded_response_for_unrelated_services_document_parses_cleanly(self) -> None:
        """A well-behaved (non-hallucinating) response to the unrelated-services adversarial
        document — two independent claims, no fabricated relationship — must parse and validate
        successfully through the full pipeline."""
        document = next(d for d in SAMPLE_DOCUMENTS if d.document_id == "en-unrelated-services")
        chunk_ids = [chunk.chunk_id for chunk in document.chunks]
        payload = {
            "summary": {
                "text": "Two services are described, each using different technology.",
                "source_chunk_ids": chunk_ids,
            },
            "entities": [
                {
                    "name": "Service A",
                    "description": "A Java-based service using PostgreSQL.",
                    "source_chunk_ids": [chunk_ids[0]],
                },
                {
                    "name": "Service B",
                    "description": "A Node.js service using Kafka.",
                    "source_chunk_ids": [chunk_ids[1]],
                },
            ],
            "claims": [
                {
                    "text": "Service A persists customer records in PostgreSQL.",
                    "source_chunk_ids": [chunk_ids[0]],
                },
                {
                    "text": "Service B publishes analytics events to Kafka.",
                    "source_chunk_ids": [chunk_ids[1]],
                },
            ],
        }
        llm = FakeStreamingLLMProvider(chunks=(json.dumps(payload),))

        result = await run_document(llm, _detector(), document)

        assert result.parse_succeeded is True
        assert result.language_fidelity_ok is True
        assert result.knowledge is not None
        assert len(result.knowledge.claims) == 2
