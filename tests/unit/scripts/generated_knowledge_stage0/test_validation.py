"""Tests for scripts/generated_knowledge_stage0/validation.py — schema, provenance, and
language-fidelity validation, all deterministic and independent of any real LLM."""

import json

import pytest

from app.core.config import Settings
from app.rag.language import ScriptBasedLanguageDetector, SupportedLanguage
from scripts.generated_knowledge_stage0.models import (
    GeneratedClaim,
    GeneratedEntity,
    GeneratedKnowledge,
    GeneratedSummary,
)
from scripts.generated_knowledge_stage0.validation import (
    Stage0ValidationError,
    check_language_fidelity,
    parse_and_validate,
)

_ALLOWED = {"doc-0", "doc-1"}


def _detector() -> ScriptBasedLanguageDetector:
    return ScriptBasedLanguageDetector(Settings(DEFAULT_RESPONSE_LANGUAGE="en"))


def _valid_payload() -> dict:
    return {
        "summary": {
            "text": "Payment Service publishes events consumed by Reconciliation Service.",
            "source_chunk_ids": ["doc-0", "doc-1"],
        },
        "entities": [
            {
                "name": "Payment Service",
                "description": "Processes payments.",
                "source_chunk_ids": ["doc-0"],
            },
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


class TestParseAndValidateHappyPath:
    def test_valid_json_parses_into_generated_knowledge(self) -> None:
        raw = json.dumps(_valid_payload())

        result = parse_and_validate(raw, document_id="doc", allowed_chunk_ids=_ALLOWED)

        assert isinstance(result, GeneratedKnowledge)
        assert result.document_id == "doc"
        assert result.summary == GeneratedSummary(
            text="Payment Service publishes events consumed by Reconciliation Service.",
            source_chunk_ids=("doc-0", "doc-1"),
        )
        assert len(result.entities) == 2
        assert result.entities[0] == GeneratedEntity(
            name="Payment Service", description="Processes payments.", source_chunk_ids=("doc-0",)
        )
        assert len(result.claims) == 1
        assert result.claims[0].source_chunk_ids == ("doc-0", "doc-1")

    def test_tolerates_markdown_json_code_fence(self) -> None:
        raw = f"```json\n{json.dumps(_valid_payload())}\n```"

        result = parse_and_validate(raw, document_id="doc", allowed_chunk_ids=_ALLOWED)

        assert result.summary.text.startswith("Payment Service")

    def test_duplicate_chunk_ids_are_normalized_not_rejected(self) -> None:
        payload = _valid_payload()
        payload["summary"]["source_chunk_ids"] = ["doc-0", "doc-0", "doc-1"]

        result = parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

        assert result.summary.source_chunk_ids == ("doc-0", "doc-1")


class TestParseAndValidateMalformedOutput:
    def test_invalid_json_is_rejected_with_clear_error(self) -> None:
        with pytest.raises(Stage0ValidationError, match="not valid JSON"):
            parse_and_validate("not json at all {{{", document_id="doc", allowed_chunk_ids=_ALLOWED)

    def test_non_object_top_level_is_rejected(self) -> None:
        with pytest.raises(Stage0ValidationError, match="must be a JSON object"):
            parse_and_validate("[1, 2, 3]", document_id="doc", allowed_chunk_ids=_ALLOWED)

    def test_missing_summary_field_is_rejected(self) -> None:
        payload = _valid_payload()
        del payload["summary"]

        with pytest.raises(Stage0ValidationError):
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

    def test_entities_not_a_list_is_rejected(self) -> None:
        payload = _valid_payload()
        payload["entities"] = "Payment Service"

        with pytest.raises(Stage0ValidationError, match="'entities' must be a JSON array"):
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)


class TestParseAndValidateProvenance:
    def test_unknown_chunk_id_is_rejected(self) -> None:
        payload = _valid_payload()
        payload["summary"]["source_chunk_ids"] = ["doc-99"]

        with pytest.raises(Stage0ValidationError, match="unknown chunk_id"):
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

    def test_chunk_id_from_another_document_is_rejected(self) -> None:
        """allowed_chunk_ids is scoped to one document — a foreign chunk id is indistinguishable
        from an unknown one, which is exactly how cross-document references get rejected."""
        payload = _valid_payload()
        payload["claims"][0]["source_chunk_ids"] = ["other-document-chunk-3"]

        with pytest.raises(Stage0ValidationError, match="unknown chunk_id"):
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

    def test_missing_provenance_is_rejected(self) -> None:
        payload = _valid_payload()
        payload["summary"]["source_chunk_ids"] = []

        with pytest.raises(Stage0ValidationError, match="non-empty array"):
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

    def test_missing_source_chunk_ids_field_is_rejected(self) -> None:
        payload = _valid_payload()
        del payload["claims"][0]["source_chunk_ids"]

        with pytest.raises(Stage0ValidationError, match="non-empty array"):
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)


class TestParseAndValidateEmptyRecords:
    def test_empty_summary_text_is_rejected(self) -> None:
        payload = _valid_payload()
        payload["summary"]["text"] = "   "

        with pytest.raises(Stage0ValidationError, match="non-empty string"):
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

    def test_empty_claim_text_is_rejected(self) -> None:
        payload = _valid_payload()
        payload["claims"][0]["text"] = ""

        with pytest.raises(Stage0ValidationError, match="non-empty string"):
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

    def test_entity_missing_description_is_rejected(self) -> None:
        payload = _valid_payload()
        del payload["entities"][0]["description"]

        with pytest.raises(Stage0ValidationError, match="'description' must be a non-empty string"):
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

    def test_empty_entities_and_claims_lists_parse_successfully(self) -> None:
        """A document with no extractable entities/claims is valid — the model just didn't find
        any, which is different from a malformed record within a non-empty list."""
        payload = _valid_payload()
        payload["entities"] = []
        payload["claims"] = []

        result = parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

        assert result.entities == ()
        assert result.claims == ()

    def test_empty_entities_array_with_populated_claims_parses_successfully(self) -> None:
        """The correct Stage 0.2 response for a document with no nameable entity: an empty
        entities array alongside otherwise-normal summary/claims — must never be rejected."""
        payload = _valid_payload()
        payload["entities"] = []

        result = parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

        assert result.entities == ()
        assert len(result.claims) == 1

    def test_placeholder_entity_with_empty_name_is_still_rejected(self) -> None:
        """Stage 0.1's real runs produced exactly this shape (an entity with an empty "name" but
        a real description) for he-vacation-policy — the prompt fix must not be paired with any
        parser weakening that would let this through."""
        payload = _valid_payload()
        payload["entities"] = [
            {"name": "", "description": "מדיניות החופשה של החברה", "source_chunk_ids": ["doc-0"]}
        ]

        with pytest.raises(Stage0ValidationError, match="'name' must be a non-empty string"):
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

    def test_placeholder_entity_with_empty_name_and_description_is_still_rejected(self) -> None:
        """The second observed Stage 0.1 shape: both name and description empty."""
        payload = _valid_payload()
        payload["entities"] = [{"name": "", "description": "", "source_chunk_ids": ["doc-0"]}]

        with pytest.raises(Stage0ValidationError) as exc_info:
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

        assert any("'name' must be a non-empty string" in issue for issue in exc_info.value.issues)
        assert any(
            "'description' must be a non-empty string" in issue for issue in exc_info.value.issues
        )

    def test_whitespace_only_placeholder_name_is_still_rejected(self) -> None:
        """A name of only whitespace is exactly as invalid as a truly empty string — the parser's
        existing strip()-based emptiness check must keep catching this variant too."""
        payload = _valid_payload()
        payload["entities"] = [
            {"name": "   ", "description": "A real description.", "source_chunk_ids": ["doc-0"]}
        ]

        with pytest.raises(Stage0ValidationError, match="'name' must be a non-empty string"):
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

    def test_all_issues_are_reported_together_not_just_the_first(self) -> None:
        payload = _valid_payload()
        payload["summary"]["text"] = ""
        payload["claims"][0]["text"] = ""

        with pytest.raises(Stage0ValidationError) as exc_info:
            parse_and_validate(json.dumps(payload), document_id="doc", allowed_chunk_ids=_ALLOWED)

        assert len(exc_info.value.issues) >= 2


class TestCheckLanguageFidelity:
    def test_hebrew_prose_for_hebrew_document_passes(self) -> None:
        knowledge = GeneratedKnowledge(
            document_id="doc",
            summary=GeneratedSummary(
                text="שירות ה-Payment מפרסם אירוע לאחר עיבוד תשלום.", source_chunk_ids=("doc-0",)
            ),
            entities=(
                GeneratedEntity(
                    name="Payment Service",
                    description="שירות האחראי על עיבוד תשלומי לקוחות.",
                    source_chunk_ids=("doc-0",),
                ),
            ),
            claims=(
                GeneratedClaim(
                    text="שירות ה-Reconciliation צורך אירועי PaymentCompleted שפורסמו "
                    "על ידי שירות ה-Payment.",
                    source_chunk_ids=("doc-0",),
                ),
            ),
        )

        issues = check_language_fidelity(knowledge, SupportedLanguage.HE, _detector())

        assert issues == []

    def test_entirely_english_prose_for_hebrew_document_fails(self) -> None:
        """This is the exact Stage 0 failure this check exists to catch: a fluent, well-formed,
        but entirely English response to a Hebrew source document."""
        knowledge = GeneratedKnowledge(
            document_id="doc",
            summary=GeneratedSummary(
                text="Payment Service publishes an event after processing a payment.",
                source_chunk_ids=("doc-0",),
            ),
            entities=(
                GeneratedEntity(
                    name="Payment Service",
                    description="A service responsible for processing customer payments.",
                    source_chunk_ids=("doc-0",),
                ),
            ),
            claims=(
                GeneratedClaim(
                    text="Reconciliation Service consumes PaymentCompleted events published by "
                    "Payment Service.",
                    source_chunk_ids=("doc-0",),
                ),
            ),
        )

        issues = check_language_fidelity(knowledge, SupportedLanguage.HE, _detector())

        assert len(issues) == 3  # summary + entity description + claim all flagged
        assert any("summary" in issue for issue in issues)
        assert any("entities[0]" in issue for issue in issues)
        assert any("claims[0]" in issue for issue in issues)

    def test_english_prose_for_english_document_passes(self) -> None:
        knowledge = GeneratedKnowledge(
            document_id="doc",
            summary=GeneratedSummary(
                text="Payment Service publishes an event after processing a payment.",
                source_chunk_ids=("doc-0",),
            ),
            entities=(
                GeneratedEntity(
                    name="Payment Service",
                    description="A service responsible for processing customer payments.",
                    source_chunk_ids=("doc-0",),
                ),
            ),
            claims=(
                GeneratedClaim(
                    text="Reconciliation Service consumes PaymentCompleted events published by "
                    "Payment Service.",
                    source_chunk_ids=("doc-0",),
                ),
            ),
        )

        issues = check_language_fidelity(knowledge, SupportedLanguage.EN, _detector())

        assert issues == []

    def test_technical_identifiers_embedded_in_hebrew_prose_do_not_incorrectly_fail(self) -> None:
        """Proper nouns, API/event/class names, and English acronyms embedded inside otherwise-
        Hebrew prose must not flip the detected language — this mirrors exactly the robustness
        property ScriptBasedLanguageDetector already provides for RAG chat questions."""
        knowledge = GeneratedKnowledge(
            document_id="doc",
            summary=GeneratedSummary(
                text=(
                    "שירות ה-Payment Service מפרסם אירוע PaymentCompleted דרך Kafka לאחר עיבוד "
                    "תשלום באמצעות ה-API הפנימי."
                ),
                source_chunk_ids=("doc-0",),
            ),
            entities=(
                GeneratedEntity(
                    name="Payment Service",
                    description="שירות שאחראי על עיבוד תשלומי לקוחות ומפרסם אירועי PaymentCompleted.",
                    source_chunk_ids=("doc-0",),
                ),
            ),
            claims=(
                GeneratedClaim(
                    text=(
                        "שירות ה-Reconciliation Service צורך אירועי PaymentCompleted מתוך "
                        "ה-Kafka topic ומשווה אותם מול חשבוניות באמצעות ה-API."
                    ),
                    source_chunk_ids=("doc-0",),
                ),
            ),
        )

        issues = check_language_fidelity(knowledge, SupportedLanguage.HE, _detector())

        assert issues == []

    def test_entity_name_is_never_checked_for_language(self) -> None:
        """An entity's name field is explicitly permitted to remain untranslated (a product/API
        name) — only its description is checked, so an entirely-Latin name must never appear in
        the issues list even for a Hebrew document."""
        knowledge = GeneratedKnowledge(
            document_id="doc",
            summary=GeneratedSummary(
                text="שירות ה-Payment Service מפרסם אירוע לאחר עיבוד תשלום.",
                source_chunk_ids=("doc-0",),
            ),
            entities=(
                GeneratedEntity(
                    name="PaymentCompleted",  # entirely Latin — must not be checked
                    description="אירוע שמפורסם לאחר עיבוד תשלום בהצלחה.",
                    source_chunk_ids=("doc-0",),
                ),
            ),
            claims=(),
        )

        issues = check_language_fidelity(knowledge, SupportedLanguage.HE, _detector())

        assert issues == []
