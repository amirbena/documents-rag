"""Deterministic parsing, provenance validation, and language-fidelity validation.

Independent of real Ollama behavior — every function here operates on already-produced text
(model output or generated prose) and a supplied language detector; nothing here makes a network
call or depends on which provider produced the text.
"""

import json
import re

from app.rag.language import LanguageDetector, SupportedLanguage

from .models import GeneratedClaim, GeneratedEntity, GeneratedKnowledge, GeneratedSummary


class Stage0ValidationError(ValueError):
    """Raised when model output is malformed, ungrounded, or otherwise violates the contract.

    Carries every problem found, not just the first — a spike is only useful if a failure run
    tells you everything that went wrong, not one issue at a time across repeated invocations.
    """

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__("; ".join(issues))


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.DOTALL)


def _extract_json_object(raw_output: str) -> str:
    """Return the JSON object substring, tolerating a markdown code fence around it.

    Some local models wrap otherwise-valid JSON in ```json ... ``` despite being told not to;
    this strips that one specific, commonly-observed deviation without attempting to repair
    any other malformed structure.
    """
    fence_match = _JSON_FENCE_PATTERN.search(raw_output)
    if fence_match:
        return fence_match.group(1)
    return raw_output.strip()


def parse_and_validate(
    raw_output: str, *, document_id: str, allowed_chunk_ids: set[str]
) -> GeneratedKnowledge:
    """Parse raw model output into GeneratedKnowledge, enforcing the full Stage 0 output contract.

    Never silently accepts malformed JSON, a missing field, empty text, a chunk_id outside
    `allowed_chunk_ids` (which is scoped to exactly this document's own chunks — so this also
    rejects any cross-document reference), or a record with no provenance at all. Duplicate
    chunk_ids within one record's source_chunk_ids are normalized (deduplicated, order-preserved)
    rather than rejected — a model citing the same chunk twice is not a contract violation, just
    redundant. Raises Stage0ValidationError carrying every problem found, not just the first.

    Language fidelity is a separate concern — see check_language_fidelity() below — since it
    applies only to already-schema-valid generated prose, not to output structure.
    """
    issues: list[str] = []

    try:
        payload = json.loads(_extract_json_object(raw_output))
    except json.JSONDecodeError as exc:
        raise Stage0ValidationError([f"output is not valid JSON: {exc}"]) from exc

    if not isinstance(payload, dict):
        raise Stage0ValidationError(["top-level output must be a JSON object"])

    summary = _validate_record_dict(
        payload.get("summary"), "summary", allowed_chunk_ids, issues, text_field="text"
    )
    entities_raw = payload.get("entities")
    claims_raw = payload.get("claims")

    if not isinstance(entities_raw, list):
        issues.append("'entities' must be a JSON array")
        entities_raw = []
    if not isinstance(claims_raw, list):
        issues.append("'claims' must be a JSON array")
        claims_raw = []

    entities: list[GeneratedEntity] = []
    for index, raw_entity in enumerate(entities_raw):
        validated_entity = _validate_entity_dict(raw_entity, index, allowed_chunk_ids, issues)
        if validated_entity is not None:
            entities.append(validated_entity)

    claims: list[GeneratedClaim] = []
    for index, raw_claim in enumerate(claims_raw):
        validated_claim = _validate_record_dict(
            raw_claim, f"claims[{index}]", allowed_chunk_ids, issues, text_field="text"
        )
        if validated_claim is not None:
            claims.append(GeneratedClaim(text=validated_claim[0], source_chunk_ids=validated_claim[1]))

    if issues:
        raise Stage0ValidationError(issues)

    assert summary is not None  # no issues means every field validated successfully
    return GeneratedKnowledge(
        document_id=document_id,
        summary=GeneratedSummary(text=summary[0], source_chunk_ids=summary[1]),
        entities=tuple(entities),
        claims=tuple(claims),
    )


def _normalize_chunk_ids(
    raw_ids: object, label: str, allowed_chunk_ids: set[str], issues: list[str]
) -> tuple[str, ...] | None:
    if not isinstance(raw_ids, list) or not raw_ids:
        issues.append(f"{label}: source_chunk_ids must be a non-empty array")
        return None

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_id in raw_ids:
        if not isinstance(raw_id, str) or not raw_id.strip():
            issues.append(f"{label}: source_chunk_ids must contain only non-empty strings")
            return None
        if raw_id not in allowed_chunk_ids:
            issues.append(f"{label}: cites unknown chunk_id {raw_id!r} not present in this document")
            return None
        if raw_id not in seen:
            seen.add(raw_id)
            normalized.append(raw_id)

    return tuple(normalized)


def _validate_record_dict(
    raw: object,
    label: str,
    allowed_chunk_ids: set[str],
    issues: list[str],
    *,
    text_field: str,
) -> tuple[str, tuple[str, ...]] | None:
    if not isinstance(raw, dict):
        issues.append(f"{label}: must be a JSON object")
        return None

    text = raw.get(text_field)
    if not isinstance(text, str) or not text.strip():
        issues.append(f"{label}: {text_field!r} must be a non-empty string")
        text = None

    chunk_ids = _normalize_chunk_ids(raw.get("source_chunk_ids"), label, allowed_chunk_ids, issues)

    if text is None or chunk_ids is None:
        return None
    return text, chunk_ids


def _validate_entity_dict(
    raw: object, index: int, allowed_chunk_ids: set[str], issues: list[str]
) -> GeneratedEntity | None:
    label = f"entities[{index}]"
    if not isinstance(raw, dict):
        issues.append(f"{label}: must be a JSON object")
        return None

    name = raw.get("name")
    description = raw.get("description")
    if not isinstance(name, str) or not name.strip():
        issues.append(f"{label}: 'name' must be a non-empty string")
        name = None
    if not isinstance(description, str) or not description.strip():
        issues.append(f"{label}: 'description' must be a non-empty string")
        description = None

    chunk_ids = _normalize_chunk_ids(raw.get("source_chunk_ids"), label, allowed_chunk_ids, issues)

    if name is None or description is None or chunk_ids is None:
        return None
    return GeneratedEntity(name=name, description=description, source_chunk_ids=chunk_ids)


# --------------------------------------------------------------------------------------------
# Language fidelity
# --------------------------------------------------------------------------------------------
# Reuses the existing ScriptBasedLanguageDetector algorithm exactly (via the LanguageDetector
# contract, never a bespoke classifier) — its word-level, majority-count classification is
# already robust to a handful of embedded Latin-script technical identifiers (an entity name, an
# API/event name) not outweighing surrounding Hebrew, which is exactly the property this check
# needs: catching an entirely-English response to a Hebrew document, without penalizing correct,
# expected untranslated proper nouns inside otherwise-Hebrew prose.
#
# Only summary text, entity *descriptions*, and claim text are checked — never entity *names*,
# which the Stage 0.1 prompt explicitly permits to remain in their original form (a product name,
# an API name, a code identifier is not a language-fidelity violation).


def check_language_fidelity(
    knowledge: GeneratedKnowledge,
    expected_language: SupportedLanguage,
    detector: LanguageDetector,
) -> list[str]:
    """Return a list of language-fidelity issues (empty if every checked field matches).

    A field's detected language is compared against `expected_language`; `LanguageDetector`'s
    "no words of either script" fallback (e.g. a summary consisting only of a proper noun/digits)
    is treated as inconclusive, not a violation — checking summary/description/claim text, which
    is normally full prose, real content should almost always classify decisively.
    """
    issues: list[str] = []

    if detector.detect(knowledge.summary.text) != expected_language:
        issues.append(
            f"summary text does not match expected language {expected_language.value!r}: "
            f"{knowledge.summary.text!r}"
        )

    for index, entity in enumerate(knowledge.entities):
        if detector.detect(entity.description) != expected_language:
            issues.append(
                f"entities[{index}] description does not match expected language "
                f"{expected_language.value!r}: {entity.description!r}"
            )

    for index, claim in enumerate(knowledge.claims):
        if detector.detect(claim.text) != expected_language:
            issues.append(
                f"claims[{index}] text does not match expected language "
                f"{expected_language.value!r}: {claim.text!r}"
            )

    return issues
