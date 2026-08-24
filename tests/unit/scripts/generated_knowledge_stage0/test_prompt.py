"""Tests for scripts/generated_knowledge_stage0/prompt.py — prompt content only, deterministic."""

from app.rag.language import SupportedLanguage
from app.services.documents.chunker import DocumentChunk
from scripts.generated_knowledge_stage0.prompt import build_stage0_prompt, get_language_directive


class TestGetLanguageDirective:
    def test_hebrew_directive_names_hebrew_and_forbids_translation(self) -> None:
        directive = get_language_directive(SupportedLanguage.HE)

        assert "Hebrew" in directive
        assert "Do not translate" in directive

    def test_english_directive_names_english(self) -> None:
        directive = get_language_directive(SupportedLanguage.EN)

        assert "English" in directive


class TestBuildStage0Prompt:
    def test_prompt_includes_every_chunk_id_and_text(self) -> None:
        chunks = [
            DocumentChunk(document_id="doc", chunk_id="doc-0", text="Alpha text.", chunk_index=0),
            DocumentChunk(document_id="doc", chunk_id="doc-1", text="Beta text.", chunk_index=1),
        ]

        prompt = build_stage0_prompt(chunks, SupportedLanguage.EN)

        assert "doc-0" in prompt
        assert "Alpha text." in prompt
        assert "doc-1" in prompt
        assert "Beta text." in prompt

    def test_prompt_instructs_against_unsupported_inference(self) -> None:
        prompt = build_stage0_prompt(
            [DocumentChunk(document_id="doc", chunk_id="doc-0", text="x", chunk_index=0)],
            SupportedLanguage.EN,
        )

        assert "never invent" in prompt.lower()
        assert "not actually state" in prompt.lower()

    def test_hebrew_prompt_contains_explicit_hebrew_output_directive(self) -> None:
        prompt = build_stage0_prompt(
            [DocumentChunk(document_id="doc", chunk_id="doc-0", text="שלום", chunk_index=0)],
            SupportedLanguage.HE,
        )

        assert "SOURCE LANGUAGE: Hebrew" in prompt
        assert "Generate ALL summary, entity descriptions, and claims in Hebrew" in prompt

    def test_english_prompt_contains_explicit_english_output_directive(self) -> None:
        prompt = build_stage0_prompt(
            [DocumentChunk(document_id="doc", chunk_id="doc-0", text="hello", chunk_index=0)],
            SupportedLanguage.EN,
        )

        assert "SOURCE LANGUAGE: English" in prompt
        assert "Generate ALL summary, entity descriptions, and claims in English" in prompt

    def test_language_directive_sits_close_to_the_output_format_instruction(self) -> None:
        """The directive must not be buried only in the top-of-prompt rules list — it should sit
        immediately before the JSON output-format instruction, which is what the model reads
        right before it starts generating."""
        prompt = build_stage0_prompt(
            [DocumentChunk(document_id="doc", chunk_id="doc-0", text="שלום", chunk_index=0)],
            SupportedLanguage.HE,
        )

        directive_index = prompt.index("SOURCE LANGUAGE: Hebrew")
        output_format_index = prompt.index("Respond with ONLY a single JSON object")
        chunks_index = prompt.index("Document chunks:")

        assert directive_index < output_format_index < chunks_index

    def test_prompt_explicitly_permits_an_empty_entities_array(self) -> None:
        """Stage 0.2: the prompt must explicitly tell the model an empty entities array is a
        valid, correct answer — never a placeholder with an empty name/description."""
        prompt = build_stage0_prompt(
            [DocumentChunk(document_id="doc", chunk_id="doc-0", text="x", chunk_index=0)],
            SupportedLanguage.EN,
        )

        assert '"entities": []' in prompt
        assert "valid" in prompt.lower() or "optional" in prompt.lower()
        assert "empty name" in prompt.lower()
        assert "empty description" in prompt.lower()

    def test_entities_optional_directive_sits_close_to_the_output_format_instruction(self) -> None:
        """Same placement principle as the language directive — Stage 0.1's real runs showed a
        rule buried only in the top-of-prompt list was not reliably followed."""
        prompt = build_stage0_prompt(
            [DocumentChunk(document_id="doc", chunk_id="doc-0", text="x", chunk_index=0)],
            SupportedLanguage.EN,
        )

        entities_directive_index = prompt.index("ENTITIES ARE OPTIONAL")
        output_format_index = prompt.index("Respond with ONLY a single JSON object")
        chunks_index = prompt.index("Document chunks:")

        assert entities_directive_index < output_format_index < chunks_index
