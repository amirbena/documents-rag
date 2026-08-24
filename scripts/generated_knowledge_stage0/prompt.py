"""Stage 0 extraction prompt: instructions, source-language directive, entity-optionality
directive, and chunk formatting.

No provider/network logic here — see run.py for the actual LLMProvider call. The language
directive is deliberately placed immediately before the output-format instruction (not only in a
general rules list at the top) so it stays close to what the model reads right before generating —
Stage 0's own real-model runs showed a single top-of-prompt sentence was not enough for Hebrew
source content, which this placement exists specifically to address (see
analysis/generated-knowledge-stage0-results.md, "Stage 0.1"). The entities-optionality directive
(Stage 0.2) uses the same placement principle: Stage 0.1's real runs showed a document with only
generic, non-nameable concepts (e.g. "the company", "full-time employees") caused the model to
emit a placeholder entity with an empty name/description rather than either a real name or no
entity at all — an explicit, output-adjacent "an empty array is valid" instruction exists
specifically to give the model permission to do the latter. This is a prompt-side fix only: the
parser in validation.py remains exactly as strict as before and still rejects an empty
name/description outright — this directive changes what the model is told to produce, never what
malformed output the parser is willing to accept.
"""

from collections.abc import Sequence

from app.rag.language import SupportedLanguage
from app.services.documents.chunker import DocumentChunk

_OUTPUT_SCHEMA_EXAMPLE = """{
  "summary": {"text": "...", "source_chunk_ids": ["chunk-1"]},
  "entities": [
    {"name": "...", "description": "...", "source_chunk_ids": ["chunk-1"]}
  ],
  "claims": [
    {"text": "...", "source_chunk_ids": ["chunk-1", "chunk-2"]}
  ]
}"""

_GENERAL_INSTRUCTIONS = (
    "You are extracting a small, strictly source-grounded knowledge layer from a single "
    "document's text chunks. Follow these rules exactly:\n"
    "1. Use ONLY the supplied chunks. Never invent, assume, or infer a fact that the chunks do "
    "not actually state — this includes relationships between entities that merely appear in "
    "the same document. If two things are mentioned but no relationship between them is stated, "
    "do not claim one exists.\n"
    "2. If the source is ambiguous or incomplete about something, do not resolve the ambiguity "
    "yourself — omit the uncertain detail rather than guessing.\n"
    "3. Every summary, entity, and claim must cite the exact chunk_id(s) (from the supplied "
    "chunks below) that support it. Never cite a chunk_id that was not supplied.\n"
    "4. The summary must be concise (2-4 sentences) and must not introduce any fact not present "
    "in the chunks.\n"
    "5. Entities/topics must be meaningful concepts (e.g. a named service, system, policy, or "
    "role) — not every noun in the text. Keep the list small. Entities are OPTIONAL: if this "
    "document contains no meaningful named entity or topic worth extracting, return an empty "
    "entities array. Never invent a generic or placeholder entity, and never emit an entity "
    "object with an empty name or empty description merely to fill the array — an empty array "
    "is a correct, expected answer for some documents.\n"
    "6. Claims are natural-language sentences, never a subject/predicate/object triple and never "
    "a graph structure. A claim may combine information from more than one chunk of this same "
    "document when the chunks together genuinely support it, but never combine chunks into a "
    "claim the chunks do not actually establish together."
)

# Stage 0.1 language directives — deliberately more specific than the generic RAG chat
# response-language directive in app.rag.prompts.catalog (which is written for a short answer to
# a question, not multi-field structured extraction). Explicitly names which fields the directive
# covers and explicitly carves out proper nouns/technical identifiers, since those are expected
# and correct to keep untranslated even in otherwise-Hebrew output.
_LANGUAGE_DIRECTIVES: dict[SupportedLanguage, str] = {
    SupportedLanguage.HE: (
        "SOURCE LANGUAGE: Hebrew\n\n"
        "Generate ALL summary, entity descriptions, and claims in Hebrew. Do not translate the "
        "source into English. Entity names that are proper nouns, API names, product names, "
        "event names, or code identifiers may remain in their original form."
    ),
    SupportedLanguage.EN: (
        "SOURCE LANGUAGE: English\n\nGenerate ALL summary, entity descriptions, and claims in "
        "English."
    ),
}


_ENTITIES_OPTIONAL_DIRECTIVE = (
    "ENTITIES ARE OPTIONAL: if nothing in this document is a meaningful named entity or topic, "
    'respond with "entities": []. An empty array is a valid, correct answer. Never invent a '
    "generic placeholder, and never emit an entity object with an empty name or empty "
    "description merely to satisfy the schema."
)


def get_language_directive(language: SupportedLanguage) -> str:
    """Return the Stage 0 language directive for `language`."""
    return _LANGUAGE_DIRECTIVES[language]


def _format_chunks(chunks: Sequence[DocumentChunk]) -> str:
    return "\n\n".join(f"{chunk.chunk_id}:\n{chunk.text}" for chunk in chunks)


def build_stage0_prompt(chunks: Sequence[DocumentChunk], language: SupportedLanguage) -> str:
    """Build the Stage 0 extraction prompt for one document's chunks in its detected `language`.

    The language directive and the entities-optionality directive both sit immediately before the
    output-format instruction — close to what the model reads right before it starts generating —
    rather than relying solely on the general rules list above, which Stage 0/Stage 0.1's own
    real-model runs showed was not reliably followed on its own for either concern.
    """
    language_directive = get_language_directive(language)
    output_instruction = (
        "Respond with ONLY a single JSON object matching this exact shape — no markdown code "
        f"fences, no commentary before or after it:\n{_OUTPUT_SCHEMA_EXAMPLE}"
    )
    chunk_blocks = _format_chunks(chunks)
    return (
        f"{_GENERAL_INSTRUCTIONS}\n\n"
        f"{language_directive}\n\n"
        f"{_ENTITIES_OPTIONAL_DIRECTIVE}\n\n"
        f"{output_instruction}\n\n"
        f"Document chunks:\n\n{chunk_blocks}"
    )
