"""Request schema for the streaming chat endpoint."""

from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    """Shape accepted by POST /api/v1/chat."""

    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        """Reject an empty or whitespace-only question."""
        if not value.strip():
            raise ValueError("question must not be empty")
        return value
