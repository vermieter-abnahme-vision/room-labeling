"""Pydantic DTOs for the HTTP layer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

SHA256_PATTERN = r"^[0-9a-f]{64}$"
LABEL_PATTERN = r"^[a-z0-9_]{1,64}$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LabelOption(BaseModel):
    key: str
    display: str


class ImageDTO(BaseModel):
    sha256: str
    url: str


class SuggestionDTO(BaseModel):
    label: str
    display: str
    confidence: float
    model_version: str
    zero_shot: bool


class ProgressDTO(BaseModel):
    done: int
    total: int


class NextResponse(BaseModel):
    scheme: str
    labels: list[LabelOption]
    image: ImageDTO | None
    suggestion: SuggestionDTO | None
    progress: ProgressDTO


class SubmitRequest(_Strict):
    sha256: str = Field(pattern=SHA256_PATTERN)
    scheme: str = Field(default="room", pattern=LABEL_PATTERN)
    label: str = Field(pattern=LABEL_PATTERN)
    skipped: list[str] = Field(default_factory=list, max_length=200)


class SkipRequest(_Strict):
    scheme: str = Field(default="room", pattern=LABEL_PATTERN)
    skipped: list[str] = Field(default_factory=list, max_length=200)


class SubmitResponse(BaseModel):
    ok: bool = True
    next: NextResponse


class PreviousLabelResponse(BaseModel):
    label: str | None


class ErrorResponse(BaseModel):
    detail: str
