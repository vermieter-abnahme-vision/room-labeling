"""SQLAlchemy 2.x typed ORM models.

Annotations are generic over ``scheme`` (default "room") so multi-label element schemes
(window, radiator, ...) need no schema change.
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any, ClassVar

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DEFAULT_SCHEME = "room"
LIBRARY_LABELER_SUB = "system:library"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map: ClassVar[dict[Any, Any]] = {dict[str, Any]: JSON}


def _enum(enum_cls: type[enum.Enum], name: str) -> Enum:
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        values_callable=lambda cls: [member.value for member in cls],
        length=32,
    )


class ImageSource(enum.StrEnum):
    UPLOAD_VIDEO = "upload_video"
    FOLDER = "folder"
    LIBRARY = "library"


class ConsensusStatus(enum.StrEnum):
    OPEN = "open"  # fewer than two labels
    NEEDS_MORE = "needs_more"  # two labels disagree, third required
    CONFLICT = "conflict"  # no majority -> admin review
    FINAL = "final"


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    sha256: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    path: Mapped[str] = mapped_column(String(512), nullable=False)  # relative to IMAGES_DIR
    source: Mapped[ImageSource] = mapped_column(_enum(ImageSource, "image_source"), nullable=False)
    group_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Labeler(Base):
    __tablename__ = "labelers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    oidc_sub: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    is_admin: Mapped[bool] = mapped_column(default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    @property
    def is_library(self) -> bool:
        return self.oidc_sub == LIBRARY_LABELER_SUB


class Annotation(Base):
    __tablename__ = "annotations"
    __table_args__ = (
        UniqueConstraint("image_id", "labeler_id", "scheme", "label", name="uq_annotation"),
        Index("ix_annotations_labeler_scheme", "labeler_id", "scheme"),
        Index("ix_annotations_image_scheme", "image_id", "scheme"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"))
    labeler_id: Mapped[int] = mapped_column(ForeignKey("labelers.id", ondelete="CASCADE"))
    scheme: Mapped[str] = mapped_column(String(64), nullable=False, default=DEFAULT_SCHEME)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Consensus(Base):
    __tablename__ = "consensus"
    __table_args__ = (Index("ix_consensus_scheme_status", "scheme", "status"),)

    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id", ondelete="CASCADE"), primary_key=True
    )
    scheme: Mapped[str] = mapped_column(String(64), primary_key=True, default=DEFAULT_SCHEME)
    final_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[ConsensusStatus] = mapped_column(
        _enum(ConsensusStatus, "consensus_status"), nullable=False, default=ConsensusStatus.OPEN
    )
    agreement: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Set when an admin manually decided; recomputation then keeps the admin's label.
    resolved_by_id: Mapped[int | None] = mapped_column(
        ForeignKey("labelers.id", ondelete="SET NULL"), nullable=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    __table_args__ = (
        UniqueConstraint("image_id", "scheme", "model_version", name="uq_prediction"),
        Index("ix_predictions_scheme_version", "scheme", "model_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"))
    scheme: Mapped[str] = mapped_column(String(64), nullable=False, default=DEFAULT_SCHEME)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    # top-1 minus top-2 probability; used by margin sampling.
    margin: Mapped[float | None] = mapped_column(Float, nullable=True)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
