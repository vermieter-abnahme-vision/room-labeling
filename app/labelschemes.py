"""Central definition of annotation schemes (single- vs. multi-label, allowed labels).

Annotations are generic over a ``scheme`` name so that e.g. element labels
(window, radiator, ...) can be added later without touching the DB schema.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache

from app.config import Settings, get_settings

ROOM_SCHEME = "room"

_DISPLAY_NAMES: Mapping[str, str] = {
    "bad": "Bad",
    "kueche": "Küche",
    "wohnzimmer": "Wohnzimmer",
    "schlafzimmer": "Schlafzimmer",
    "flur": "Flur",
    "unklar": "Unklar",
}

# English zero-shot prompts for SigLIP. Unknown (config-added) labels fall back to the label name.
ZERO_SHOT_PROMPTS: Mapping[str, str] = {
    "bad": "a photo of a bathroom",
    "kueche": "a photo of a kitchen",
    "wohnzimmer": "a photo of a living room",
    "schlafzimmer": "a photo of a bedroom",
    "flur": "a photo of a hallway",
    "unklar": "a blurry photo of an unidentifiable room",
}


class SchemeKind(enum.StrEnum):
    SINGLE = "single"  # exactly one label per (image, labeler)
    MULTI = "multi"  # any number of labels per (image, labeler)


class UnknownSchemeError(KeyError):
    pass


class InvalidLabelError(ValueError):
    pass


@dataclass(frozen=True)
class LabelScheme:
    name: str
    kind: SchemeKind
    labels: tuple[str, ...]
    display: Mapping[str, str] = field(default_factory=dict)
    prompts: Mapping[str, str] = field(default_factory=dict)

    def display_name(self, label: str) -> str:
        return self.display.get(label) or label.replace("_", " ").title()

    def prompt(self, label: str) -> str:
        return self.prompts.get(label) or f"a photo of {label.replace('_', ' ')}"

    def validate_label(self, label: str) -> str:
        if label not in self.labels:
            raise InvalidLabelError(f"Label '{label}' is not allowed in scheme '{self.name}'.")
        return label

    @property
    def is_single(self) -> bool:
        return self.kind is SchemeKind.SINGLE


class SchemeRegistry:
    def __init__(self, schemes: list[LabelScheme]) -> None:
        self._schemes = {scheme.name: scheme for scheme in schemes}

    def get(self, name: str) -> LabelScheme:
        try:
            return self._schemes[name]
        except KeyError:
            raise UnknownSchemeError(name) from None

    def names(self) -> list[str]:
        return list(self._schemes)

    @classmethod
    def from_settings(cls, settings: Settings) -> SchemeRegistry:
        room = LabelScheme(
            name=ROOM_SCHEME,
            kind=SchemeKind.SINGLE,
            labels=tuple(settings.room_labels),
            display=_DISPLAY_NAMES,
            prompts=ZERO_SHOT_PROMPTS,
        )
        # Future: LabelScheme("window", SchemeKind.MULTI, (...)) etc.
        return cls([room])


@lru_cache
def default_registry() -> SchemeRegistry:
    return SchemeRegistry.from_settings(get_settings())
