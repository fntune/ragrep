"""Collection records that can be indexed as ragrep documents."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from ragrep.models import Document, MetadataValue

_RESERVED_FIELDS = {"id", "source", "title", "content"}


@dataclass(frozen=True)
class Record:
    """A mutable collection record before chunking and embedding."""

    id: str
    source: str
    content: str
    title: str = ""
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text("id", self.id)
        _require_text("source", self.source)
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")
        if not isinstance(self.title, str):
            raise TypeError("title must be a string")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be an object")
        object.__setattr__(self, "metadata", _validate_metadata(self.metadata))

    @classmethod
    def from_document(cls, document: Document) -> "Record":
        return cls(
            id=document.id,
            source=document.source,
            content=document.content,
            title=document.title,
            metadata=dict(document.metadata),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "Record":
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be an object")
        return cls(
            id=_mapping_text(data, "id"),
            source=_mapping_text(data, "source"),
            content=_mapping_text(data, "content", allow_empty=True),
            title=_mapping_text(data, "title", default="", allow_empty=True),
            metadata=_validate_metadata(metadata),
        )

    def to_document(self) -> Document:
        return Document(
            id=self.id,
            source=self.source,
            content=self.content,
            title=self.title,
            metadata=dict(self.metadata),
        )

    def fields(self) -> dict[str, MetadataValue]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "content": self.content,
            **self.metadata,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "source": self.source,
            "content": self.content,
            "title": self.title,
            "metadata": dict(self.metadata),
        }


def _mapping_text(
    data: Mapping[str, object],
    key: str,
    *,
    default: str | None = None,
    allow_empty: bool = False,
) -> str:
    if key not in data:
        if default is not None:
            return default
        raise ValueError(f"{key} is required")
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    if not allow_empty:
        _require_text(key, value)
    return value


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_metadata(metadata: Mapping[object, object]) -> dict[str, MetadataValue]:
    validated: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("metadata keys must be non-empty strings")
        if key in _RESERVED_FIELDS:
            raise ValueError(f"metadata key {key!r} is reserved")
        if not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"metadata value for {key!r} must be a scalar")
        validated[key] = value
    return validated
