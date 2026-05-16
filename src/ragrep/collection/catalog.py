"""Persisted collection catalog for mutable records."""

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from ragrep.collection.record import Record
from ragrep.models import Document, MetadataValue
from ragrep.query.filter import MetadataFilter, matches_filters

RecordInput = Record | Document | Mapping[str, object]


class Catalog:
    """Record catalog backed by a JSONL file."""

    def __init__(self, path: str | Path, records: Iterable[RecordInput] | None = None) -> None:
        self.path = Path(path)
        self._records: dict[str, Record] = {}
        if records:
            self.upsert(records)

    @classmethod
    def load(cls, path: str | Path) -> "Catalog":
        catalog = cls(path)
        if not catalog.path.exists():
            return catalog

        records = []
        with catalog.path.open() as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    if not isinstance(raw, dict):
                        raise TypeError("record line must be an object")
                    records.append(Record.from_mapping(raw))
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid collection record at {catalog.path}:{line_number}: {exc}") from exc
        catalog.upsert(records)
        return catalog

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        with tmp.open("w") as f:
            for record in self._records.values():
                f.write(json.dumps(record.to_dict(), sort_keys=True))
                f.write("\n")
        tmp.replace(self.path)

    def upsert(self, records: Iterable[RecordInput]) -> int:
        count = 0
        for item in records:
            record = _coerce_record(item)
            self._records[record.id] = record
            count += 1
        return count

    def fetch(self, ids: Iterable[str]) -> dict[str, Record]:
        found: dict[str, Record] = {}
        for record_id in ids:
            record = self._records.get(str(record_id))
            if record is not None:
                found[record.id] = record
        return found

    def fetch_metadata(self, ids: Iterable[str]) -> dict[str, dict[str, MetadataValue]]:
        return {record_id: record.fields() for record_id, record in self.fetch(ids).items()}

    def list_ids(
        self,
        *,
        source: str | None = None,
        filters: MetadataFilter | None = None,
    ) -> list[str]:
        return [record.id for record in self._records.values() if self._matches(record, source, filters)]

    def records(
        self,
        *,
        source: str | None = None,
        filters: MetadataFilter | None = None,
    ) -> list[Record]:
        return [record for record in self._records.values() if self._matches(record, source, filters)]

    def documents(
        self,
        *,
        source: str | None = None,
        filters: MetadataFilter | None = None,
    ) -> list[Document]:
        return [record.to_document() for record in self.records(source=source, filters=filters)]

    def count(
        self,
        *,
        source: str | None = None,
        filters: MetadataFilter | None = None,
    ) -> int:
        return len(self.list_ids(source=source, filters=filters))

    def delete(self, ids: Iterable[str]) -> int:
        deleted = 0
        for record_id in ids:
            if self._records.pop(str(record_id), None) is not None:
                deleted += 1
        return deleted

    def clear(
        self,
        *,
        source: str | None = None,
        filters: MetadataFilter | None = None,
    ) -> int:
        ids = self.list_ids(source=source, filters=filters)
        return self.delete(ids)

    def _matches(self, record: Record, source: str | None, filters: MetadataFilter | None) -> bool:
        if source and record.source != source:
            return False
        return matches_filters(record.fields(), filters or {})


def _coerce_record(item: RecordInput) -> Record:
    if isinstance(item, Record):
        return item
    if isinstance(item, Document):
        return Record.from_document(item)
    return Record.from_mapping(item)
