"""Metadata filter parsing and matching for search/retrieval contracts."""

import json
import re

FilterScalar = str | int | float | bool
MetadataFilter = dict[str, object]


def parse_filters(raw: list[str]) -> MetadataFilter:
    """Parse CLI/HTTP filter input.

    Supported forms:
    - repeated `key=value` filters
    - one JSON object, including top-level `$or`
    """
    if not raw:
        return {}

    if len(raw) == 1 and raw[0].lstrip().startswith("{"):
        try:
            parsed = json.loads(raw[0])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid filter JSON: {exc.msg}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Filter JSON must be an object")
        return _validate_filter(parsed)

    filters: MetadataFilter = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"Invalid filter: {item!r} (expected key=value or JSON object)")
        key, _, value = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid filter: {item!r} (empty key)")
        filters[key] = _parse_scalar(value.strip())
    return _validate_filter(filters)


def matches_filters(
    metadata: dict,
    filters: MetadataFilter,
    after: str | None = None,
    before: str | None = None,
) -> bool:
    """Match exact metadata filters plus optional date range."""
    if not _matches_filter(metadata, filters):
        return False
    if after or before:
        date_str = str(metadata.get("date", ""))[:10]
        if not date_str or len(date_str) < 10:
            return False
        if after and date_str < after:
            return False
        if before and date_str >= before:
            return False
    return True


def _matches_filter(metadata: dict, filters: MetadataFilter) -> bool:
    for key, expected in filters.items():
        if key == "$or":
            if not isinstance(expected, list):
                return False
            if not any(_matches_filter(metadata, option) for option in expected if isinstance(option, dict)):
                return False
            continue

        if not _matches_scalar(metadata.get(key), expected):
            return False
    return True


def _matches_scalar(actual: object, expected: object) -> bool:
    if actual is None or isinstance(expected, (dict, list)):
        return False
    if actual == expected:
        return True
    return str(actual).casefold() == str(expected).casefold()


def _parse_scalar(value: str) -> FilterScalar:
    lower = value.casefold()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if re.fullmatch(r"-?(0|[1-9]\d*)", value):
        return int(value)
    if re.fullmatch(r"-?(0|[1-9]\d*)\.\d+", value):
        return float(value)
    return value


def _validate_filter(filters: dict) -> MetadataFilter:
    validated: MetadataFilter = {}
    for key, value in filters.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Filter keys must be non-empty strings")
        if key == "$or":
            if not isinstance(value, list) or not all(isinstance(option, dict) for option in value):
                raise ValueError("$or filter must be a list of objects")
            validated[key] = [_validate_filter(option) for option in value]
            continue
        if isinstance(value, (dict, list)) or not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"Unsupported filter value for {key!r}: {value!r}")
        validated[key] = value
    return validated
