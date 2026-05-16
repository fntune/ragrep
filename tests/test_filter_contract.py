import pytest

from ragrep.query.filter import matches_filters, parse_filters


def test_key_value_filters_parse_typed_values_and_match_exactly() -> None:
    filters = parse_filters(["status=2", "published=true", "author=Alice"])

    assert filters == {"status": 2, "published": True, "author": "Alice"}
    assert matches_filters({"status": 2, "published": True, "author": "alice"}, filters)
    assert not matches_filters({"status": 2, "published": True, "author": "lic"}, filters)


def test_json_filters_support_top_level_or() -> None:
    filters = parse_filters(['{"$or": [{"folder_name": "Live Offers"}, {"folder_id": "80000723382"}]}'])

    assert matches_filters({"folder_name": "Live Offers", "folder_id": "other"}, filters)
    assert matches_filters({"folder_name": "Archive", "folder_id": "80000723382"}, filters)
    assert not matches_filters({"folder_name": "Archive", "folder_id": "other"}, filters)


def test_filter_matching_accepts_numeric_strings_for_external_metadata() -> None:
    filters = parse_filters(["portal_id=80000083721"])

    assert matches_filters({"portal_id": "80000083721"}, filters)


def test_date_range_stays_part_of_filter_contract() -> None:
    filters = parse_filters(["author=alice"])

    assert matches_filters({"author": "Alice", "date": "2026-05-10"}, filters, after="2026-05-01", before="2026-06-01")
    assert not matches_filters({"author": "Alice", "date": "2026-04-30"}, filters, after="2026-05-01")


def test_invalid_filter_shapes_fail_at_boundary() -> None:
    with pytest.raises(ValueError, match="expected key=value or JSON object"):
        parse_filters(["status"])
    with pytest.raises(ValueError, match=r"\$or filter must be a list of objects"):
        parse_filters(['{"$or": {"status": 2}}'])
    with pytest.raises(ValueError, match="Unsupported filter value"):
        parse_filters(['{"status": {"$eq": 2}}'])
