import pytest

from ragrep.collection import Catalog, Record
from ragrep.models import Document
from ragrep.query.filter import parse_filters


def test_catalog_upserts_fetches_and_persists_records(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    catalog = Catalog(path)

    assert (
        catalog.upsert([
            Record(
                id="article-1",
                source="article",
                title="Wallet limits",
                content="How wallet limits work",
                metadata={"status": 2, "portal_id": "80000083721", "updated_at": "2026-05-16T10:00:00Z"},
            ),
            {
                "id": "video-1",
                "source": "youtube",
                "title": "Funding tutorial",
                "content": "How to add funds",
                "metadata": {"playlist_id": "playlist-a", "updated_at": "2026-05-16T11:00:00Z"},
            },
        ])
        == 2
    )
    catalog.save()

    loaded = Catalog.load(path)

    assert loaded.count() == 2
    assert loaded.fetch_metadata(["article-1", "missing", "video-1"]) == {
        "article-1": {
            "id": "article-1",
            "source": "article",
            "title": "Wallet limits",
            "content": "How wallet limits work",
            "status": 2,
            "portal_id": "80000083721",
            "updated_at": "2026-05-16T10:00:00Z",
        },
        "video-1": {
            "id": "video-1",
            "source": "youtube",
            "title": "Funding tutorial",
            "content": "How to add funds",
            "playlist_id": "playlist-a",
            "updated_at": "2026-05-16T11:00:00Z",
        },
    }


def test_catalog_replaces_records_and_lists_ids_by_metadata_filter(tmp_path) -> None:
    catalog = Catalog(tmp_path / "records.jsonl")
    catalog.upsert([
        Record(id="article-1", source="article", content="old", metadata={"status": 1, "folder_id": "old"}),
        Record(
            id="article-2", source="article", content="live offer", metadata={"status": 2, "folder_name": "Live Offers"}
        ),
    ])
    catalog.upsert([
        Record(id="article-1", source="article", content="new", metadata={"status": 2, "folder_id": "live"})
    ])

    assert catalog.count() == 2
    assert catalog.fetch(["article-1"])["article-1"].content == "new"
    assert catalog.list_ids(filters=parse_filters(["status=2"])) == ["article-1", "article-2"]
    assert catalog.list_ids(
        filters=parse_filters(['{"$or": [{"folder_name": "Live Offers"}, {"folder_id": "live"}]}'])
    ) == [
        "article-1",
        "article-2",
    ]


def test_catalog_deletes_and_clears_filtered_records(tmp_path) -> None:
    catalog = Catalog(tmp_path / "records.jsonl")
    catalog.upsert([
        Record(id="article-1", source="article", content="one", metadata={"portal_id": "p1"}),
        Record(id="article-2", source="article", content="two", metadata={"portal_id": "p2"}),
        Record(id="video-1", source="youtube", content="video", metadata={"playlist_id": "p1"}),
    ])

    assert catalog.delete(["missing", "article-2"]) == 1
    assert catalog.list_ids(source="article") == ["article-1"]
    assert catalog.clear(filters=parse_filters(["playlist_id=p1"])) == 1
    assert catalog.list_ids() == ["article-1"]
    assert catalog.clear() == 1
    assert catalog.count() == 0


def test_catalog_exports_records_as_documents(tmp_path) -> None:
    catalog = Catalog(tmp_path / "records.jsonl")
    catalog.upsert([
        Document(id="doc-1", source="article", content="content", title="Title", metadata={"status": 2}),
    ])

    assert catalog.documents() == [
        Document(id="doc-1", source="article", content="content", title="Title", metadata={"status": 2}),
    ]


def test_catalog_rejects_invalid_record_boundaries(tmp_path) -> None:
    with pytest.raises(ValueError, match="id must not be empty"):
        Record(id="", source="article", content="content")
    with pytest.raises(TypeError, match="metadata value"):
        Record(id="article-1", source="article", content="content", metadata={"tags": ["wallet"]})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reserved"):
        Record(id="article-1", source="article", content="content", metadata={"title": "shadow"})

    path = tmp_path / "records.jsonl"
    path.write_text('{"id": "article-1", "source": "article", "content": "content", "metadata": {"bad": null}}\n')
    with pytest.raises(ValueError, match="Invalid collection record"):
        Catalog.load(path)
