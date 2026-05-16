from ragrep import Catalog, EmbedModel, Index, Record, RecordHit
from ragrep.models import IngestStats, QueryResult, SearchResult
from ragrep.query.filter import parse_filters


def test_index_upserts_records_persists_catalog_and_reindexes(monkeypatch, tmp_path) -> None:
    index = Index(tmp_path, [EmbedModel(provider="test", model_name="embed")])
    calls = []

    def ingest(docs, *, force=False, batch=False):
        calls.append((docs, force, batch))
        return IngestStats(documents=len(docs), chunks=len(docs) * 2)

    monkeypatch.setattr(index, "ingest", ingest)

    stats = index.upsert_records(
        [
            Record(id="article-1", source="article", title="Wallet", content="Wallet limits", metadata={"status": 2}),
            {
                "id": "video-1",
                "source": "youtube",
                "title": "Wallet tutorial",
                "content": "Video about wallet limits",
                "metadata": {"playlist_id": "playlist-a"},
            },
        ],
        force=True,
        batch=True,
    )

    assert stats.documents == 2
    assert stats.chunks == 4
    assert [doc.id for doc in calls[0][0]] == ["article-1", "video-1"]
    assert calls[0][1:] == (True, True)
    assert Catalog.load(index.catalog_path).list_ids() == ["article-1", "video-1"]


def test_index_record_lookup_methods_use_persisted_catalog(monkeypatch, tmp_path) -> None:
    index = Index(tmp_path, [EmbedModel(provider="test", model_name="embed")])
    monkeypatch.setattr(index, "ingest", lambda docs, **_: IngestStats(documents=len(docs)))

    index.upsert_records([
        Record(id="article-1", source="article", content="one", metadata={"status": 2, "folder_id": "live"}),
        Record(id="article-2", source="article", content="two", metadata={"status": 1, "folder_name": "Drafts"}),
        Record(id="video-1", source="youtube", content="three", metadata={"playlist_id": "playlist-a"}),
    ])

    assert index.fetch_records_metadata(["article-1"])["article-1"]["status"] == 2
    assert index.list_record_ids(source_filter="article", metadata_filter=parse_filters(["status=2"])) == ["article-1"]
    assert index.count_records(metadata_filter=parse_filters(["playlist_id=playlist-a"])) == 1


def test_index_deletes_and_clears_records_before_reindexing(monkeypatch, tmp_path) -> None:
    index = Index(tmp_path, [EmbedModel(provider="test", model_name="embed")])
    reindexed = []

    def ingest(docs, **_):
        reindexed.append([doc.id for doc in docs])
        return IngestStats(documents=len(docs))

    monkeypatch.setattr(index, "ingest", ingest)
    index.upsert_records([
        Record(id="article-1", source="article", content="one", metadata={"portal_id": "p1"}),
        Record(id="article-2", source="article", content="two", metadata={"portal_id": "p2"}),
        Record(id="video-1", source="youtube", content="three", metadata={"playlist_id": "p1"}),
    ])

    index.delete_records(["article-2"])
    index.clear_records(metadata_filter=parse_filters(["playlist_id=p1"]))

    assert reindexed[-2:] == [["article-1", "video-1"], ["article-1"]]
    assert index.list_record_ids() == ["article-1"]


def test_index_clears_search_artifacts_when_no_records_remain(tmp_path) -> None:
    index = Index(tmp_path, [EmbedModel(provider="test", model_name="embed")])
    for name in ("faiss.index", "chunks.pkl", "bm25.pkl"):
        (tmp_path / name).write_text("stale")
    models_dir = tmp_path / "models" / "test--embed"
    models_dir.mkdir(parents=True)
    (models_dir / "faiss.index").write_text("stale")

    stats = index.clear_records()

    assert stats.documents == 0
    assert not (tmp_path / "faiss.index").exists()
    assert not (tmp_path / "chunks.pkl").exists()
    assert not (tmp_path / "bm25.pkl").exists()
    assert not (tmp_path / "models").exists()


def test_index_search_records_returns_record_hits(monkeypatch, tmp_path) -> None:
    index = Index(tmp_path, [EmbedModel(provider="test", model_name="embed")])

    def query(question, *, top_k, source_filter, metadata_filter, no_generate):
        assert (question, top_k, source_filter, metadata_filter, no_generate) == (
            "wallet",
            9,
            "article",
            {"status": 2},
            True,
        )
        return QueryResult(
            answer="",
            query=question,
            sources=[
                SearchResult(
                    chunk_id="article:with:colon:0",
                    content="Wallet limits",
                    title="Wallet",
                    source="article",
                    metadata={"status": 2},
                    rrf_score=0.25,
                ),
                SearchResult(
                    chunk_id="article:with:colon:1",
                    content="Wallet limits continued",
                    title="Wallet",
                    source="article",
                    metadata={"status": 2},
                    rrf_score=0.2,
                ),
            ],
        )

    monkeypatch.setattr(index, "query", query)

    assert index.search_records("wallet", top_k=3, source_filter="article", metadata_filter={"status": 2}) == [
        RecordHit(
            record_id="article:with:colon",
            chunk_id="article:with:colon:0",
            content="Wallet limits",
            title="Wallet",
            source="article",
            metadata={"status": 2},
            score=0.25,
            rrf_score=0.25,
        )
    ]
