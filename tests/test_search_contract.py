import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from ragrep.models import Chunk, SearchResult
from ragrep.search import search_grep, search_hybrid, search_semantic


def _module(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        raw_dir=tmp_path / "raw",
        index_dir=tmp_path / "index",
        ingest=SimpleNamespace(max_chunk_tokens=128, chunk_overlap_tokens=0),
        embedding=SimpleNamespace(provider="test", model_name="test-model"),
        reranker=SimpleNamespace(provider="test", model_name="test-reranker"),
        retrieval=SimpleNamespace(top_k_dense=3, top_k_bm25=4, rrf_k=60),
    )


def test_grep_search_filters_by_source_metadata_and_date(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(
        sys.modules,
        "ragrep.ingest.store",
        _module("ragrep.ingest.store", index_exists=lambda _index_dir: False, load_index=None),
    )
    config = _config(tmp_path)
    config.raw_dir.mkdir()
    (config.raw_dir / "git.jsonl").write_text(
        "\n".join([
            json.dumps({
                "repo": "api",
                "hash": "a1",
                "subject": "Auth flow",
                "body": "Token refresh flow validates the auth session.",
                "author": "Alice",
                "date": "2026-05-10",
            }),
            json.dumps({
                "repo": "api",
                "hash": "b2",
                "subject": "Billing config",
                "body": "Billing limits live in product settings.",
                "author": "Bob",
                "date": "2026-04-01",
            }),
        ])
    )

    result = search_grep(
        config,
        "auth",
        source="git",
        filters={"author": "alice"},
        after="2026-05-01",
        before="2026-06-01",
        context=80,
        metadata=True,
    )

    assert result["mode"] == "grep"
    assert result["total_matches"] == 1
    assert result["results"] == [
        {
            "rank": 1,
            "id": "git:api:a1:0",
            "source": "git",
            "title": "api: Auth flow",
            "snippet": "Auth flow Token refresh flow validates the auth session. Author: Alice Date: 202...",
            "metadata": {"repo": "api", "author": "Alice", "date": "2026-05-10", "chunk_idx": 0},
        }
    ]


def test_semantic_search_formats_scores_and_metadata(monkeypatch, tmp_path) -> None:
    chunk = Chunk(
        id="doc:1:0",
        doc_id="doc:1",
        content="Auth flow validates sessions.",
        title="Auth",
        source="git",
        metadata={"repo": "api", "date": "2026-05-10"},
    )
    calls: dict[str, object] = {}

    class Embedder:
        def embed_query(self, term: str) -> str:
            calls["query"] = term
            return "embedding"

    def dense_search(query_embedding, _faiss, chunks, top_k, source_filter=None, metadata_filter=None):
        calls["dense"] = (query_embedding, chunks, top_k, source_filter, metadata_filter)
        return [(0, 0.98765)]

    monkeypatch.setitem(
        sys.modules,
        "ragrep.ingest.store",
        _module("ragrep.ingest.store", load_index=lambda _index_dir: ("faiss", [chunk], "bm25")),
    )
    monkeypatch.setitem(
        sys.modules,
        "ragrep.ingest.embed",
        _module("ragrep.ingest.embed", make_embedder=lambda _provider, _model_name: Embedder()),
    )
    monkeypatch.setitem(
        sys.modules,
        "ragrep.query.retrieve",
        _module("ragrep.query.retrieve", dense_search=dense_search),
    )

    result = search_semantic(
        _config(tmp_path),
        "auth",
        source="git",
        n=1,
        filters={"repo": "api"},
        after="2026-05-01",
        context=24,
        scores=True,
        metadata=True,
    )

    assert calls["query"] == "auth"
    assert calls["dense"] == ("embedding", [chunk], 5, "git", {"repo": "api"})
    assert result == {
        "query": "auth",
        "mode": "semantic",
        "results": [
            {
                "rank": 1,
                "id": "doc:1:0",
                "source": "git",
                "title": "Auth",
                "score": 0.988,
                "snippet": "Auth flow validates sess...",
                "metadata": {"repo": "api", "date": "2026-05-10"},
            }
        ],
    }


def test_hybrid_search_formats_retrieval_and_rerank_scores(monkeypatch, tmp_path) -> None:
    candidate = SearchResult(
        chunk_id="doc:2:0",
        content="Offer rules live in Freshdesk.",
        title="Live Offers",
        source="freshdesk",
        metadata={"folder_id": "offers", "date": "2026-05-09"},
        dense_score=0.81,
        bm25_score=2.4,
        rrf_score=0.12345,
        rerank_score=0.95,
    )
    calls: dict[str, object] = {}

    class Embedder:
        def embed_query(self, term: str) -> str:
            return f"embedding:{term}"

    class Reranker:
        def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
            calls["rerank"] = (query, candidates, top_k)
            return candidates[:top_k]

    def retrieve(**kwargs):
        calls["retrieve"] = kwargs
        return [candidate]

    monkeypatch.setitem(
        sys.modules,
        "ragrep.ingest.store",
        _module("ragrep.ingest.store", load_index=lambda _index_dir: ("faiss", ["chunk"], "bm25")),
    )
    monkeypatch.setitem(
        sys.modules,
        "ragrep.ingest.embed",
        _module("ragrep.ingest.embed", make_embedder=lambda _provider, _model_name: Embedder()),
    )
    monkeypatch.setitem(sys.modules, "ragrep.query.retrieve", _module("ragrep.query.retrieve", retrieve=retrieve))
    monkeypatch.setitem(
        sys.modules,
        "ragrep.query.rerank",
        _module("ragrep.query.rerank", make_reranker=lambda _provider, _model_name: Reranker()),
    )

    result = search_hybrid(
        _config(tmp_path),
        "offers",
        n=1,
        filters={"folder_id": "offers"},
        before="2026-06-01",
        context=40,
        scores=True,
        metadata=True,
    )

    assert calls["retrieve"]["query_embedding"] == "embedding:offers"
    assert calls["retrieve"]["metadata_filter"] == {"folder_id": "offers"}
    assert calls["rerank"] == ("offers", [candidate], 1)
    assert result == {
        "query": "offers",
        "mode": "hybrid",
        "results": [
            {
                "rank": 1,
                "id": "doc:2:0",
                "source": "freshdesk",
                "title": "Live Offers",
                "rerank": 0.95,
                "rrf": 0.123,
                "dense": 0.81,
                "bm25": 2.4,
                "snippet": "Offer rules live in Freshdesk.",
                "metadata": {"folder_id": "offers", "date": "2026-05-09"},
            }
        ],
    }


def test_http_search_handler_validates_filters(monkeypatch) -> None:
    server = _import_server_with_fake_fastapi(monkeypatch)

    response = asyncio.run(server.search(q="auth", filter="not-a-filter"))

    assert response.status_code == 400
    assert response.content == {"error": "Invalid filter: 'not-a-filter' (expected key=value)"}


def test_http_search_handler_passes_search_contract(monkeypatch) -> None:
    server = _import_server_with_fake_fastapi(monkeypatch)
    server.app.state.config = object()
    calls: dict[str, object] = {}

    def fake_search(**kwargs):
        calls.update(kwargs)
        return {"query": kwargs["term"], "mode": "grep", "results": []}

    import ragrep.search

    monkeypatch.setattr(ragrep.search, "search_grep", fake_search)

    result = asyncio.run(
        server.search(
            q="auth",
            mode="grep",
            n=7,
            source="git",
            filter="author=alice",
            after="2026-05-01",
            context=42,
            scores=False,
            metadata=True,
        )
    )

    assert result == {"query": "auth", "mode": "grep", "results": []}
    assert calls == {
        "config": server.app.state.config,
        "term": "auth",
        "source": "git",
        "n": 7,
        "filters": {"author": "alice"},
        "after": "2026-05-01",
        "before": None,
        "context": 42,
        "full": False,
        "scores": False,
        "metadata": True,
    }


def _import_server_with_fake_fastapi(monkeypatch):
    class FakeFastAPI:
        def __init__(self, *args, **kwargs) -> None:
            self.state = SimpleNamespace()

        def get(self, _path: str):
            return lambda fn: fn

    class FakeJSONResponse:
        def __init__(self, status_code: int, content: dict) -> None:
            self.status_code = status_code
            self.content = content

    monkeypatch.setitem(
        sys.modules, "fastapi", _module("fastapi", FastAPI=FakeFastAPI, Query=lambda default, **_: default)
    )
    monkeypatch.setitem(sys.modules, "fastapi.responses", _module("fastapi.responses", JSONResponse=FakeJSONResponse))
    sys.modules.pop("ragrep.server", None)
    return importlib.import_module("ragrep.server")
