import importlib
import pickle
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _restore_store_module():
    previous = sys.modules.get("ragrep.ingest.store")
    yield
    if previous is None:
        sys.modules.pop("ragrep.ingest.store", None)
    else:
        sys.modules["ragrep.ingest.store"] = previous


class _FakeIndex:
    def __init__(self, value: str) -> None:
        self.value = value
        self.ntotal = 1


class _FakeBm25:
    pass


def _load_store(monkeypatch, *, fail_on: str | None = None):
    def write_index(index: _FakeIndex, path: str) -> None:
        if index.value == fail_on:
            raise RuntimeError("write failed")
        with open(path, "w") as f:
            f.write(index.value)

    def read_index(path: str) -> _FakeIndex:
        with open(path) as f:
            return _FakeIndex(f.read())

    monkeypatch.setitem(
        sys.modules, "faiss", SimpleNamespace(IndexFlatIP=_FakeIndex, write_index=write_index, read_index=read_index)
    )
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(ndarray=object))
    monkeypatch.setitem(sys.modules, "rank_bm25", SimpleNamespace(BM25Okapi=_FakeBm25))
    sys.modules.pop("ragrep.ingest.store", None)
    return importlib.import_module("ragrep.ingest.store")


def test_save_multi_index_publishes_complete_artifacts(monkeypatch, tmp_path) -> None:
    store = _load_store(monkeypatch)

    chunks = ["chunk"]
    bm25 = _FakeBm25()
    store.save_multi_index(
        tmp_path, {"model-a": _FakeIndex("vectors-a"), "model-b": _FakeIndex("vectors-b")}, chunks, bm25
    )

    assert (tmp_path / "faiss.index").read_text() == "vectors-a"
    assert (tmp_path / "models" / "model-a" / "faiss.index").read_text() == "vectors-a"
    assert (tmp_path / "models" / "model-b" / "faiss.index").read_text() == "vectors-b"
    with open(tmp_path / "chunks.pkl", "rb") as f:
        assert pickle.load(f) == chunks
    with open(tmp_path / "bm25.pkl", "rb") as f:
        assert isinstance(pickle.load(f), _FakeBm25)


def test_save_multi_index_failure_leaves_previous_artifacts(monkeypatch, tmp_path) -> None:
    store = _load_store(monkeypatch)

    store.save_multi_index(tmp_path, {"model-a": _FakeIndex("old")}, ["old-chunk"], _FakeBm25())

    store = _load_store(monkeypatch, fail_on="new")
    try:
        store.save_multi_index(tmp_path, {"model-a": _FakeIndex("new")}, ["new-chunk"], _FakeBm25())
    except RuntimeError as exc:
        assert str(exc) == "write failed"
    else:
        raise AssertionError("expected write failure")

    assert (tmp_path / "faiss.index").read_text() == "old"
    assert (tmp_path / "models" / "model-a" / "faiss.index").read_text() == "old"
    with open(tmp_path / "chunks.pkl", "rb") as f:
        assert pickle.load(f) == ["old-chunk"]


def test_save_index_removes_obsolete_multi_model_artifacts(monkeypatch, tmp_path) -> None:
    store = _load_store(monkeypatch)

    store.save_multi_index(tmp_path, {"model-a": _FakeIndex("multi")}, ["chunk"], _FakeBm25())
    store.save_index(tmp_path, _FakeIndex("single"), ["single-chunk"], _FakeBm25())

    assert (tmp_path / "faiss.index").read_text() == "single"
    assert not (tmp_path / "models").exists()
    with open(tmp_path / "chunks.pkl", "rb") as f:
        assert pickle.load(f) == ["single-chunk"]


def test_save_multi_index_rejects_empty_model_indexes(monkeypatch, tmp_path) -> None:
    store = _load_store(monkeypatch)

    try:
        store.save_multi_index(tmp_path, {}, [], _FakeBm25())
    except ValueError as exc:
        assert str(exc) == "at least one model index required"
    else:
        raise AssertionError("expected empty model index failure")

    assert not (tmp_path / "faiss.index").exists()


def test_index_existence_checks_use_published_artifacts(monkeypatch, tmp_path) -> None:
    store = _load_store(monkeypatch)

    assert not store.index_exists(tmp_path)
    assert not store.multi_index_exists(tmp_path, ["model-a"])

    store.save_multi_index(tmp_path, {"model-a": _FakeIndex("vectors")}, ["chunk"], _FakeBm25())

    assert store.index_exists(tmp_path)
    assert store.multi_index_exists(tmp_path, ["model-a"])
    assert not store.multi_index_exists(tmp_path, ["missing-model"])
