# Lazy imports — avoid pulling heavy deps (numpy, torch, faiss) at package import
# so `pip install ragrep` (client-only) works without them.


def __getattr__(name: str):
    if name in ("EmbedModel", "Index"):
        from ragrep.index import EmbedModel, Index
        return {"EmbedModel": EmbedModel, "Index": Index}[name]
    if name in ("Document", "QueryResult", "SearchResult"):
        from ragrep.models import Document, QueryResult, SearchResult
        return {"Document": Document, "QueryResult": QueryResult, "SearchResult": SearchResult}[name]
    raise AttributeError(f"module 'ragrep' has no attribute {name!r}")


__all__ = ["Document", "EmbedModel", "Index", "QueryResult", "SearchResult"]
