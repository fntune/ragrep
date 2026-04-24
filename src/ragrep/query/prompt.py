"""Prompt assembly with retrieved context and citations."""

from ragrep.models import SearchResult

SYSTEM_PROMPT = (
    "You are a knowledge assistant. "
    "Answer questions using the provided context documents. "
    "If the context doesn't contain enough information to answer fully, say so. "
    "Always cite your sources using [Source: title] format."
)


def build_prompt(query: str, results: list[SearchResult]) -> list[dict[str, str]]:
    """Assemble chat messages for the LLM with retrieved context."""
    # Build context block
    context_parts = []
    for i, result in enumerate(results, 1):
        source_label = f"{result.title} ({result.source})"
        context_parts.append(f"[{i}] Source: {source_label}\n{result.content}")

    context = "\n\n".join(context_parts)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nBased on the context above, answer: {query}"},
    ]
