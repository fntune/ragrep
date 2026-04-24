"""LLM generation via Ollama HTTP API."""

import logging

import httpx

log = logging.getLogger(__name__)


def generate(
    messages: list[dict[str, str]],
    model: str = "gemma3:4b",
    ollama_url: str = "http://localhost:11434",
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """Generate a response via Ollama chat API."""
    url = f"{ollama_url}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    try:
        response = httpx.post(url, json=payload, timeout=120.0)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")
    except httpx.ConnectError:
        log.error("Cannot connect to Ollama at %s. Is it running?", ollama_url)
        raise
    except httpx.HTTPStatusError as e:
        log.error("Ollama returned %d: %s", e.response.status_code, e.response.text)
        raise
