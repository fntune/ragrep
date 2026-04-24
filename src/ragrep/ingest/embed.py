"""Embedding providers: Voyage AI (API) and sentence-transformers (local)."""

import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen

import numpy as np

log = logging.getLogger(__name__)

QUERY_INSTRUCTION = (
    "Given a question about an internal company knowledge base, "
    "retrieve relevant documents that answer the question"
)


def _notify(msg: str) -> None:
    """Send a notification via ntfy.sh if NTFY_TOPIC env var is set."""
    topic = os.environ.get("NTFY_TOPIC", "")
    if not topic:
        return
    try:
        req = Request(
            f"https://ntfy.sh/{topic}",
            data=msg.encode(),
            headers={"Title": "ragrep-ingest"},
        )
        urlopen(req, timeout=5)
    except Exception:
        log.debug("ntfy notification failed", exc_info=True)


class _AdaptiveThrottle:
    """Adapts request rate based on API responses with rate-limit memory.

    Starts conservative (30s for free tier 10K TPM with 16-chunk batches).
    Ramps down after consecutive successes. On rate limit, backs off and
    remembers the boundary — won't immediately probe below it again.
    The floor decays over time, allowing gradual probing (e.g. after adding
    a payment method, the floor erodes as successes accumulate).
    """

    def __init__(
        self,
        initial_delay: float = 30.0,
        min_delay: float = 0.2,
        max_delay: float = 90.0,
    ):
        self.delay = initial_delay
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._consecutive_ok = 0
        self._floor = 0.0  # learned lower bound from rate limits
        self._floor_successes = 0  # successes at or near floor

    def on_success(self) -> None:
        self._consecutive_ok += 1
        # Erode the floor: every 10 successes at the floor, lower it by 15%
        if self._floor > 0 and self.delay <= self._floor * 1.1:
            self._floor_successes += 1
            if self._floor_successes >= 10:
                old_floor = self._floor
                self._floor = max(self.min_delay, self._floor * 0.85)
                log.info("Floor eroded %.1fs -> %.1fs", old_floor, self._floor)
                self._floor_successes = 0

        if self._consecutive_ok >= 3:
            old = self.delay
            new = max(self.min_delay, self.delay * 0.75)
            # Don't go below the learned floor
            self.delay = max(new, self._floor)
            if self.delay < old:
                log.info("Throttle %.1fs -> %.1fs", old, self.delay)
            self._consecutive_ok = 0

    def on_rate_limit(self, retry_after: float | None = None) -> None:
        self._consecutive_ok = 0
        self._floor_successes = 0
        old = self.delay
        if retry_after and retry_after > 0:
            self.delay = min(self.max_delay, retry_after + 2.0)
        else:
            self.delay = min(self.max_delay, self.delay * 1.5)
        # Remember this as the floor — the delay that was too fast
        self._floor = self.delay
        log.info("Throttle %.1fs -> %.1fs (floor set)", old, self.delay)

    def wait(self) -> None:
        if self.delay >= 1.0:
            log.debug("Throttle: sleeping %.1fs", self.delay)
        time.sleep(self.delay)


_BATCH_POLL_INTERVAL = 60.0
_BATCH_POLL_TIMEOUT = 86400.0
_VOYAGE_BASE = "https://api.voyageai.com/v1"


class _RateLimit(Exception):
    """Raised when Voyage API returns 429."""

    def __init__(self, retry_after: float | None = None):
        self.retry_after = retry_after


def _write_jsonl(rows: list[dict]) -> Path:
    """Write rows to a temp JSONL file for batch upload."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, prefix="ragrep_batch_")
    for row in rows:
        tmp.write(json.dumps(row) + "\n")
    tmp.close()
    return Path(tmp.name)


def _poll_until_done(
    poll_fn: Callable[[], str],
    *,
    interval: float = _BATCH_POLL_INTERVAL,
    timeout: float = _BATCH_POLL_TIMEOUT,
    label: str = "batch",
) -> None:
    """Block until poll_fn returns a terminal status. Raises RuntimeError on failure/timeout."""
    t0 = time.monotonic()
    while True:
        status = poll_fn()
        if status == "completed":
            log.info("%s completed after %.0fs", label, time.monotonic() - t0)
            return
        if status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"{label} ended with status: {status}")
        elapsed = time.monotonic() - t0
        if elapsed > timeout:
            raise RuntimeError(f"{label} timed out after {elapsed:.0f}s")
        log.info("%s status=%s, elapsed=%.0fs", label, status, elapsed)
        _notify(f"{label}: {status} ({elapsed / 60:.0f}min)")
        time.sleep(interval)


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...
    def embed_query(self, query: str) -> np.ndarray: ...


class VoyageEmbedder:
    """Voyage AI API embedder — code + text in the same space."""

    # Cap API batch size to stay under free-tier 10K TPM per request.
    # 16 chunks * ~300 avg tokens = ~4,800 tokens, well under 10K.
    _MAX_API_BATCH = 16

    def __init__(self, model_name: str = "voyage-code-3"):
        self._api_key = os.environ.get("VOYAGE_API_KEY", "")
        self.model = model_name
        self.dim = 1024
        log.info("Voyage embedder ready (model=%s, dim=%d)", model_name, self.dim)

    def _call_voyage_embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        import httpx

        resp = httpx.post(
            f"{_VOYAGE_BASE}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self.model, "input": texts, "input_type": input_type},
            timeout=120,
        )
        if resp.status_code == 429:
            try:
                retry_after: float | None = float(resp.headers.get("retry-after") or 0) or None
            except (TypeError, ValueError):
                retry_after = None
            raise _RateLimit(retry_after)
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in data]

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 128,
        checkpoint_path: Path | None = None,
    ) -> np.ndarray:
        """Batch embed documents via Voyage API with adaptive rate limiting.

        Checkpoints progress to disk every 500 batches. If checkpoint_path is
        set and a checkpoint exists for the same total, resumes from there.
        """
        total = len(texts)
        api_batch = min(batch_size, self._MAX_API_BATCH)

        # Resume from checkpoint if available
        all_emb: list[list[float]] = []
        start_idx = 0
        if checkpoint_path and checkpoint_path.exists():
            try:
                data = np.load(checkpoint_path)
                ckpt_total = int(data["total"])
                ckpt_done = int(data["n_done"])
                if ckpt_total == total and ckpt_done > 0:
                    all_emb = data["embeddings"].tolist()
                    start_idx = ckpt_done
                    log.info("Resuming from checkpoint: %d/%d done", start_idx, total)
                    _notify(f"Resuming: {start_idx}/{total} already done")
                else:
                    log.warning(
                        "Checkpoint mismatch (total %d vs %d), starting fresh",
                        ckpt_total, total,
                    )
            except Exception:
                log.warning("Failed to load checkpoint, starting fresh", exc_info=True)

        log.info(
            "Embedding %d texts via Voyage API (api_batch=%d, start=%d)",
            total, api_batch, start_idx,
        )
        if start_idx == 0:
            _notify(f"Starting: {total} texts, api_batch={api_batch}")

        throttle = _AdaptiveThrottle()
        last_notify = time.monotonic()
        t0 = time.monotonic()

        for i in range(start_idx, total, api_batch):
            batch = texts[i : i + api_batch]

            # Throttle between batches (skip before first request)
            if i > start_idx:
                throttle.wait()

            for attempt in range(1, 6):
                try:
                    all_emb.extend(self._call_voyage_embed(batch, "document"))
                    throttle.on_success()
                    break
                except _RateLimit as exc:
                    if attempt >= 5:
                        _notify(f"FAILED at {len(all_emb)}/{total}: exhausted retries")
                        self._save_checkpoint(checkpoint_path, all_emb, total)
                        raise RuntimeError("Voyage rate limit exhausted after 5 attempts") from exc
                    throttle.on_rate_limit(exc.retry_after)
                    log.warning(
                        "Rate limited, waiting %.0fs (attempt %d/5, retry-after=%s)",
                        throttle.delay, attempt, exc.retry_after,
                    )
                    time.sleep(throttle.delay)
                except Exception as e:
                    _notify(f"FAILED at {len(all_emb)}/{total}: {e}")
                    self._save_checkpoint(checkpoint_path, all_emb, total)
                    raise

            done = len(all_emb)
            elapsed = time.monotonic() - t0
            # Rate calculation accounts for chunks done this session only
            session_done = done - start_idx
            rate = session_done / elapsed * 60 if elapsed > 0 else 0

            # Log progress every 10 batches or at end
            if done % (api_batch * 10) < api_batch or done >= total:
                eta_min = (total - done) / rate if rate > 0 else 0
                log.info(
                    "Progress: %d/%d (%.0f/min, ETA %.0fmin, throttle=%.1fs, floor=%.1fs)",
                    done, total, rate, eta_min, throttle.delay, throttle._floor,
                )

            # Checkpoint every 500 batches (8,000 chunks)
            if checkpoint_path and done % (api_batch * 500) < api_batch and done < total:
                self._save_checkpoint(checkpoint_path, all_emb, total)

            # ntfy every hour
            now = time.monotonic()
            if now - last_notify >= 3600:
                eta_min = (total - done) / rate if rate > 0 else 0
                _notify(
                    f"Progress: {done}/{total} ({done * 100 // total}%)\n"
                    f"Rate: {rate:.0f}/min | ETA: {eta_min:.0f}min\n"
                    f"Throttle: {throttle.delay:.1f}s (floor={throttle._floor:.1f}s)"
                )
                last_notify = now

        elapsed = time.monotonic() - t0
        _notify(f"Done: {total} texts in {elapsed / 60:.1f}min ({(total - start_idx) / elapsed * 60:.0f}/min)")

        # Clean up checkpoint on success
        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()
            log.info("Checkpoint removed (embedding complete)")

        return np.array(all_emb, dtype=np.float32)

    @staticmethod
    def _save_checkpoint(
        path: Path | None, embeddings: list[list[float]], total: int,
    ) -> None:
        if not path or not embeddings:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        n_done = len(embeddings)
        np.savez(path, n_done=n_done, total=total, embeddings=np.array(embeddings, dtype=np.float32))
        log.info("Checkpoint saved: %d/%d at %s", n_done, total, path)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        return np.array(
            self._call_voyage_embed([query], "query")[0], dtype=np.float32,
        ).reshape(1, -1)

    def submit_batch(self, texts: list[str]) -> dict:
        """Submit texts to Voyage Batch API. Returns handle for collect_batch()."""
        import httpx

        total = len(texts)
        log.info("Voyage batch: submitting %d texts", total)

        headers = {"Authorization": f"Bearer {self._api_key}"}

        rows = [
            {"custom_id": str(i), "body": {"input": [text]}}
            for i, text in enumerate(texts)
        ]
        jsonl_path = _write_jsonl(rows)

        with httpx.Client(timeout=httpx.Timeout(300, connect=30)) as client:
            with open(jsonl_path, "rb") as f:
                resp = client.post(
                    f"{_VOYAGE_BASE}/files", headers=headers,
                    files={"file": ("batch.jsonl", f, "application/jsonl")},
                    data={"purpose": "batch"},
                )
        jsonl_path.unlink(missing_ok=True)
        resp.raise_for_status()
        file_id = resp.json()["id"]
        log.info("Voyage batch: uploaded file %s", file_id)

        resp = httpx.post(
            f"{_VOYAGE_BASE}/batches", headers=headers,
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/embeddings",
                "completion_window": "12h",
                "request_params": {"model": self.model, "input_type": "document"},
            },
            timeout=30,
        )
        resp.raise_for_status()
        batch_id = resp.json()["id"]
        log.info("Voyage batch: created %s", batch_id)
        _notify(f"Voyage batch submitted: {batch_id}")

        return {"batch_id": batch_id, "total": total}

    def collect_batch(self, handle: dict) -> np.ndarray:
        """Poll and download results from a submitted Voyage batch."""
        import httpx

        batch_id = handle["batch_id"]
        total = handle["total"]
        headers = {"Authorization": f"Bearer {self._api_key}"}

        def poll() -> str:
            r = httpx.get(f"{_VOYAGE_BASE}/batches/{batch_id}", headers=headers, timeout=30)
            r.raise_for_status()
            data = r.json()
            status = data["status"]
            if status in ("failed", "expired", "cancelled"):
                log.error("Voyage batch %s failed: %s", batch_id, json.dumps(data, indent=2))
            return status

        _poll_until_done(poll, label=f"Voyage batch {batch_id}")

        r = httpx.get(f"{_VOYAGE_BASE}/batches/{batch_id}", headers=headers, timeout=30)
        r.raise_for_status()
        output_file_id = r.json()["output_file_id"]

        r = httpx.get(
            f"{_VOYAGE_BASE}/files/{output_file_id}/content", headers=headers,
            timeout=120, follow_redirects=True,
        )
        r.raise_for_status()
        lines = r.text.strip().split("\n")

        emb_by_id: dict[int, list[float]] = {}
        for line in lines:
            rec = json.loads(line)
            idx = int(rec["custom_id"])
            emb_by_id[idx] = rec["response"]["body"]["data"][0]["embedding"]

        if len(emb_by_id) != total:
            raise RuntimeError(f"Voyage batch: expected {total} results, got {len(emb_by_id)}")

        ordered = [emb_by_id[i] for i in range(total)]
        return np.array(ordered, dtype=np.float32)

    def embed_documents_batch(self, texts: list[str]) -> np.ndarray:
        """Embed via Voyage Batch API (33% discount, ≤12h). Blocking."""
        return self.collect_batch(self.submit_batch(texts))


class OpenAIEmbedder:
    """OpenAI API embedder (text-embedding-3-small / text-embedding-3-large)."""

    _MAX_API_BATCH = 64

    def __init__(self, model_name: str = "text-embedding-3-small"):
        import openai

        self.client = openai.OpenAI()
        self.model = model_name
        dims = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}
        self.dim = dims.get(model_name, 1536)
        log.info("OpenAI embedder ready (model=%s, dim=%d)", model_name, self.dim)

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 128,
        checkpoint_path: Path | None = None,
    ) -> np.ndarray:
        """Batch embed documents via OpenAI API with adaptive rate limiting."""
        import openai

        total = len(texts)
        api_batch = min(batch_size, self._MAX_API_BATCH)

        all_emb: list[list[float]] = []
        start_idx = 0
        if checkpoint_path and checkpoint_path.exists():
            try:
                data = np.load(checkpoint_path)
                if int(data["total"]) == total and int(data["n_done"]) > 0:
                    all_emb = data["embeddings"].tolist()
                    start_idx = int(data["n_done"])
                    log.info("Resuming from checkpoint: %d/%d done", start_idx, total)
            except Exception:
                log.warning("Failed to load checkpoint, starting fresh", exc_info=True)

        log.info("Embedding %d texts via OpenAI API (api_batch=%d, start=%d)", total, api_batch, start_idx)

        throttle = _AdaptiveThrottle(initial_delay=0.5, min_delay=0.05)
        t0 = time.monotonic()

        for i in range(start_idx, total, api_batch):
            batch = texts[i : i + api_batch]

            if i > start_idx:
                throttle.wait()

            for attempt in range(1, 6):
                try:
                    result = self.client.embeddings.create(input=batch, model=self.model)
                    for item in sorted(result.data, key=lambda x: x.index):
                        all_emb.append(item.embedding)
                    throttle.on_success()
                    break
                except openai.RateLimitError:
                    if attempt >= 5:
                        self._save_checkpoint(checkpoint_path, all_emb, total)
                        raise
                    throttle.on_rate_limit()
                    log.warning("Rate limited, waiting %.0fs (attempt %d/5)", throttle.delay, attempt)
                    time.sleep(throttle.delay)
                except Exception:
                    self._save_checkpoint(checkpoint_path, all_emb, total)
                    raise

            done = len(all_emb)
            if done % (api_batch * 10) < api_batch or done >= total:
                elapsed = time.monotonic() - t0
                session_done = done - start_idx
                rate = session_done / elapsed * 60 if elapsed > 0 else 0
                log.info("Progress: %d/%d (%.0f/min, throttle=%.1fs)", done, total, rate, throttle.delay)

            if checkpoint_path and done % (api_batch * 500) < api_batch and done < total:
                self._save_checkpoint(checkpoint_path, all_emb, total)

        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()

        embeddings = np.array(all_emb, dtype=np.float32)
        # L2-normalize for IndexFlatIP (cosine via inner product)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms
        return embeddings

    @staticmethod
    def _save_checkpoint(path: Path | None, embeddings: list[list[float]], total: int) -> None:
        if not path or not embeddings:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, n_done=len(embeddings), total=total, embeddings=np.array(embeddings, dtype=np.float32))
        log.info("Checkpoint saved: %d/%d at %s", len(embeddings), total, path)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        result = self.client.embeddings.create(input=[query], model=self.model)
        vec = np.array(result.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.reshape(1, -1)

    def submit_batch(self, texts: list[str]) -> dict:
        """Submit texts to OpenAI Batch API. Returns handle for collect_batch()."""
        total = len(texts)
        log.info("OpenAI batch: submitting %d texts", total)

        rows = [
            {
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": self.model, "input": text},
            }
            for i, text in enumerate(texts)
        ]
        jsonl_path = _write_jsonl(rows)

        with open(jsonl_path, "rb") as f:
            file_obj = self.client.files.create(file=f, purpose="batch")
        jsonl_path.unlink(missing_ok=True)
        log.info("OpenAI batch: uploaded file %s", file_obj.id)

        batch = self.client.batches.create(
            input_file_id=file_obj.id, endpoint="/v1/embeddings", completion_window="24h",
        )
        log.info("OpenAI batch: created %s", batch.id)
        _notify(f"OpenAI batch submitted: {batch.id}")

        return {"batch_id": batch.id, "total": total}

    def collect_batch(self, handle: dict) -> np.ndarray:
        """Poll and download results from a submitted OpenAI batch."""
        batch_id = handle["batch_id"]
        total = handle["total"]

        def poll() -> str:
            return self.client.batches.retrieve(batch_id).status

        _poll_until_done(poll, label=f"OpenAI batch {batch_id}")

        batch = self.client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            raise RuntimeError(f"OpenAI batch {batch_id}: no output file")
        content = self.client.files.content(batch.output_file_id)
        lines = content.text.strip().split("\n")

        emb_by_id: dict[int, list[float]] = {}
        for line in lines:
            rec = json.loads(line)
            idx = int(rec["custom_id"])
            emb_by_id[idx] = rec["response"]["body"]["data"][0]["embedding"]

        if len(emb_by_id) != total:
            raise RuntimeError(f"OpenAI batch: expected {total} results, got {len(emb_by_id)}")

        ordered = [emb_by_id[i] for i in range(total)]
        embeddings = np.array(ordered, dtype=np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms
        return embeddings

    def embed_documents_batch(self, texts: list[str]) -> np.ndarray:
        """Embed via OpenAI Batch API (50% discount, ≤24h). Blocking."""
        return self.collect_batch(self.submit_batch(texts))


class GeminiEmbedder:
    """Google Gemini API embedder (gemini-embedding-001)."""

    _MAX_API_BATCH = 100

    def __init__(self, model_name: str = "gemini-embedding-001"):
        from google import genai

        self.client = genai.Client()
        self.model = model_name
        self.dim = 3072
        log.info("Gemini embedder ready (model=%s, dim=%d)", model_name, self.dim)

    def embed_documents(
        self,
        texts: list[str],
        batch_size: int = 128,
        checkpoint_path: Path | None = None,
    ) -> np.ndarray:
        """Batch embed documents via Gemini API."""
        from google.genai import types

        total = len(texts)
        api_batch = min(batch_size, self._MAX_API_BATCH)

        all_emb: list[list[float]] = []
        start_idx = 0
        if checkpoint_path and checkpoint_path.exists():
            try:
                data = np.load(checkpoint_path)
                if int(data["total"]) == total and int(data["n_done"]) > 0:
                    all_emb = data["embeddings"].tolist()
                    start_idx = int(data["n_done"])
                    log.info("Resuming from checkpoint: %d/%d done", start_idx, total)
            except Exception:
                log.warning("Failed to load checkpoint, starting fresh", exc_info=True)

        log.info("Embedding %d texts via Gemini API (api_batch=%d, start=%d)", total, api_batch, start_idx)

        throttle = _AdaptiveThrottle(initial_delay=0.5, min_delay=0.05)
        t0 = time.monotonic()

        for i in range(start_idx, total, api_batch):
            batch = texts[i : i + api_batch]

            if i > start_idx:
                throttle.wait()

            for attempt in range(1, 6):
                try:
                    result = self.client.models.embed_content(
                        model=self.model,
                        contents=batch,
                        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
                    )
                    for emb in result.embeddings:
                        all_emb.append(list(emb.values))
                    throttle.on_success()
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if attempt >= 5:
                            self._save_checkpoint(checkpoint_path, all_emb, total)
                            raise
                        throttle.on_rate_limit()
                        log.warning("Rate limited, waiting %.0fs (attempt %d/5)", throttle.delay, attempt)
                        time.sleep(throttle.delay)
                    else:
                        self._save_checkpoint(checkpoint_path, all_emb, total)
                        raise

            done = len(all_emb)
            if done % (api_batch * 10) < api_batch or done >= total:
                elapsed = time.monotonic() - t0
                session_done = done - start_idx
                rate = session_done / elapsed * 60 if elapsed > 0 else 0
                log.info("Progress: %d/%d (%.0f/min, throttle=%.1fs)", done, total, rate, throttle.delay)

            if checkpoint_path and done % (api_batch * 500) < api_batch and done < total:
                self._save_checkpoint(checkpoint_path, all_emb, total)

        if checkpoint_path and checkpoint_path.exists():
            checkpoint_path.unlink()

        embeddings = np.array(all_emb, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms
        return embeddings

    @staticmethod
    def _save_checkpoint(path: Path | None, embeddings: list[list[float]], total: int) -> None:
        if not path or not embeddings:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, n_done=len(embeddings), total=total, embeddings=np.array(embeddings, dtype=np.float32))
        log.info("Checkpoint saved: %d/%d at %s", len(embeddings), total, path)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query."""
        from google.genai import types

        result = self.client.models.embed_content(
            model=self.model,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        vec = np.array(result.embeddings[0].values, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.reshape(1, -1)

    def submit_batch(self, texts: list[str]) -> dict:
        """Submit texts to Gemini Batch API. Returns handle for collect_batch()."""
        from google import genai

        total = len(texts)
        log.info("Gemini batch: submitting %d texts", total)

        rows = [
            {
                "key": str(i),
                "request": {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": "RETRIEVAL_DOCUMENT",
                },
            }
            for i, text in enumerate(texts)
        ]
        jsonl_path = _write_jsonl(rows)

        client = genai.Client()
        uploaded = client.files.upload(file=str(jsonl_path), config={"mime_type": "application/jsonl"})
        jsonl_path.unlink(missing_ok=True)
        log.info("Gemini batch: uploaded file %s", uploaded.name)

        batch_job = client.batches.create_embeddings(
            model=self.model, src={"file_name": uploaded.name},
        )
        log.info("Gemini batch: created job %s", batch_job.name)
        _notify(f"Gemini batch submitted: {batch_job.name}")

        return {"job_name": batch_job.name, "total": total}

    def collect_batch(self, handle: dict) -> np.ndarray:
        """Poll and download results from a submitted Gemini batch."""
        from google import genai

        job_name = handle["job_name"]
        total = handle["total"]
        client = genai.Client()

        _STATE_MAP = {
            "JOB_STATE_SUCCEEDED": "completed",
            "JOB_STATE_FAILED": "failed",
            "JOB_STATE_CANCELLED": "cancelled",
        }

        def poll() -> str:
            job = client.batches.get(name=job_name)
            state = str(job.state) if not isinstance(job.state, str) else job.state
            return _STATE_MAP.get(state, "in_progress")

        _poll_until_done(poll, label=f"Gemini batch {job_name}")

        job = client.batches.get(name=job_name)
        result_content = client.files.download(file=job.dest.file_name)
        if isinstance(result_content, bytes):
            result_content = result_content.decode()
        lines = result_content.strip().split("\n")

        emb_by_id: dict[int, list[float]] = {}
        for line in lines:
            rec = json.loads(line)
            idx = int(rec["key"])
            emb_by_id[idx] = rec["response"]["embedding"]["values"]

        if len(emb_by_id) != total:
            raise RuntimeError(f"Gemini batch: expected {total} results, got {len(emb_by_id)}")

        ordered = [emb_by_id[i] for i in range(total)]
        embeddings = np.array(ordered, dtype=np.float32)

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings /= norms
        return embeddings

    def embed_documents_batch(self, texts: list[str]) -> np.ndarray:
        """Embed via Gemini Batch API (50% discount, ≤24h). Blocking."""
        return self.collect_batch(self.submit_batch(texts))


class STEmbedder:
    """Local sentence-transformers embedder (e.g. Qwen3-Embedding-0.6B)."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", device: str = "mps"):
        from sentence_transformers import SentenceTransformer

        log.info("Loading ST embedding model: %s (device=%s)", model_name, device)
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 512
        self.dim = self.model.get_sentence_embedding_dimension()
        log.info("ST embedding model loaded (dim=%d, max_seq=%d)", self.dim, self.model.max_seq_length)

    def embed_documents(self, texts: list[str], batch_size: int = 32, **kwargs: object) -> np.ndarray:
        """Embed document chunks. Returns L2-normalized float32 array."""
        log.info("Embedding %d texts (batch_size=%d)", len(texts), batch_size)
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str, instruction: str = QUERY_INSTRUCTION) -> np.ndarray:
        """Embed a query with instruction prefix."""
        embedding = self.model.encode(
            query,
            prompt=f"Instruct: {instruction}\nQuery: ",
            normalize_embeddings=True,
        )
        return np.asarray(embedding, dtype=np.float32).reshape(1, -1)


def make_embedder(
    provider: str, model_name: str, device: str = "mps",
) -> VoyageEmbedder | OpenAIEmbedder | GeminiEmbedder | STEmbedder:
    """Factory: create embedder based on provider."""
    if provider == "voyage":
        return VoyageEmbedder(model_name)
    if provider == "openai":
        return OpenAIEmbedder(model_name)
    if provider == "gemini":
        return GeminiEmbedder(model_name)
    return STEmbedder(model_name, device)
