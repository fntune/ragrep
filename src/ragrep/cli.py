"""CLI entry point for ragrep."""

import argparse
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_PIPELINE_COMMANDS = frozenset({"scrape", "ingest", "query", "stats", "eval", "inspect"})
_PIPELINE_GLOBAL_OPTIONS = frozenset({"--config", "--log-level"})


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_scrape(args: argparse.Namespace) -> None:
    from ragrep.config import load_config

    config = load_config(args.config)
    raw_dir = config.raw_dir
    sources = args.source.split(",") if args.source else ["slack", "atlassian", "gdrive", "git", "bitbucket", "code"]

    for source in sources:
        source = source.strip()
        log.info("Scraping: %s", source)
        if source == "slack":
            from ragrep.ingest.scrape_slack import scrape
            scrape(raw_dir, config.scrape.slack)
        elif source == "atlassian":
            from ragrep.ingest.scrape_atlassian import scrape
            scrape(raw_dir, config.scrape.atlassian)
        elif source == "gdrive":
            from ragrep.ingest.scrape_gdrive import scrape
            scrape(raw_dir, config.scrape.gdrive)
        elif source == "git":
            from ragrep.ingest.scrape_git import scrape
            scrape(raw_dir, config.scrape.git)
        elif source == "bitbucket":
            from ragrep.ingest.scrape_bitbucket import scrape
            scrape(raw_dir, config.scrape.bitbucket)
        elif source == "code":
            from ragrep.ingest.scrape_code import scrape
            scrape(raw_dir, config.scrape.code)
        elif source == "files":
            from ragrep.ingest.extract import extract_all
            extract_all(raw_dir)
        else:
            log.error("Unknown source: %s", source)

    print(f"\nScrape complete. Raw data in {raw_dir}/")


def cmd_ingest(args: argparse.Namespace) -> None:
    from ragrep.config import load_config
    from ragrep.ingest.pipeline import ingest

    config = load_config(args.config)
    stats = ingest(config, force=args.force, source_filter=args.source)

    if stats.documents:
        print(f"\nIngestion complete: {stats.documents} docs → {stats.chunks} chunks in {stats.elapsed_s:.1f}s")
        for source, count in sorted(stats.sources.items()):
            print(f"  {source}: {count}")


def cmd_query(args: argparse.Namespace) -> None:
    from ragrep.config import load_config
    from ragrep.query.pipeline import QueryEngine

    config = load_config(args.config)
    engine = QueryEngine(config)

    if args.query:
        _run_query(engine, args.query, args.source, args.no_generate, args.verbose)
    else:
        # Interactive REPL
        print("ragrep interactive mode. Type a question, Ctrl+C to exit.\n")
        try:
            while True:
                question = input("Q: ").strip()
                if not question:
                    continue
                _run_query(engine, question, args.source, args.no_generate, args.verbose)
                print()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")


def _run_query(
    engine: object,
    question: str,
    source_filter: str | None,
    no_generate: bool,
    verbose: bool,
) -> None:

    result = engine.query(question, source_filter=source_filter, no_generate=no_generate)

    if result.answer:
        print(f"\nA: {result.answer}")

    if result.sources:
        print(f"\nSources ({len(result.sources)}):")
        for i, src in enumerate(result.sources, 1):
            line = f"  [{i}] {src.title} ({src.source})"
            if verbose:
                line += f" — dense={src.dense_score:.3f} bm25={src.bm25_score:.3f} rrf={src.rrf_score:.4f} rerank={src.rerank_score:.3f}"
            print(line)

    if verbose and result.timings:
        parts = [f"{k}={v:.2f}s" for k, v in result.timings.items()]
        print(f"\nTimings: {', '.join(parts)}")


def cmd_stats(args: argparse.Namespace) -> None:
    from ragrep.config import load_config
    from ragrep.ingest.store import index_exists, load_index

    config = load_config(args.config)
    index_dir = config.index_dir

    if not index_exists(index_dir):
        print(f"No index found at {index_dir}. Run 'make ingest' first.")
        return

    faiss_index, chunks, _ = load_index(index_dir)

    print(f"Index: {index_dir}")
    print(f"  Vectors: {faiss_index.ntotal}")
    print(f"  Chunks:  {len(chunks)}")

    # Source distribution
    sources: dict[str, int] = {}
    for chunk in chunks:
        sources[chunk.source] = sources.get(chunk.source, 0) + 1

    print("  Sources:")
    for source, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"    {source}: {count}")

    # Index file sizes
    for name in ("faiss.index", "chunks.pkl", "bm25.pkl"):
        path = index_dir / name
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            print(f"  {name}: {size_mb:.1f} MB")


def cmd_inspect(args: argparse.Namespace) -> None:
    import hashlib
    import json
    import random
    from collections import Counter

    import numpy as np

    from ragrep.config import load_config
    from ragrep.ingest.chunk import chunk_all
    from ragrep.ingest.normalize import normalize_all

    config = load_config(args.config)
    mode = args.mode

    if mode == "raw":
        raw_dir = config.raw_dir
        files = sorted(raw_dir.glob("*.jsonl"))
        print(f"{'file':<30} {'size':>8} {'records':>8}")
        print("-" * 50)
        for f in files:
            sz = f.stat().st_size
            if sz == 0:
                print(f"{f.name:<30} {'0':>8} {'0':>8}")
                continue
            n = sum(1 for _ in open(f) if _.strip())
            s = f"{sz/1048576:.1f}M" if sz > 1048576 else f"{sz/1024:.0f}K"
            print(f"{f.name:<30} {s:>8} {n:>8}")
        return

    # Load pipeline for docs/chunks/sample modes
    docs = normalize_all(config.raw_dir)
    chunks = chunk_all(docs, config.ingest.max_chunk_tokens, config.ingest.chunk_overlap_tokens)

    # Apply source filter
    if args.source:
        docs = [d for d in docs if d.source == args.source]
        chunks = [c for c in chunks if c.source == args.source]

    if mode == "docs":
        doc_sources = Counter(d.source for d in docs)
        chunk_sources = Counter(c.source for c in chunks)
        print(f"{'source':<12} {'docs':>8} {'chunks':>8} {'ratio':>6}")
        print("-" * 38)
        for src in sorted(doc_sources, key=lambda s: -doc_sources[s]):
            nd, nc = doc_sources[src], chunk_sources.get(src, 0)
            print(f"{src:<12} {nd:>8} {nc:>8} {nc/nd:>5.1f}x")
        print("-" * 38)
        print(f"{'TOTAL':<12} {len(docs):>8} {len(chunks):>8} {len(chunks)/len(docs):>5.1f}x")

        # Chunk size stats
        lens = [len(c.content) for c in chunks]
        toks = [max(len(c.content.split()), len(c.content) // 4) for c in chunks]
        print(f"\nchunk chars: min={min(lens)} med={int(np.median(lens))} p90={int(np.percentile(lens, 90))} max={max(lens)}")
        print(f"chunk tokens: min={min(toks)} med={int(np.median(toks))} p90={int(np.percentile(toks, 90))} max={max(toks)}")

        # Dedup
        hashes = set(hashlib.sha256(c.content.encode()).hexdigest() for c in chunks)
        print(f"\nunique hashes: {len(hashes)} / {len(chunks)} chunks ({len(chunks)-len(hashes)} dupes)")

    elif mode == "sample":
        n = args.n or 3
        if args.grep:
            pool = [c for c in chunks if args.grep.lower() in c.content.lower()]
            if not pool:
                print(f"No chunks matching '{args.grep}'")
                return
            sample = pool[:n]
        else:
            sample = random.sample(chunks, min(n, len(chunks)))

        for i, c in enumerate(sample):
            print(f"--- [{i+1}/{len(sample)}] {c.id} ---")
            print(f"source:   {c.source}")
            print(f"title:    {c.title}")
            print(f"doc_id:   {c.doc_id}")
            print(f"metadata: {json.dumps(c.metadata, default=str)}")
            content = c.content
            if args.full:
                print(f"content:  ({len(content)} chars)")
                print(content)
            else:
                print(f"content:  ({len(content)} chars, showing first 500)")
                print(content[:500])
                if len(content) > 500:
                    print("...")
            print()

    elif mode == "grep":
        if not args.grep:
            print("--grep required for grep mode")
            return
        matches = [(c, c.content.lower().find(args.grep.lower())) for c in chunks]
        matches = [(c, pos) for c, pos in matches if pos >= 0]
        print(f"{len(matches)} chunks match '{args.grep}'\n")
        sources = Counter(c.source for c, _ in matches)
        for src, cnt in sources.most_common():
            print(f"  {src}: {cnt}")
        n = args.n or 5
        if matches:
            print(f"\nTop {min(n, len(matches))} matches:")
            for c, pos in matches[:n]:
                start = max(0, pos - 40)
                end = min(len(c.content), pos + len(args.grep) + 40)
                snippet = c.content[start:end].replace("\n", " ")
                print(f"  [{c.source}] {c.title[:50]}  ...{snippet}...")


def cmd_eval(args: argparse.Namespace) -> None:
    from ragrep.config import load_config
    from ragrep.eval.harness import evaluate

    config = load_config(args.config)
    evaluate(config, output_path=args.output)


def main() -> None:
    """Pipeline CLI: ragrep scrape|ingest|query|stats|eval|inspect."""
    parser = argparse.ArgumentParser(prog="ragrep", description="Hybrid FAISS + BM25 RAG pipeline")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.toml (default: search CWD, RAGREP_CONFIG, ~/.config/ragrep/)")
    parser.add_argument("--log-level", default="INFO")

    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Build FAISS + BM25 index")
    p_ingest.add_argument("--force", action="store_true", help="Rebuild even if index exists")
    p_ingest.add_argument("--source", help="Only ingest this source type")

    # scrape
    p_scrape = sub.add_parser("scrape", help="Scrape data from sources")
    p_scrape.add_argument("--source", help="Comma-separated sources: slack,atlassian,gdrive,git,bitbucket,code,files")

    # query
    p_query = sub.add_parser("query", help="Query the knowledge base")
    p_query.add_argument("-q", "--query", help="Question (omit for interactive mode)")
    p_query.add_argument("--source", help="Filter to this source type")
    p_query.add_argument("--no-generate", action="store_true", help="Show retrieved docs without LLM")
    p_query.add_argument("--verbose", "-v", action="store_true", help="Show scores and timing")

    # stats
    sub.add_parser("stats", help="Show index statistics")

    # eval
    p_eval = sub.add_parser("eval", help="Run evaluation harness")
    p_eval.add_argument("--output", type=Path, help="Save results to file")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect raw data and pipeline output")
    p_inspect.add_argument("mode", choices=["raw", "docs", "sample", "grep"], help="Inspection mode")
    p_inspect.add_argument("--source", help="Filter to this source type")
    p_inspect.add_argument("--grep", help="Search string (for sample/grep modes)")
    p_inspect.add_argument("-n", type=int, help="Number of results to show")
    p_inspect.add_argument("--full", action="store_true", help="Show full chunk content")

    args = parser.parse_args()
    setup_logging(args.log_level)

    commands = {
        "scrape": cmd_scrape,
        "ingest": cmd_ingest,
        "query": cmd_query,
        "stats": cmd_stats,
        "eval": cmd_eval,
        "inspect": cmd_inspect,
    }
    commands[args.command](args)


# ---------------------------------------------------------------------------
# Spinner
# ---------------------------------------------------------------------------

class _Spinner:
    """Stderr spinner context manager."""

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, msg: str) -> None:
        import threading
        self._msg = msg
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "_Spinner":
        import sys
        import threading
        # Only spin if stderr is a TTY (skip in pipes/agent captures)
        if not sys.stderr.isatty():
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        import sys
        self._stop.set()
        if self._thread:
            self._thread.join()
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()

    def update(self, msg: str) -> None:
        self._msg = msg

    def _spin(self) -> None:
        import itertools
        import sys
        for frame in itertools.cycle(self._FRAMES):
            if self._stop.is_set():
                break
            sys.stderr.write(f"\r{frame} {self._msg}")
            sys.stderr.flush()
            self._stop.wait(0.08)


# ---------------------------------------------------------------------------
# Filter helpers
# ---------------------------------------------------------------------------

def _parse_date(s: str) -> str:
    """Parse 'YYYY-MM-DD' or relative '3m', '2w', '90d', '1y' to YYYY-MM-DD."""
    import re
    from datetime import date, timedelta

    m = re.fullmatch(r"(\d+)([dwmy])", s.strip().lower())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"d": timedelta(days=n), "w": timedelta(weeks=n),
                 "m": timedelta(days=n * 30), "y": timedelta(days=n * 365)}[unit]
        return (date.today() - delta).isoformat()
    date.fromisoformat(s)  # validate
    return s


def _parse_filters(raw: list[str]) -> dict[str, str]:
    """Parse 'key=value' strings into a dict."""
    filters: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"Invalid filter: {item!r} (expected key=value)")
        k, _, v = item.partition("=")
        filters[k.strip()] = v.strip()
    return filters


def _matches_filters(
    metadata: dict,
    filters: dict[str, str],
    after: str | None = None,
    before: str | None = None,
) -> bool:
    """Check if metadata matches all key=value filters + temporal range."""
    for key, val in filters.items():
        chunk_val = metadata.get(key)
        if chunk_val is None or val.lower() not in str(chunk_val).lower():
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


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def _snippet(content: str, length: int, term: str = "") -> str:
    """Extract a snippet from content, centered on term if found."""
    content = content.replace("\n", " ").strip()
    if length <= 0 or length >= len(content):
        return content
    if term:
        pos = content.lower().find(term.lower())
        if pos >= 0:
            start = max(0, pos - length // 4)
            end = min(len(content), start + length)
            s = content[start:end]
            return ("..." if start > 0 else "") + s + ("..." if end < len(content) else "")
    return content[:length] + ("..." if len(content) > length else "")


def _format_result(
    rank: int,
    chunk_id: str,
    source: str,
    title: str,
    content: str,
    scores: dict[str, float],
    metadata: dict,
    term: str,
    context: int,
    full: bool,
    as_json: bool,
    show_scores: bool = False,
    include_metadata: bool = False,
) -> str | dict:
    """Format a single search result for display or JSON."""
    if as_json:
        # Truncate title to 80 chars
        json_title = title if len(title) <= 80 else title[:77] + "..."

        # Start with essential fields
        rec: dict = {"rank": rank, "id": chunk_id, "source": source, "title": json_title}

        # Add scores only if requested (rounded to 3 decimals)
        if show_scores:
            rec.update({k: round(v, 3) for k, v in scores.items()})

        # Add content/snippet only if explicitly requested
        if full:
            rec["content"] = content
        elif context > 0:
            rec["snippet"] = _snippet(content, context, term)

        # Add metadata if requested
        if include_metadata:
            rec["metadata"] = metadata

        return rec

    # Human-readable (unchanged, scores rounded to 3 decimals)
    score_str = "  ".join(f"{k}={v:.3f}" for k, v in scores.items())
    lines = [f"  [{rank}] [{source}] {title}  {score_str}"]
    lines.append(f"      id: {chunk_id}")
    if full:
        lines.append(f"      {content}")
    elif context > 0:
        lines.append(f"      {_snippet(content, context, term)}")
    return "\n".join(lines)


def _search_and_print(
    config: object,
    mode: str,
    term: str,
    source: str | None,
    n: int,
    full: bool,
    context: int,
    as_json: bool,
    filters: dict[str, str] | None = None,
    after: str | None = None,
    before: str | None = None,
    show_scores: bool = False,
    include_metadata: bool = False,
) -> None:
    """Run search via search.py and print results."""
    import json

    from ragrep.search import search_grep, search_hybrid, search_semantic

    search_fn = {"grep": search_grep, "semantic": search_semantic, "hybrid": search_hybrid}[mode]
    label = {"grep": "Loading...", "semantic": "Searching...", "hybrid": "Searching..."}[mode]

    with _Spinner(label):
        result = search_fn(
            config=config, term=term, source=source, n=n,
            filters=filters, after=after, before=before,
            context=context, full=full, scores=show_scores, metadata=include_metadata,
        )

    if as_json:
        print(json.dumps(result, indent=2))
        return

    # Human-readable output
    items = result["results"]
    total = result.get("total_matches")
    if mode == "grep" and total is not None:
        print(f"{total} chunks match '{term}'\n")
    else:
        print(f"Top {len(items)} results for '{term}'\n")

    for item in items:
        score_parts = []
        for k in ("rerank", "rrf", "dense", "bm25", "score"):
            if k in item:
                score_parts.append(f"{k}={item[k]:.3f}" if isinstance(item[k], float) else f"{k}={item[k]}")
        score_str = "  ".join(score_parts)
        print(f"  [{item['rank']}] [{item['source']}] {item['title']}  {score_str}")
        print(f"      id: {item['id']}")
        if "content" in item:
            print(f"      {item['content']}")
        elif "snippet" in item:
            print(f"      {item['snippet']}")


# ---------------------------------------------------------------------------
# ragrep entry point
# ---------------------------------------------------------------------------

def _query_server(server_url: str, args: argparse.Namespace, filters: dict[str, str] | None, after: str | None, before: str | None) -> None:
    """Proxy search to the ragrep HTTP server."""
    import json
    import sys
    import urllib.error
    import urllib.parse
    import urllib.request

    params: dict[str, str] = {"q": args.term, "mode": args.mode, "n": str(args.n)}
    if args.source:
        params["source"] = args.source
    if filters:
        params["filter"] = ",".join(f"{k}={v}" for k, v in filters.items())
    if after:
        params["after"] = after
    if before:
        params["before"] = before
    if args.context:
        params["context"] = str(args.context)
    if args.full:
        params["full"] = "1"
    if args.scores:
        params["scores"] = "1"
    if args.metadata:
        params["metadata"] = "1"

    url = f"{server_url.rstrip('/')}/search?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url)  # noqa: S310

    # Add identity token for Cloud Run auth (if gcloud is available)
    if ".run.app" in server_url:
        try:
            import subprocess
            token = subprocess.check_output(
                ["gcloud", "auth", "print-identity-token"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
            if token:
                req.add_header("Authorization", f"Bearer {token}")
        except Exception:
            pass  # Fall through — server may allow unauthenticated

    # Cloud Run cold starts can take 2-3 min (downloads ~791 MB index from GCS)
    timeout = 180 if ".run.app" in server_url else 30
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 403:
            print(f"Error: 403 Forbidden from {server_url}", file=sys.stderr)
            print("Run: gcloud auth application-default login", file=sys.stderr)
            sys.exit(1)
        raise
    except (TimeoutError, urllib.error.URLError) as e:
        print(f"Error: Could not reach server at {server_url}: {e}", file=sys.stderr)
        print("The server may be cold-starting (first request after idle takes ~2 min).", file=sys.stderr)
        print("Try again in a moment, or use --server='' for local mode.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        items = result.get("results", [])
        total = result.get("total_matches")
        if result.get("mode") == "grep" and total is not None:
            print(f"{total} chunks match '{args.term}'\n")
        else:
            print(f"Top {len(items)} results for '{args.term}'\n")
        for item in items:
            score_parts = []
            for k in ("rerank", "rrf", "dense", "bm25", "score"):
                if k in item:
                    score_parts.append(f"{k}={item[k]:.3f}" if isinstance(item[k], float) else f"{k}={item[k]}")
            score_str = "  ".join(score_parts)
            print(f"  [{item['rank']}] [{item['source']}] {item['title']}  {score_str}")
            print(f"      id: {item['id']}")
            if "content" in item:
                print(f"      {item['content']}")
            elif "snippet" in item:
                print(f"      {item['snippet']}")


_SPLASH_LOGO = (
    "██████╗  █████╗  ██████╗ ██████╗ ███████╗██████╗ \n"
    "██╔══██╗██╔══██╗██╔════╝ ██╔══██╗██╔════╝██╔══██╗\n"
    "██████╔╝███████║██║  ███╗██████╔╝█████╗  ██████╔╝\n"
    "██╔══██╗██╔══██║██║   ██║██╔══██╗██╔══╝  ██╔═══╝ \n"
    "██║  ██║██║  ██║╚██████╔╝██║  ██║███████╗██║     \n"
    "╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝     "
)


def _print_splash() -> None:
    """First-run welcome screen: ASCII logo + setup guidance.

    Shown when ragrep is invoked with no args, or when the local environment
    has no index and no raw data to search (nothing to do yet).
    """
    import os
    import sys

    use_color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    def c(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    print()
    for line in _SPLASH_LOGO.splitlines():
        print("  " + c(line, "38;5;81"))
    print()
    print("  " + c("ripgrep for your team's knowledge base.", "1"))
    print("  " + c("hybrid retrieval · self-hosted · single command.", "2"))
    print()
    print("  " + c("Get started", "1;4"))
    print()
    print("  " + c("Point at a running server:", "2"))
    print('    export RAGREP_SERVER=http://your-server:8321')
    print('    ragrep "your question"')
    print()
    print("  " + c("Or build a local index:", "2"))
    print("    git clone https://github.com/fntune/ragrep && cd ragrep")
    print("    cp .env.example .env" + c("   # add VOYAGE_API_KEY, SLACK_TOKEN, etc.", "2"))
    print("    make install && make scrape && make ingest")
    print('    ragrep "your question"')
    print()
    print(f"  {c('Docs  ', '2')} https://ragrep.cc")
    print(f"  {c('Issues', '2')} https://github.com/fntune/ragrep/issues")
    print()


def grep() -> None:
    """ragrep <term> [-n 5] [-s git] [-m hybrid] [--full]"""
    import os
    import sys
    import time

    if len(sys.argv) == 1:
        _print_splash()
        return

    from ragrep.config import load_env_files

    load_env_files()

    parser = argparse.ArgumentParser(
        prog="ragrep",
        description="Search RAG chunks",
        epilog=(
            "modes: grep (substring), semantic (FAISS dense), hybrid (FAISS+BM25+rerank, default). "
            "pipeline commands: scrape, ingest, query, stats, eval, inspect."
        ),
    )
    parser.add_argument("term", help="Search string")
    parser.add_argument("-n", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("-s", "--source", help="Filter to source type")
    parser.add_argument("-m", "--mode", choices=["grep", "semantic", "hybrid"], default="hybrid", help="Search mode (default: hybrid)")
    parser.add_argument("-f", "--filter", action="append", default=[], metavar="KEY=VAL",
                        help="Metadata filter (repeatable, AND, substring match)")
    parser.add_argument("--after", help="On/after date: YYYY-MM-DD or relative (3m, 2w, 90d, 1y)")
    parser.add_argument("--before", help="Before date: YYYY-MM-DD or relative (3m, 2w, 90d, 1y)")
    parser.add_argument("-c", "--context", type=int, default=200, help="Snippet length in chars (default: 200, 0=none)")
    parser.add_argument("--full", action="store_true", help="Show full chunk content")
    parser.add_argument("--json", action="store_true", help="JSON output (for agents/scripts)")
    parser.add_argument("--scores", action="store_true",
                        help="Show scores in JSON output (rounded to 3 decimals, default: hidden)")
    parser.add_argument("--metadata", action="store_true",
                        help="Include metadata in JSON output")
    parser.add_argument("--server", default=os.environ.get("RAGREP_SERVER"),
                        help="Server URL (set RAGREP_SERVER env var or pass --server). Omit for local mode.")
    parser.add_argument("--config", type=Path, default=None,
                        help="Path to config.toml (default: search CWD, RAGREP_CONFIG, ~/.config/ragrep/)")

    args = parser.parse_args()
    setup_logging("WARN")

    # For JSON mode, default context to 0 (no snippet) unless explicitly set or --full
    if args.json and not args.full and "-c" not in sys.argv and "--context" not in sys.argv:
        args.context = 0

    # Parse filters and dates
    from ragrep.search import parse_date, parse_filters

    filters = parse_filters(args.filter) if args.filter else None
    try:
        after = parse_date(args.after) if args.after else None
        before = parse_date(args.before) if args.before else None
    except ValueError as e:
        parser.error(f"Invalid date: {e}")

    # Server mode: proxy to HTTP server
    if args.server:
        wall_start = time.perf_counter()
        _query_server(args.server, args, filters, after, before)
        wall = time.perf_counter() - wall_start
        print(f"\n({wall:.2f}s wall, server mode)", file=sys.stderr)
        return

    # Local mode: load index and search directly
    from ragrep.config import load_config
    from ragrep.ingest.store import index_exists

    config = load_config(args.config)
    mode = args.mode

    has_index = index_exists(config.index_dir)
    has_raw = config.raw_dir.exists() and any(config.raw_dir.iterdir())
    if not has_index and not has_raw:
        _print_splash()
        return

    if mode != "grep" and not has_index:
        print("No index found. Run 'make ingest' first. Falling back to grep.", file=sys.stderr)
        mode = "grep"

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    _search_and_print(
        config=config, mode=mode, term=args.term, source=args.source, n=args.n,
        full=args.full, context=args.context, as_json=args.json,
        filters=filters, after=after, before=before,
        show_scores=args.scores, include_metadata=args.metadata,
    )

    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    print(f"\n({wall:.2f}s wall, {cpu:.2f}s cpu)", file=sys.stderr)


def _is_pipeline_invocation(argv: list[str]) -> bool:
    """Return True when argv targets the scrape/ingest/query command surface."""
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in _PIPELINE_COMMANDS:
            return True
        if arg == "--":
            return False
        if arg in _PIPELINE_GLOBAL_OPTIONS:
            i += 2
            continue
        if any(arg.startswith(f"{option}=") for option in _PIPELINE_GLOBAL_OPTIONS):
            i += 1
            continue
        return False
    return False


def entrypoint() -> None:
    """Installed CLI entry point.

    `ragrep "term"` is the fast search surface. `ragrep scrape|ingest|...`
    remains the pipeline surface documented by the README and Makefile.
    """
    import sys

    if _is_pipeline_invocation(sys.argv[1:]):
        main()
        return
    grep()


if __name__ == "__main__":
    entrypoint()
