"""Parity test: Rust ragrep vs Python ragrep on the same index.

Runs N queries through both binaries in --json mode, extracts top-10 chunk
IDs, and reports per-query overlap. Exits 1 if any query falls below the
overlap threshold.
"""

import argparse
import json
import subprocess
import sys
import time

# Mix of single-word, multi-word, natural-language, and identifier-style queries.
# Intentional overlap with realistic queries someone would actually type.
QUERIES = {
    "grep": [
        "auth", "login", "deploy", "incident", "miner",
        "airflow", "syncer", "openclaw", "feature flag", "api key",
    ],
    "semantic": [
        "how does the auth flow work",
        "deploy",
        "incident response",
        "what's the rate limit story",
        "embeddings cache",
        "feature flags",
        "OAuth setup",
        "BM25 vs dense retrieval",
        "memory mapping",
        "single binary distribution",
    ],
    "hybrid": [
        "how does the auth flow work",
        "what's the deploy process",
        "incident postmortem",
        "scrape Slack",
        "FAISS index format",
        "dedup hash",
        "voyageai SDK",
        "Cloud Run identity token",
        "rate limit backoff",
        "ASCII splash",
    ],
}


def run_query(binary, term, mode, n=10):
    """Returns the parsed JSON output, or None on failure."""
    args = [*binary, term, "-m", mode, "-n", str(n), "--json"]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        return None, proc.stderr.strip()
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError as e:
        return None, f"json decode: {e}; stdout starts with: {proc.stdout[:200]!r}"


def top_ids(payload, n):
    return [r["id"] for r in payload["results"][:n]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rust", default="./rust/target/release/ragrep")
    p.add_argument("--python", nargs="+", default=["uv", "run", "ragrep"])
    p.add_argument("--mode", choices=["grep", "semantic", "hybrid", "all"], default="all")
    p.add_argument("--n", type=int, default=10, help="top-n to compare")
    p.add_argument("--threshold", type=float, default=0.9, help="min top-n overlap (0..1)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    rust = [args.rust]
    python = args.python

    modes = ["grep", "semantic", "hybrid"] if args.mode == "all" else [args.mode]
    failures = []
    started = time.monotonic()

    for mode in modes:
        print(f"\n=== {mode} ({len(QUERIES[mode])} queries, n={args.n}) ===")
        for q in QUERIES[mode]:
            r_pay, r_err = run_query(rust, q, mode, args.n)
            p_pay, p_err = run_query(python, q, mode, args.n)
            if r_err or p_err:
                msg = f"  ERROR: {q!r}  rust={r_err}  python={p_err}"
                print(msg)
                failures.append((mode, q, msg))
                continue
            r_ids = top_ids(r_pay, args.n)
            p_ids = top_ids(p_pay, args.n)
            overlap = len(set(r_ids) & set(p_ids)) / max(len(p_ids), 1)
            ok = overlap >= args.threshold
            mark = "OK" if ok else "FAIL"
            print(f"  [{mark}] {q!r:50s}  overlap={overlap:.0%}")
            if args.verbose or not ok:
                missing_in_rust = [i for i in p_ids if i not in r_ids]
                extra_in_rust = [i for i in r_ids if i not in p_ids]
                if missing_in_rust:
                    print(f"        in python only: {missing_in_rust}")
                if extra_in_rust:
                    print(f"        in rust only:   {extra_in_rust}")
            if not ok:
                failures.append((mode, q, f"overlap={overlap:.0%}"))

    elapsed = time.monotonic() - started
    total = sum(len(QUERIES[m]) for m in modes)
    print(f"\n--- {total - len(failures)}/{total} pass, {elapsed:.1f}s wall ---")
    if failures:
        print("Failures:")
        for mode, q, msg in failures:
            print(f"  {mode}: {q!r}  {msg}")
        sys.exit(1)


if __name__ == "__main__":
    main()
