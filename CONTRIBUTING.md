# Contributing to ragrep

Thanks for taking a look. This is a small project — issues and PRs are welcome, especially for new scrapers, retrieval tuning, and rough edges in the CLI.

## Setup

```bash
git clone https://github.com/fntune/ragrep
cd ragrep
uv sync --extra full --extra dev
source .venv/bin/activate
```

You'll need Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). The `full` extra pulls FAISS, sentence-transformers, and torch (~2 GB). Skip it if you only want to work on the CLI client or server (`uv sync --extra dev`).

Copy `.env.example` to `.env` and fill in any API keys you need for the components you're working on. Most code paths don't require all of them.

## Run the checks

```bash
make check   # ruff lint + format + mypy
make test    # pytest
```

PRs should leave `make check` no worse than they found it. There's currently a baseline of pre-existing lint findings (subprocess hygiene, urllib hardening) — not a blocker for new work, just don't add new ones.

## Conventions

- **Style**: ruff handles formatting (`make check` runs it). Line length 120, `"`-strings, type hints on function signatures.
- **No `from __future__ import annotations`.** Drop it if you see it.
- **Imports**: stdlib, then third-party, then local. Explicit imports — no `__init__.py` re-export tricks.
- **Comments**: default to none. Add a comment only when the WHY isn't obvious from the code (workarounds, surprising constraints, hidden invariants). A comment explaining a bad name should become a better name.
- **Errors**: fail fast at boundaries (API, env, IO). Don't swallow exceptions. Don't add fallback logic for hypothetical cases.

## Where things live

```
src/ragrep/
  cli.py         CLI entry points (search, scrape, ingest, eval, ...)
  server.py     FastAPI HTTP server
  config.py     Config loading + .env discovery
  index.py      Public Python SDK
  ingest/       scrape → normalize → chunk → embed → store
  query/        retrieve → rerank → generate
  eval/         entity-graph eval harness
docs/           landing page (https://ragrep.cc) + install.sh
```

## Adding a scraper

1. Create `src/ragrep/ingest/scrape_<source>.py` with a `scrape(raw_dir, config)` function.
2. Wire it into `cmd_scrape` in `src/ragrep/cli.py`.
3. Add the source to `ScrapeConfig` in `src/ragrep/config.py`.
4. Document required env vars in `.env.example`.
5. Add a row to the **Data sources** table in `README.md`.

## Filing issues

For bugs: include `ragrep --version`, the command you ran, and the full error output.

For feature requests: state the use case and what you'd expect the CLI / SDK to look like — concrete examples beat abstract feature lists.
