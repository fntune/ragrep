# Full Rust rewrite of ragrep

## Context

`ragrep` today is ~6,800 LOC of Python across 28 files, distributed via `curl ragrep.cc/install.sh | sh` which calls `uv tool install ragrep[full] @ git+...`. The user just shipped the public release. Two pain points motivate this rewrite:

1. **Startup latency.** Even after the recent `voyageai` SDK → `httpx` swap that cut cold-import from ~20s to ~110ms, every Python invocation pays ~100ms+ of import overhead before the first useful instruction. A native binary starts in <10ms.
2. **Single-binary distribution.** Today users need `uv` (which installs Python, which sets up a venv) before `ragrep` exists on PATH. A static binary skips all of that — same shape as ripgrep, ruff, uv, bun, deno.

The user picked **Rust**, **full rewrite**, motivated by **startup latency + single-binary distribution**. The `ragrep` name on crates.io is taken (active, v0.2.0, Nov 2025) so we ship via the vanity-domain installer, not `cargo install`. The branding alignment with `ripgrep`/`ruff`/`uv`/`tantivy` is a bonus.

This plan is the rewrite. It's substantial — ~10–12 weeks part-time — so it's phased to ship value early (Phase 1 = working CLI binary against an existing Python-built index) and let ingest catch up later.

## On-disk reality (what we're migrating from)

```
data/index/  →  1.3 GB total, 103,377 chunks × 1024-d float32 (L2-normalized)
  faiss.index    404 MB    FAISS native binary           NOT directly used by Rust (see pivot below)
  chunks.pkl     203 MB    Python pickle of list[Chunk]  → chunks.msgpack
  bm25.pkl       185 MB    Python pickle of BM25Okapi    → drop, rebuild from chunks
  embed_cache.pkl 485 MB   Python pickle of {sha256: vec} → embed_cache.msgpack
```

Spike outcome (already known): we are **not** using the `faiss` crate. See "FAISS-rs pivot" below. The migration script extracts embeddings from `faiss.index` via `faiss.reconstruct_n()` and writes them as a raw f32 array to a new file, then drops the FAISS dependency entirely.

### Migration target layout

```
data/index/  →  ~625 MB total (down from 1.3 GB; embed_cache is the same size, just msgpack)
  embeddings.bin    423 MB   103377 × 1024 × 4 bytes  raw f32, mmap-ready, no header
  chunks.msgpack    ~200 MB  Vec<Chunk>, position i ↔ embeddings row i
  bm25.msgpack      ~200 MB  hand-rolled Okapi (term postings + doc lengths + IDF)
  embed_cache.msgpack ~485 MB HashMap<[u8; 32], Vec<f32>> for ingest dedup
```

The FAISS `.index` file is not in the runtime layout. Migration reads it once, writes `embeddings.bin`, then it's no longer needed (users can keep it for rollback).

### FAISS-rs pivot (decided in this planning session)

The original plan said "FAISS via faiss = 0.13". Two reasons we dropped it:

1. **`faiss-rs` only supports Faiss v1.7.2 (custom `c_api_head` fork).** Our existing index was written by `faiss-cpu >= 1.9.0`. Format compatibility for `IndexFlatIP` is *probably* fine (header + raw float array), but unverified.
2. **`libfaiss_c` build is heavy.** Either system install (CMake + BLAS, the maintainer's fork only) or `static` feature (CMake + BLAS at build time, ~50MB binary bloat). Cross-compile gets ugly per target.

**Replacement: hand-rolled flat brute-force search over a memory-mapped raw f32 array.**

- Workload sizing: 103k vectors × 1024 dims × 4 dot-products-per-element = 105M FLOPs per query.
- Pure Rust with explicit SIMD via the `wide` crate (f32x8 on AVX2 / NEON via stdsimd-fallbacks): ~3–5ms per query on Apple Silicon.
- mmap via `memmap2` means cold start is <1ms (kernel page table only; pages fault in lazily).
- We control the on-disk format → consistent with chunks.msgpack, no foreign binary format.

When we cross ~1M chunks (and HNSW becomes worth it), the swap-in is `arroy` (pure Rust, mmap-first, used by Meilisearch). Until then, brute force is faster end-to-end (no index-build cost, no recall loss).

## Module layout

**Single crate, one binary, subcommand surface.** The path is the namespace; nothing in the tree repeats `ragrep`.

```
ragrep/
├── Cargo.toml                       # one crate, one [[bin]] name = "ragrep"
├── src/
│   ├── main.rs                      # clap entry; dispatches to cli::*
│   ├── lib.rs                       # module declarations
│   ├── splash.rs
│   ├── models.rs                    # shared: Document, Chunk, SearchResult, QueryResult
│   ├── config.rs                    # shared: Config, load, env + XDG search
│   ├── embed/                       # shared capability (ingest batches; query one-shots)
│   │   ├── mod.rs                   # Embedder trait + factory
│   │   ├── voyage.rs
│   │   ├── openai.rs
│   │   ├── gemini.rs
│   │   ├── cache.rs                 # content-hash → vec
│   │   └── throttle.rs              # adaptive rate-limit throttle
│   ├── index/                       # on-disk persistence (both sides touch)
│   │   ├── mod.rs
│   │   ├── store.rs                 # load/save chunks.msgpack + embeddings.bin + bm25.msgpack
│   │   ├── flat.rs                  # mmapped brute-force IP search (the FAISS replacement)
│   │   └── bm25.rs                  # hand-rolled Okapi data structure
│   ├── ingest/                      # write side
│   │   ├── mod.rs
│   │   ├── pipeline.rs              # scrape → normalize → chunk → embed → store
│   │   ├── chunk.rs                 # 512-token chunking
│   │   ├── extract.rs               # gemini multimodal HTTP for PDF/DOCX/PPTX/XLSX
│   │   ├── normalize/
│   │   │   ├── mod.rs
│   │   │   ├── slack.rs
│   │   │   ├── atlassian.rs
│   │   │   ├── gdrive.rs
│   │   │   ├── git.rs
│   │   │   ├── bitbucket.rs
│   │   │   ├── code.rs
│   │   │   └── file.rs
│   │   └── scrape/
│   │       ├── mod.rs
│   │       ├── slack.rs
│   │       ├── atlassian.rs
│   │       ├── gdrive.rs
│   │       ├── git.rs
│   │       ├── bitbucket.rs
│   │       └── code.rs
│   ├── query/                       # read side
│   │   ├── mod.rs                   # pub fn grep, fn semantic, fn hybrid (public engine API)
│   │   ├── retrieve.rs              # dense + bm25 + rrf fusion
│   │   ├── rerank.rs                # voyage HTTP rerank
│   │   └── generate.rs              # ollama HTTP
│   ├── eval/
│   │   ├── mod.rs
│   │   ├── entity.rs                # Jira-ID extraction
│   │   └── harness.rs               # source_recall, entity_recall, MRR
│   ├── serve/                       # HTTP server (backs `ragrep serve`)
│   │   ├── mod.rs                   # axum router
│   │   ├── auth.rs                  # Cloud Run JWT middleware
│   │   └── search.rs                # GET /search handler
│   └── cli/                         # subcommand dispatchers (user boundary)
│       ├── mod.rs                   # Subcommand enum + dispatch
│       ├── search.rs                # `ragrep <term>` default
│       ├── serve.rs                 # `ragrep serve`
│       ├── ingest.rs
│       ├── scrape.rs
│       ├── eval.rs
│       ├── stats.rs
│       ├── inspect.rs
│       └── rebuild_bm25.rs          # transient bridge for pickle-migrated indexes
├── tools/
│   └── migrate.py                   # one-shot pickle → msgpack
└── docs/                            # unchanged: install.sh, index.html, fonts/
```

### How paths read as namespaces

Read vs write vs shared is legible from the tree alone:

- `embed::voyage` — shared capability. `Embedder`, `embed`, `embed_query`.
- `index::store::load` — on-disk, both sides touch.
- `index::bm25::Okapi` — the BM25 data structure type.
- `ingest::scrape::slack::run` — write-side, API fetch, source-specific.
- `ingest::normalize::slack::normalize` — write-side, raw JSON → `Document`.
- `ingest::chunk::all` — write-side step.
- `ingest::pipeline::run` — write-side orchestrator.
- `query::hybrid` — public engine API; top-level in `query::`.
- `query::retrieve::dense`, `query::retrieve::rrf` — retrieval building blocks.
- `query::rerank::voyage::rerank` — rerank building block.
- `cli::ingest::run` — CLI subcommand handler.
- `serve::search::handle` — HTTP endpoint handler.
- `eval::entity::extract` — entity extraction for eval.

No concept appears twice in any full path. The write/read/shared split is inferable without opening a single file.

### User-facing surface

One binary, clap subcommands (mirrors ripgrep/cargo/uv):

```
ragrep                              # welcome splash (no args)
ragrep <term>                       # search (default, implicit subcommand)
ragrep <term> -m grep -n 5
ragrep serve --port 8321            # HTTP server (replaces FastAPI)
ragrep ingest [--force]
ragrep scrape --source slack,git
ragrep eval
ragrep stats
ragrep inspect raw
ragrep rebuild-bm25                 # one-off bridge for pickle-migrated indexes
```

### Why single crate

The case for a workspace was compile-time isolation — the CLI binary shouldn't link axum. Resolution: feature flags in a single crate if compile time becomes a real complaint. It won't for v1 (total Rust LOC lands around 15k; whole-crate `cargo check` stays under 10s on modern hardware). A workspace would earn its keep when another project depends on `ragrep-retrieve` as a library. No such demand today.

### Taste exceptions taken

- `cli/` as a directory repeats the project identity (ragrep is a CLI). Alternative was putting subcommand dispatchers flat at `src/{search,serve,ingest,...}.rs`, which collides with the engine modules of the same names. `cli/` earns the exception as a boundary between "user-facing subcommand handlers" and "engine building blocks those handlers call."
- `tools/migrate.py` stays Python because it runs inside the user's existing venv to read pickle format. Cross-language script in the tools/ boundary is correct.

## Library picks (and why)

| Capability         | Crate                          | Notes                                                                                     |
|--------------------|--------------------------------|-------------------------------------------------------------------------------------------|
| Vector search      | hand-rolled (~150 LOC)         | Brute-force IP over mmapped raw f32 array. SIMD via `wide` crate. See "FAISS-rs pivot"   |
| Memory mapping     | `memmap2`                      | Maps `embeddings.bin` to `&[u8]`; we cast to `&[f32]` via `bytemuck::cast_slice`          |
| Cast / SIMD        | `bytemuck`, `wide`             | `bytemuck` for safe `&[u8]→&[f32]`; `wide` for portable f32x8 (NEON/AVX2 autodispatch)    |
| BM25               | hand-rolled (~200 LOC)         | Mirrors `rank_bm25.BM25Okapi`. Avoids the only-half-maintained crate options              |
| HTTP client        | `reqwest` (async, rustls TLS)  | Voyage embed/rerank, Slack, Atlassian, Bitbucket, Gemini, Ollama                          |
| Async runtime      | `tokio`                        | Multi-threaded scheduler for scrapers; CLI uses `current_thread`                          |
| CLI                | `clap` (derive)                | Standard. Matches the existing argparse surface                                           |
| HTTP server        | `axum`                         | Modern tokio-native. Replaces FastAPI                                                     |
| Serialization      | `serde` + `rmp-serde` + `serde_json` | msgpack for on-disk chunks/embed-cache; JSON for API I/O                              |
| Config             | `toml = "0.8"`                 | Reads existing `config.toml` 1:1                                                          |
| Logging            | `tracing` + `tracing-subscriber` | Structured + level filtering                                                              |
| Tokenization       | `tiktoken-rs`                  | For 512-token chunking. cl100k_base is close enough to Voyage's tokenizer for chunk sizing |
| Git scraping       | `git2`                         | libgit2 bindings. Reads commit history without subprocess                                 |
| GDrive auth        | `yup-oauth2`                   | Service-account JSON or installed-app OAuth                                               |
| .env loading       | `dotenvy`                      | Loads `./` and `~/.config/ragrep/.env`                                                    |
| Ctrl-C handling    | `ctrlc`                        | Graceful shutdown for long ingest runs                                                    |

## Scope cut: local model providers (regression flagged)

Current Python supports `provider = "sentence-transformers"` for both embedding and reranking (loads Qwen3-Embedding-0.6B / Qwen3-Reranker-0.6B locally via PyTorch). **v1 Rust drops this**: HTTP-only providers (Voyage, OpenAI, Gemini for embed; Voyage for rerank).

Reason: replacing it requires `candle` or `ort` + ONNX-converted models + GPU/MPS feature detection. That's 3+ weeks of work for a feature only used when users have no API keys. Better to ship v1 without it and add v2 as a feature flag.

Migration note for affected users: switch `provider = "voyage"` in `config.toml` and add `VOYAGE_API_KEY` to `.env`. Voyage gives 200M tokens free/month.

## Phased delivery

### Phase 0 — flat-search spike + scaffolding (1 week)

**Step 0.1 — flat-search spike (1 day, do first).**

The spike validates the FAISS-pivot decision against real data before any other Rust code is written. The risk being de-risked: "can we mmap a 423MB f32 array, brute-force-search it from Rust, and get the same top-10 chunks Python's FAISS returns?"

Concrete steps:
1. `tools/extract_embeddings.py` — read `data/index/faiss.index`, call `faiss.reconstruct_n(0, ntotal)`, write the resulting (103377, 1024) f32 array as raw bytes to `data/index/embeddings.bin`. ~10 LOC. Verify byte count = 103377 × 1024 × 4 = 423,329,792.
2. Minimal Rust crate at repo root with only what the spike needs:
   - `Cargo.toml` deps: `memmap2`, `bytemuck`, `wide`, `serde`, `rmp-serde`, `clap` (for the spike's CLI args)
   - `src/main.rs` — takes a `--query-vec-from <path-to-json-of-1024-floats>` arg (or hardcodes one extracted from Python for the parity check)
   - `src/index/flat.rs` — mmap `embeddings.bin`, expose `fn search(query: &[f32], top_k: usize) -> Vec<(u32, f32)>` (returns chunk index + score)
3. Parity check: run the same query embedding through Python (`faiss_index.search(...)`) and through the Rust spike. Top-10 chunk indices must match exactly (same metric, same normalized vectors → same scores → same ranking).
4. Bench: time 10 query searches in Rust. Target <10ms per search on the user's M-series Mac. If we're within 2× of FAISS Python (which uses native libfaiss SIMD), the pivot is validated.

**Pass criteria for Step 0.1:**
- Top-10 indices match Python FAISS for ≥3 different query embeddings (drawn from `embed_query("auth")`, `embed_query("deploy")`, `embed_query("incident")`).
- Per-query latency <10ms warm.
- mmap load (`Mmap::map(&file)`) returns in <2ms for the 423MB file.

**If spike fails:**
- Indices don't match → bug in our IP search loop. Fix; re-verify. Likely culprit: wrong stride, wrong byte order, missed normalization.
- Latency >>10ms → SIMD isn't kicking in. Try `wide::f32x8` explicitly, or check if release build with `target-cpu=native` is on. Worst case fallback: `usearch` with mmap.
- If still failing after 1 day: stop, escalate to user, reconsider.

**Step 0.2 — scaffolding (rest of week).**

Only run after Step 0.1 passes:
- Flesh out `Cargo.toml` with the full dep list (clap derive, tokio, reqwest, axum, serde, rmp-serde, toml, tracing, dotenvy, ctrlc, tiktoken-rs, git2, yup-oauth2)
- Empty stubs for every module in the `src/` tree per the layout above. Each stub compiles (so `cargo check` is green from day one).
- Clap subcommand skeleton: every `cli::*::run` exists, takes its args, prints "not implemented" and exits 1. `ragrep --help` shows the full subcommand tree.
- GitHub Actions matrix: `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`, `x86_64-apple-darwin`, `aarch64-unknown-linux-gnu`, `x86_64-pc-windows-msvc`. Each builds a stripped release binary and uploads as a workflow artifact.
- Updated `docs/install.sh`: detects platform via `uname`, downloads matching binary from GitHub Releases, verifies sha256, installs to `~/.local/bin/ragrep`. `--legacy` flag falls back to `uv tool install` for the transition window.

**Verify (Phase 0 ship gate):**
- `cargo check` is green for the whole tree.
- `cargo build --release` produces `target/release/ragrep`. `./target/release/ragrep --help` returns in <50ms wall, lists all subcommands.
- `cross build --target $TGT` succeeds for all 5 targets in CI.
- The flat-search spike binary still passes its parity check against Python.

### Phase 1 — query path (2 weeks)
- `models` — serde mirrors of `Document`, `Chunk`, `SearchResult`, `QueryResult`
- `config` — reads `config.toml`, env file + XDG search (mirror of `config.py`)
- `index::bm25` — hand-rolled Okapi data structure
- `index::flat` — promote spike code to its module home; add multi-model variant (one mmap per `provider--model_name`)
- `index::store::load` — `chunks.msgpack` (rmp-serde) + `embeddings.bin` (mmap) + `bm25.msgpack` (rmp-serde). Single `load` returns `(FlatIndex, Vec<Chunk>, BM25)`.
- `query::retrieve` — `fn dense` (calls `index::flat::search`), `fn bm25` (calls `index::bm25::score`), `fn rrf`, `fn multi_signal_rrf`
- `query::rerank::voyage` — reqwest port of the recently-ported httpx rerank call
- `embed::{voyage, openai, gemini}` — HTTP only, no SDK
- `query::{grep, semantic, hybrid}` — public engine API, top-level fns in `query/mod.rs`
- `cli::search::run` — clap args mirror argparse 1:1 (`-n`, `-s`, `-m`, `-f KEY=VAL`, `--after`, `--before`, `-c`, `--full`, `--json`, `--scores`, `--metadata`, `--server`, `--config`)
- `splash` — port the ASCII splash; same trigger conditions
- `cli::search::proxy` — port `_query_server`, including Cloud Run identity-token via `gcloud auth print-identity-token`
- `tools/migrate.py` — reads existing `chunks.pkl` and `embed_cache.pkl`, writes msgpack. Drops `bm25.pkl` (rebuild handled below)
- `cli::rebuild_bm25::run` — one-time helper; loads `chunks.msgpack`, builds BM25, writes `bm25.msgpack`. Transitional, goes away once Phase 2 ingest can do this.

**Verify (parity test)**: pick 50 representative queries (mix of grep, semantic, hybrid; mix of source filters and date filters). Run them through Python ragrep, capture top-10 chunk IDs. Run the same queries through Rust ragrep against the same index. Top-10 sets match exactly for grep, ≥9/10 overlap for semantic/hybrid (floating-point drift). Script: `tools/parity.sh`.

**Ship gate**: end of Phase 1, the binary is usable against an existing Python-built index. Release as `v0.2.0`; users get the startup-latency win immediately.

### Phase 2 — ingest pipeline (3 weeks, the write side)

Phase 1 shipped the read side (search modes + filters + server proxy + 30/30 parity).
Phase 2 closes the loop by porting the Python ingest pipeline so a deployed Rust
binary can scrape → normalize → chunk → embed → store from scratch.

What Phase 1 already left in place that Phase 2 builds on:
  `models`, `config`, `index::flat` (mmap read), `index::bm25` (build + save),
  `index::store::{save_chunks, save_bm25}` (atomic write done), `embed::voyage`
  (sync HTTP one-shot port), `query::filters`, `cli::run` dispatch.

Missing pieces (in build order):
  `embed::Embedder`      trait + factory; 3 provider impls (voyage / openai / gemini)
  `embed::cache`         binary `embed_cache/{provider}--{model}.bin`, mmap-friendly
  `embed::throttle`      port `_AdaptiveThrottle` from `embed.py:39`
  `embed::voyage::batch` (+ `openai::batch`, `gemini::batch`) — sync HTTP, retry/backoff
  `ingest::chunk`        port `chunk.py` (no real tokenizer; see Tokenization note)
  `ingest::normalize/*`  9 source-specific modules, port `normalize.py` (526 LOC)
  `ingest::pipeline`     orchestrator, mirrors `pipeline.py::ingest`
  `cli::ingest::run`     `ragrep ingest [--force] [--source X]`

#### Step 2.1a — `embed::Embedder` trait + factory (~½ day)

Design the embedding layer so providers and models are swappable. The pipeline
holds a `Box<dyn Embedder>` and never knows whether it's talking to Voyage,
OpenAI, or Gemini. `cli::ingest::run` reads `cfg.embedding.{provider, model_name}`
and calls the factory.

```rust
// embed/mod.rs
pub trait Embedder: Send + Sync {
    fn provider(&self) -> &str;
    fn model(&self) -> &str;
    fn dim(&self) -> usize;
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
    fn embed_documents(
        &self,
        texts: &[&str],
        batch_size: usize,
        checkpoint: Option<&Path>,
    ) -> Result<Vec<Vec<f32>>>;
}

pub fn make(provider: &str, model: &str) -> Result<Box<dyn Embedder>> { ... }
```

Phase 1 already has the `embed_query` half ported (`embed::voyage::embed_query`).
Phase 2 lifts that into the trait alongside `embed_documents`. The current
`embed::embed_query` free function (used by `cli::search::run`) becomes a thin
wrapper over `make(...).embed_query(...)`.

#### Step 2.1b — `embed::cache` (~1 day)

Per-(provider, model) cache file under `data/index/embed_cache/`. Compact binary
format (50 % smaller, 2× faster load than msgpack-of-f64-lists):

```
embed_cache/{provider}--{model}.bin layout:
  [0..4]    u32 magic   = 0x52414331  ("RAG1")
  [4..8]    u32 version = 1
  [8..12]   u32 dim
  [12..20]  u64 n_entries
  [20..]    n × (32 bytes sha256 || dim × 4 bytes f32, little-endian)
```

API:
- `pub struct Cache { provider: String, model: String, dim: usize, entries: HashMap<[u8;32], Vec<f32>> }`
- `pub fn hash(text: &str) -> [u8;32]` (`sha2::Sha256::digest`)
- `pub fn load(dir: &Path, provider: &str, model: &str) -> Result<Cache>` — returns empty cache if file absent
- `pub fn save(&self, dir: &Path) -> Result<()>` — atomic via tempfile + rename

Bootstrap path: if no cache file exists for this `(provider, model)` but
`embeddings.bin` + `chunks.msgpack` exist (and `cfg.embedding.{provider,
model_name}` matches what built that index), walk them and seed the cache.
Mirrors Python's `_load_embed_cache` fallback in `pipeline.py:27`. One-shot
transition aid for users coming from v0.2.

Read perf: 500 MB binary file load is ~2 s on SSD with mmap + `bytemuck::
cast_slice` (zero decode cost). msgpack equivalent was ~5 s + 2× memory.

`tools/migrate.py` updates: write `embed_cache/voyage--voyage-code-3.bin` in
the new format (replaces the existing `embed_cache.msgpack` write).

#### Step 2.2 — `embed::throttle` (~1 day)

Port `_AdaptiveThrottle` from `src/ragrep/ingest/embed.py:39` exactly:
  initial 30s, min 0.2s, max 90s, 0.75× decay after 3-streak success (clamped to
  floor), 1.5× backoff on rate-limit (or `retry-after + 2s`), floor remembered =
  current delay, floor erodes 0.85× every 10 successes within 1.1× of floor.
  3 unit tests covering decay / backoff / floor erosion transitions.

#### Step 2.3 — Three provider impls of `Embedder::embed_documents` (~3 days)

Each provider gets its own `embed::{voyage,openai,gemini}::Embedder` struct
that implements the trait. The HTTP shapes differ but the loop pattern is the
same — extract a private `_embed_batch_loop` helper if duplication gets noisy.

Common loop shape:
  - Iterate `texts.chunks(batch_size)` (provider-specific cap, see table)
  - Sync HTTP via `reqwest::blocking`
  - Throttle between batches via `embed::throttle`
  - HTTP 429: parse `Retry-After`, `throttle.on_rate_limit(retry_after)`,
    sleep, retry up to 5×
  - Checkpoint every 500 batches: `embeddings_checkpoint.bin`
    (`u32 n_done + u32 total + n_done × dim × f32`). Resume on restart if
    total matches. Format is Rust-internal (process-local).
  - On success, delete the checkpoint

Per-provider specifics:

| Provider | Endpoint                                   | Batch cap | Key envs           | Notes                                              |
|----------|--------------------------------------------|-----------|---------------------|----------------------------------------------------|
| voyage   | POST `api.voyageai.com/v1/embeddings`      | 16        | `VOYAGE_API_KEY`    | `input_type=document`. Tight rate limits → throttle is load-bearing. |
| openai   | POST `api.openai.com/v1/embeddings`        | 64        | `OPENAI_API_KEY`    | Higher rate limits. `dimensions` query param for `text-embedding-3-large` (which is 3072 by default; allow 1024 for compat with our `dim` slot). |
| gemini   | POST `generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents` | 100 | `GEMINI_API_KEY` | Different request shape; per-instance task_type. |

Output normalization: all 3 providers emit L2-normalized f32 vectors. If a
provider returns un-normalized (rare), normalize at the boundary so the rest
of the pipeline can assume `‖v‖ = 1`.

Tests per provider: integration test using a recorded HTTP fixture (or
`mockito`) that asserts the trait's contract: identical query → identical
embedding bytes (to f32 precision).

#### Step 2.4 — `ingest::chunk` (~1–2 days)

Port `chunk.py` exactly — including the heuristic `_approx_tokens = max(word_count, len // 4)`.
Do **not** add `tiktoken-rs`: the Python uses the heuristic on purpose because
tab-delimited content (CSVs, code) underestimates with real tokens. Drop
`tiktoken-rs` from `Cargo.toml`.

  - `SEPARATORS = ["\n\n", "\n", "\t", ". ", " "]` recursive split.
  - 64-word overlap (trailing N words of prev chunk → start of next).
  - `MAX_CHARS = 4000` post-split safety valve.
  - `NO_CHUNK_SOURCES = {bookmark, pin}` (both small, never split).
  - Each chunk: `id = "{doc_id}:{chunk_idx}"`, copies doc.metadata + adds `chunk_idx`.
  - Golden tests: pick 5 representative documents from the existing real
    `chunks.msgpack` and assert Rust chunker produces the same chunk count +
    content as Python (or document any drift).

#### Step 2.5 — `ingest::normalize/*` (~5–7 days, the biggest piece)

Each Python `normalize_*` becomes one Rust submodule. All read JSONL from
`data/raw/<file>.jsonl` (or sidecar `users.json` / `channels.json` for slack)
and emit `Vec<Document>`.

Build order (simplest → hardest; ship + test each before the next):

| #  | Module                  | Input                       | Python LOC | Notes                                                                |
|----|-------------------------|-----------------------------|------------|----------------------------------------------------------------------|
| a  | `normalize::bookmark`   | `bookmarks.jsonl`           | ~25        | Trivial: title + link → Document                                     |
| b  | `normalize::pin`        | `pins.jsonl` + users.json   | ~25        | Needs `clean_text` + user_map                                        |
| c  | `normalize::file`       | `files_extracted.jsonl`     | ~25        | Trivial mapping                                                      |
| d  | `normalize::code`       | `code.jsonl`                | ~30        | Prefix content with `# repo/path`                                    |
| e  | `normalize::atlassian`  | `atlassian.jsonl`           | ~30        | Branches on `type` (jira vs page)                                    |
| f  | `normalize::gdrive`     | `gdrive.jsonl`              | ~30        | Skip spreadsheets / Google Sheets                                    |
| g  | `normalize::bitbucket`  | `bitbucket.jsonl`           | ~50        | PR description + comments concatenated                               |
| h  | `normalize::git`        | `git.jsonl`                 | ~80        | Commit msg + body + author + date + diff + files                     |
| i  | `normalize::slack`      | `messages.jsonl` + sidecars | ~90        | Hardest: thread grouping, mrkdwn cleaning, system-subtype filtering  |

Shared infra in `ingest::normalize::mod`:
  - `read_jsonl(path) -> Vec<serde_json::Value>` (or strongly-typed via serde derive).
  - `load_users(path) -> HashMap<String, UserInfo>` + `load_channels(...)`.
  - `clean_text(text, &user_map) -> String` — port the 10 regex patterns from
    `normalize.py:16` via the `regex` crate (compiled once via `OnceLock`).
  - `user_label(uid, &user_map) -> String` — port the `name (title)` formatter.
  - `normalize_all(raw_dir) -> Vec<Document>` — calls each normalizer + dedups by `Document.id`.

Each normalizer gets a golden test reading 10 representative records from real
`data/raw/<source>/` and asserting the produced `Document` matches Python field-for-field.

#### Step 2.6 — `ingest::pipeline` (~1 day)

`pub fn run(cfg, force, source_filter) -> Result<IngestStats>` mirrors
`pipeline.py::ingest`:
  1. Resolve embedder via factory: `let embedder = embed::make(&cfg.embedding.provider, &cfg.embedding.model_name)?`
  2. Load per-(provider, model) cache: `embed::cache::load(index_dir, embedder.provider(), embedder.model())?`
  3. `normalize_all(raw_dir)` → docs (optionally `source_filter`-pruned)
  4. `chunk::all(docs, max_tokens, overlap)`
  5. content-hash diff vs cache
  6. `embedder.embed_documents(misses, batch_size, checkpoint_path)` (or skip if 0)
  7. Assemble full embedding `Vec<Vec<f32>>` in chunk order
  8. `index::store::save_chunks` + `index::bm25::Bm25::build` + `save_bm25` +
     write `embeddings.bin` (concat'd raw f32, atomic via tempfile + rename)
  9. `cache.save(index_dir)`
  10. Log per-source counts + elapsed
- Pre-warm rayon at the top so chunk + normalize benefit on the M-series Mac.

The pipeline never names a provider directly. Switching `cfg.embedding =
{provider="openai", model_name="text-embedding-3-large"}` is a config edit;
no Rust changes. The cache is namespaced by (provider, model) so the switch
doesn't poison the existing voyage cache.

#### Step 2.7 — `cli::ingest::run` (~½ day)

Already stubbed in Phase 0. Wire to `ingest::pipeline::run`. Print stats + elapsed.

#### Scope cuts (Phase 2 v1 will not include)

- **Voyage Batch API** (`submit_batch` / `collect_batch` in `embed.py:299`).
  33% discount, ≤12h completion window. Useful only for first-time bulk ingest
  of a large corpus. Defer until someone actually needs the discount; the sync
  API + adaptive throttle handles the regular case.
- **Multi-model embedding (multi-FAISS index per query).**
  `config.embedding.models = [...]` lets one index hold parallel embeddings
  from N models, fused via multi-signal RRF at query time. Adds index format
  complexity. The trait + per-(provider, model) cache shipped in this phase
  makes a future v2.5 multi-model addition mechanical.
- **`sentence-transformers` local embedder.** Already cut in the Phase 0 plan.
- **`ntfy.sh` mid-batch notifications.** Useful for hours-long Modal runs;
  defer until ingest runs that long are common.
- **Modal-based remote ingest** (`modal_ingest.py`). Stays Python for now —
  Modal's Python SDK is the entry point; nothing to gain by porting.

#### Parallelism

Normalize and chunk are embarrassingly parallel:
  - `normalize_all`: process the 9 source modules in parallel via
    `rayon::scope` (each I/O-bound on its JSONL file).
  - `chunk::all`: `docs.par_iter().flat_map(chunk_document).collect()`. With
    103k → 1M+ chunks, the speedup is meaningful on the M-series Mac.
  - `embed`: rate-limited; sequential (don't parallelize).

Ship serial first; add rayon after profiling shows it matters. Pre-warming
rayon happens in `cli::ingest::run` (same pattern as `cli::search::run`).

#### Why a custom binary cache (vs. keeping msgpack)

The currently migrated `embed_cache.msgpack` is 1.1 GB for 121k entries ×
1024 dim because `msgpack-python` serializes f32 lists as float64. Switching
to the binary format described in Step 2.1b cuts that to ~500 MB and is
~2× faster on read (mmap + `bytemuck::cast_slice` is zero-copy; msgpack
allocates a `Vec<f32>` per entry). It also makes `(provider, model)` part
of the filename, which is the cleanest way to namespace per-model caches.

`tools/migrate.py` updates to write the new format under
`embed_cache/voyage--voyage-code-3.bin`. The old `embed_cache.msgpack` can
be deleted after migration.

#### Verify (Phase 2 ship gate)

1. **From-scratch ingest works.** `rm -rf data/index/{*.msgpack,*.bin,embed_cache} &&
   ragrep ingest` produces `chunks.msgpack` + `embeddings.bin` +
   `bm25.msgpack` + `embed_cache/voyage--voyage-code-3.bin`.
2. **Chunk count parity.** New `chunks.msgpack` count matches the migrated
   v0.2 baseline within 5% (small drift from edge cases in the chunker is
   acceptable; document any > 5%).
3. **Search parity vs Python.** `tools/parity.py` against the freshly-Rust-built
   index passes 30/30 queries at ≥90% top-10 overlap (Phase 1's bar).
4. **Cache-hit idempotence.** A second `ragrep ingest` with no `data/raw/`
   changes triggers 0 embed API calls; all chunks hit cache. Wall < 15s for
   103k chunks (binary cache load + BM25 build).
5. **`--force` re-embeds.** Wipes the per-model cache, hits the embed API,
   produces a fresh `embeddings.bin` byte-identical to the prior run (for
   normalized vectors, the API is deterministic on the same input + model).
6. **`--source slack` partial.** Only re-embeds chunks where `chunk.source ==
   "slack"`. Other sources stay cached.
7. **Throttle behavior under rate-limit.** Manually trigger 429 (or use a
   mocked endpoint) and confirm `embed::throttle` backs off + recovers.
8. **Checkpoint resume.** Kill mid-batch (Ctrl-C after 500+ batches), restart,
   resume from checkpoint. End state matches uninterrupted run.
9. **Provider swap is config-only.** Set `[embedding] provider = "openai",
   model_name = "text-embedding-3-large"` in `config.toml`; `ragrep ingest`
   embeds against OpenAI, writes `embed_cache/openai--text-embedding-3-large.bin`,
   leaves the voyage cache untouched. Search still works. Switching back to
   voyage reuses the cached voyage embeddings (zero re-embed).

### Phase 3 — scrapers (3 weeks)
- `ingest::scrape::git` — git2 bindings, port `scrape_git.py`. Do first as proof — easiest source.
- `ingest::scrape::slack` — raw HTTP to api.slack.com/api/* with pagination, threads, bookmarks, pins, files (port the 823-LOC `scrape_slack.py`)
- `ingest::scrape::atlassian` — Confluence pages + Jira issues + comments via basic auth
- `ingest::scrape::bitbucket` — OAuth client-creds flow, PR descriptions + comments
- `ingest::scrape::gdrive` — yup-oauth2 service account or ADC, list + download docs/sheets/slides/PDFs
- `ingest::scrape::code` — file walking + extension filter
- `ingest::extract` — PDF/DOCX/PPTX/XLSX → text via Gemini multimodal HTTP (port `extract.py`)
- `cli::scrape::run` — `ragrep scrape [--source slack,git,...]`

**Verify**: each scraper produces NDJSON in `data/raw/{source}/` byte-identical (modulo timestamps) to the Python equivalent for ≥100 real records. Diff with `jq -S` to ignore key ordering.

### Phase 4 — server + eval (2 weeks)
- `serve` — axum router in `serve/mod.rs`
- `serve::search::handle` — `GET /search` handler matching FastAPI's JSON shape
- `serve::auth` — Cloud Run JWT verification middleware
- `cli::serve::run` — `ragrep serve`; GCS startup download if `RAGREP_GCS_BUCKET` set
- `eval::entity::extract` — Jira-ID extraction only (service-pattern was stripped during pre-publish cleanup)
- `eval::harness::run` — port `harness.py`, source_recall / entity_recall / MRR per stage
- `cli::{eval, stats, inspect}::run`
- Cloud Run deploy: build `aarch64-unknown-linux-gnu` binary, package in distroless container

**Verify**: `ragrep eval` numbers match the Python baseline within ±1pp. `ragrep serve &` followed by `curl localhost:8321/search?q=auth&mode=grep` returns the same JSON shape as the Python server.

### Phase 5 — switch over (1 week)
- Delete `src/ragrep/` (Python source)
- Delete `pyproject.toml`, `uv.lock`
- Keep `tools/migrate.py` for users still on the Python release
- Update `README.md`: cargo-build instructions for contributors, install.sh for users
- Update `docs/index.html`: drop "ripgrep for your team's knowledge base" if it now feels redundant given the Rust lineage
- `Cargo.toml` metadata: same author, MIT, repo URL
- Release `v1.0.0` to GitHub Releases. Bump `install.sh` to download v1.0.0 by default
- Write `MIGRATION.md` with the one-line `python tools/migrate.py data/index/` recipe and the local-provider regression note

**Verify**: from a clean Docker container (`debian:bookworm-slim` and `archlinux`), `curl ragrep.cc/install.sh | sh && ragrep --help` works. From a fresh macOS, same. `ragrep "auth" --server http://...cloud-run-url` works (proves Cloud Run binary too).

## Hot-path performance targets

These are the success criteria for "did the rewrite achieve the latency goal":

Workload assumed: 103,377 chunks × 1024-d normalized f32 embeddings. Voyage rerank latency dominates hybrid-mode wall time (~150ms p50 over network, unchanged).

| Metric                                       | Python today  | Rust target |
|----------------------------------------------|---------------|-------------|
| `ragrep --help` cold start                   | ~150ms        | <20ms       |
| Index load (mmap, no I/O)                    | ~250ms (faiss read) | <2ms (mmap)  |
| Vector search (one query, 103k×1024)         | ~6ms (faiss native SIMD) | <10ms (`wide` f32x8) |
| `ragrep "term" -m grep` (warm OS cache)      | ~1.2s         | <300ms      |
| `ragrep "term" -m hybrid` (warm)             | ~2.5s         | <800ms      |
| Server `/search?mode=grep` p50               | ~50ms         | <10ms       |
| Server cold start (Cloud Run, index downloaded) | ~3 min     | ~2 min (GCS download dominates) |
| Binary size (stripped, single arch)          | n/a           | <20 MB (no libfaiss baggage) |

## Risks & mitigations

- **Flat-search SIMD performance.** Hand-rolled with `wide::f32x8` should land within 2× of FAISS's native SIMD on Apple Silicon. _Mitigation_: Phase 0.1 spike measures this directly. If we're slower than 10ms per query at 103k chunks, fall back to `usearch` (single-header, mmap, mature; ~half-day swap). Decision point inside Phase 0, day 1.
- **mmap correctness on Linux/macOS/Windows.** `memmap2` handles all three but Windows file-locking semantics differ. _Mitigation_: deferred until Phase 0.2 cross-compile CI exercises Windows; if Windows mmap is fragile, gate the Windows binary on `--no-mmap` mode (read-into-Vec) until fixed.
- **GDrive OAuth in Rust.** `yup-oauth2` requires either service-account JSON (cleanest) or an installed-app flow with localhost callback. _Mitigation_: require service-account JSON for v1; document how to create one. The Python path used `gcloud auth application-default login` which is convenient for the user but harder to match in Rust without shelling out.
- **Async/sync split in CLI.** `ragrep "query"` doesn't need an async runtime (one HTTP call to Voyage rerank). Spinning up tokio for that is overkill (~5ms cost). _Mitigation_: `embed::` and `rerank::` expose both sync (`ureq`) and async (`reqwest`) variants; the CLI search path uses sync, the `serve` subcommand uses async.
- **tiktoken vs Voyage tokenizer drift.** Voyage uses its own tokenizer; we'd be using `cl100k_base`. For chunk-size enforcement (512 tokens) the drift is small (≤5% chunk-count differences in practice). _Mitigation_: accept the drift; chunks aren't required to be exactly 512 tokens. If Voyage rejects an oversize batch we already retry.
- **Pickle migration robustness.** The migration script must read whatever pickle protocol Python wrote. _Mitigation_: the script runs in the user's existing venv (which has the same pickle/Python version that wrote it), so this is a non-issue. Document the venv requirement in MIGRATION.md.
- **Scope creep.** 10–12 weeks part-time is a lot. _Mitigation_: Phase 1 is the ship-gate. After Phase 1, users get the latency win immediately. Phases 2–5 can take as long as they take without blocking value.

## Critical files to model the Rust port on

| Python source                              | LOC | Rust target            | Notes |
|--------------------------------------------|-----|------------------------|-------|
| `src/ragrep/models.py`                     | 61  | `models`               | Flat dataclasses → serde structs |
| `src/ragrep/config.py`                     | 159 | `config`               | Env-file + XDG search already correct shape |
| `src/ragrep/ingest/store.py`               | 140 | `index::{store, flat, bm25}` | `save_index`, `load_index`, `model_key` map to new format (chunks.msgpack + embeddings.bin + bm25.msgpack); FAISS calls become `index::flat` mmap+SIMD |
| `src/ragrep/query/retrieve.py`             | 222 | `query::retrieve`      | `dense_search`, `bm25_search`, `_reciprocal_rank_fusion`, `_multi_signal_rrf` |
| `src/ragrep/search.py`                     | 244 | `query::{grep,semantic,hybrid}` | Public API; collapses into `query/mod.rs` top-level fns |
| `src/ragrep/cli.py`                        | 777 | `cli/` subtree         | argparse → clap; splash → `splash`; server proxy → `cli::search::proxy` |
| `src/ragrep/server.py`                     | 172 | `serve::search`        | `/search` endpoint shape → axum handler |
| `src/ragrep/ingest/embed.py`               | 814 | `embed/`               | `VoyageEmbedder` is the model; throttle → `embed::throttle`, cache → `embed::cache` |
| `src/ragrep/ingest/scrape_slack.py`        | 823 | `ingest::scrape::slack` | Biggest scraper; sets the complexity-budget bar |

## Verification recipe (end-to-end)

After Phase 0.1 (spike) ships:

```bash
# Extract embeddings from existing FAISS index
python tools/extract_embeddings.py data/index/
ls -la data/index/embeddings.bin                          # expect 423,329,792 bytes

# Build the spike
cargo build --release

# Spike parity check (uses query embeddings extracted from Python)
python tools/spike_parity.py                               # writes 3 query JSON files + Python top-10
./target/release/ragrep --query data/spike/q_auth.json     # expect same top-10 indices
./target/release/ragrep --query data/spike/q_deploy.json
./target/release/ragrep --query data/spike/q_incident.json

# Bench
./target/release/ragrep --query data/spike/q_auth.json --bench 10  # expect <10ms p50
```

After Phase 1 ships:

```bash
# Build for current platform
cargo build --release

# Smoke test
./target/release/ragrep --help                            # <20ms expected
./target/release/ragrep                                   # shows splash

# One-time migration: chunks.pkl + embed_cache.pkl → msgpack (extract_embeddings already ran in Phase 0.1)
python tools/migrate.py data/index/

# Parity test against Python
./tools/parity.sh                                         # 50 queries, ≥90% top-10 overlap

# End-to-end search
./target/release/ragrep "auth" -m grep -n 5
./target/release/ragrep "auth" -m hybrid -n 5
./target/release/ragrep "auth" --json | jq

# Server mode (uses existing Python server)
RAGREP_SERVER=http://localhost:8321 ./target/release/ragrep "auth"

# Cross-compile sanity
cross build --release --target aarch64-apple-darwin
cross build --release --target x86_64-unknown-linux-gnu
```

After Phase 5 ships:

```bash
# From clean Docker
docker run --rm -it debian:bookworm-slim bash -c \
  "apt-get update && apt-get install -y curl ca-certificates && \
   curl -fsSL https://ragrep.cc/install.sh | sh && \
   ~/.local/bin/ragrep --help && \
   RAGREP_SERVER=https://your-cloud-run.run.app ~/.local/bin/ragrep 'auth'"
```

## What stays after Phase 5

- `docs/` (landing page, install.sh, fonts) — unchanged content, install.sh now downloads binary
- `tools/migrate.py` — kept indefinitely so old-Python users can still migrate
- `config.toml` format — same TOML shape, fully readable by Rust
- `.env` / `.env.example` — same env vars
- `data/index/` layout — same files, msgpack instead of pickle for chunks/embed_cache
- GitHub repo URL (`github.com/fntune/ragrep`) — same
- Vanity domain (`ragrep.cc`) — same

What goes:
- `pyproject.toml`, `src/ragrep/`, `uv.lock`, `Makefile` (replaced by `justfile` or just `cargo` commands)
- `requirements.txt` Cloud Run target (replaced by `Dockerfile` building the Rust server binary)
- All Python deps (voyageai already gone, faiss-cpu/rank-bm25/torch/transformers/sentence-transformers/google-genai/etc.)
