# Old Index Migration

The Rust runtime does not read old Python-built indexes directly.

Old layout:

- `faiss.index`
- `chunks.pkl`
- `bm25.pkl`
- `embed_cache.pkl`

Rust layout:

- `embeddings.bin`
- `chunks.msgpack`
- `bm25.msgpack`
- `embed_cache/{provider}--{model}.bin`

## Migrate An Existing Index

Run the migration helper in an environment that has the old pickle dependencies available:

```bash
python -m pip install faiss-cpu numpy msgpack
python tools/migrate.py data/index
```

Then rebuild BM25 in the Rust format:

```bash
cargo run --manifest-path rust/Cargo.toml -- rebuild-bm25 --config config.toml
```

Check the result:

```bash
cargo run --manifest-path rust/Cargo.toml -- stats --config config.toml
cargo run --manifest-path rust/Cargo.toml -- "auth" -m grep -n 5 --config config.toml
```

## Fresh Rust Index

For new indexes, use the Rust pipeline directly:

```bash
cargo run --manifest-path rust/Cargo.toml -- scrape --config config.toml
cargo run --manifest-path rust/Cargo.toml -- ingest --config config.toml
```

`ragrep ingest` writes all runtime files directly:

- `data/index/embeddings.bin`
- `data/index/chunks.msgpack`
- `data/index/bm25.msgpack`
- `data/index/embed_cache/{provider}--{model}.bin`

## Notes

- `tools/migrate.py` is the only retained Python file. It exists solely to read `faiss.index`, `chunks.pkl`, and `embed_cache.pkl`.
- If the old index used sentence-transformers, build a fresh Rust index with an HTTP provider before comparing semantic or hybrid results. Grep parity is independent of embedding provider.
- Upload Rust server indexes to GCS with the Rust filenames: `embeddings.bin`, `chunks.msgpack`, and `bm25.msgpack`.
