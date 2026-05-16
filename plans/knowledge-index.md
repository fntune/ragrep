# Knowledge Index Tasklist

This tasklist groups the Pinecone replacement work into PR-sized chunks. Each chunk should leave one coherent contract behind, with obsolete names and compatibility paths removed unless an external boundary requires them.

## Ragrep

- [x] CLI contract: make the installed `ragrep` command support both direct search and the documented pipeline commands.
- [x] Search contract tests: cover grep, semantic, hybrid formatting, metadata filters, dates, and HTTP search responses without requiring external model calls.
- [x] Metadata filters: replace substring-only flat filters with a small query contract that supports exact values, `$or`, and typed metadata.
- [x] Collection catalog: add a persisted record contract for upsert, fetch metadata by ids, list ids by metadata, delete, count, clear, and document export.
- [ ] Collection search API: connect collection records to indexed search without requiring callers to know chunk/index internals.
- [ ] Mutable persistence: define how FAISS/BM25/chunks update safely after record changes, including compaction or rebuild triggers.
- [ ] Server write surface: expose authenticated collection endpoints that support support-chatbot sync jobs without sharing local filesystem state.
- [ ] Deployment contract: document index storage, rebuild flow, backup/restore, and readiness behavior for a long-running internal service.

## Support Chatbot

- [ ] Rename the dependency: replace Pinecone-specific settings, vendor names, result types, logs, and docs with a knowledge-index contract.
- [ ] Article search path: move `query_knowledge_base`, offers search, admin search, article listing, metadata lookup, sync, delete, and stats to the new client.
- [ ] YouTube path: move video search, playlist reconciliation, metadata fetch, delete, and sync diagnostics to the new client.
- [ ] Config and deploy: remove `pinecone==7.3.0`, replace `PINECONE_*` env with knowledge-index settings, and update Docker/Kubernetes/secrets/docs.
- [ ] Migration and verification: build a one-time backfill from Freshdesk/YouTube into the new index, then run config tests plus focused tool/sync regression tests.
