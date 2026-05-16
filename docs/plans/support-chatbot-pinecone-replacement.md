# Support Chatbot Pinecone Replacement

## Current Finding

`support-chatbot` uses Pinecone as the knowledge index behind `query_knowledge_base`.
It is not only a search dependency. The current Pinecone wrapper owns article
search, video search, record upserts, deletes, metadata fetches, full metadata
listing, namespace counts, and health checks.

Ragrep already owns the right retrieval domain for replacement: local/served
hybrid search over indexed chunks. The missing work is not another vector-store
adapter. It is making ragrep a dependable knowledge-index service for an
external application that syncs Freshdesk articles and YouTube videos
independently.

## support-chatbot Dependency Surface

Article search:

- `ToolsService.query_knowledge_base` calls `PineconeApi.search_articles`.
- It filters Freshdesk articles by `portal_id` when configured.
- Results are exposed as `knowledges` with `id`, `title`, `content`, `link`,
  `updated_at`, and `score`.

YouTube search:

- `ToolsService.query_knowledge_base` also calls
  `PineconeApi.search_youtube_videos` and returns `videos`.
- Video results need `video_id`, `title`, `description`, `video_url`,
  `thumbnail_url`, and `score`.
- YouTube is optional, but a partially configured index should fail clearly.

Sync and admin behavior:

- Freshdesk sync upserts published articles, compares `updated_at`, deletes
  unpublished or removed articles, fetches metadata by ID, lists all indexed
  article metadata, and reports a namespace count.
- YouTube sync reconciles playlist videos, lists indexed IDs by playlist, fetches
  metadata by ID, upserts changed videos, and deletes removed videos.
- Article admin routes list, fetch, sync, delete, and search article records.
- Readiness currently treats Pinecone as a required dependency.

## Completed Rust Foundations

- Runtime index publication is locked and staged, so `chunks.msgpack`,
  `embeddings.bin`, and `bm25.msgpack` are published as one generation.
- `ragrep ingest --source <source>` replaces only chunks from that source and
  preserves the other sources already in the runtime index.
- `/knowledge/search` returns the article and video shapes consumed by
  `support-chatbot`, backed by the current ragrep search engine.

## Remaining Ragrep Gaps

- Ragrep has no first-class upsert/delete/list/fetch record API. The current
  model is scrape raw files, normalize, ingest, and serve.
- `ragrep serve` loads the index at startup. A production replacement needs a
  clear refresh story after sync publishes a new index.

## Tasklist

- [x] Atomic runtime index publication.
- [x] Source-scoped ingest that preserves other sources.
- [x] Support knowledge result contract.
- [ ] Record sync commands for support sources.
   Add a support-source ingestion path that writes Freshdesk and YouTube raw
   records with stable IDs and metadata, then invokes source-scoped ingest. Do
   not add a second vector-store abstraction.
- [ ] Service refresh path.
   Define and implement how `ragrep serve` observes a newly published index:
   explicit reload endpoint, signal, or process restart contract. Make health
   report the loaded index generation and chunk count.
- [ ] support-chatbot adapter migration.
   In `support-chatbot`, replace `PineconeApi` with a `KnowledgeIndex` boundary
   that calls ragrep. Update config, readiness, sync services, routes, docs, and
   tests in one coherent migration.

## Next Task

Add record sync commands for support sources. They should write Freshdesk and
YouTube records into ragrep-owned raw data, then call source-scoped ingest so
the support application does not need filesystem access to mutate the index.
