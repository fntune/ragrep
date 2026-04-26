SHELL := /usr/bin/env bash
-include .env
export
.PHONY: help install scrape ingest query stats eval inspect check test clean serve \
        upload-index download-index

CONFIG ?= config.toml
LOG_LEVEL ?= INFO
GCS_BUCKET ?= $(RAGREP_GCS_BUCKET)
INDEX_DIR ?= data/index
RAGREP := cargo run --manifest-path rust/Cargo.toml --
INDEX_FILES := embeddings.bin chunks.msgpack bm25.msgpack

help:
	@echo "ragrep — Rust hybrid retrieval pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  install          Build the release binary"
	@echo ""
	@echo "Pipeline:"
	@echo "  scrape           Scrape data from sources (SOURCE=slack|git|atlassian|gdrive|bitbucket|files)"
	@echo "  ingest           Build embeddings.bin + chunks.msgpack + bm25.msgpack"
	@echo "  query            Query the index (Q='question')"
	@echo "  stats            Show index statistics"
	@echo "  eval             Run evaluation harness"
	@echo "  inspect          Inspect raw data and pipeline output"
	@echo ""
	@echo "Server:"
	@echo "  serve            Run local search server on port 8321"
	@echo ""
	@echo "Index management:"
	@echo "  upload-index     Upload index to GCS (requires RAGREP_GCS_BUCKET)"
	@echo "  download-index   Download index from GCS (requires RAGREP_GCS_BUCKET)"
	@echo ""
	@echo "Development:"
	@echo "  check            Run linting and type checking"
	@echo "  test             Run tests"
	@echo "  clean            Remove data/index/ directory"
	@echo ""
	@echo "Search:"
	@echo "  ragrep <term> [-n 5] [-s git] [--full]"
	@echo "  ragrep <term> --server http://localhost:8321"

install:
	cargo build --manifest-path rust/Cargo.toml --release --bin ragrep

# GCS index management (requires RAGREP_GCS_BUCKET env var)
upload-index:
	@test -n "$(GCS_BUCKET)" || (echo "Error: RAGREP_GCS_BUCKET not set" && exit 1)
	@echo "Uploading index to gs://$(GCS_BUCKET)/ ..."
	@for file in $(INDEX_FILES); do \
		gcloud storage cp "$(INDEX_DIR)/$$file" "gs://$(GCS_BUCKET)/$$file"; \
	done
	@echo "Upload complete. Embedding cache excluded (ingestion only)."

download-index:
	@test -n "$(GCS_BUCKET)" || (echo "Error: RAGREP_GCS_BUCKET not set" && exit 1)
	@mkdir -p $(INDEX_DIR)
	@echo "Downloading index from gs://$(GCS_BUCKET)/ ..."
	@for file in $(INDEX_FILES); do \
		gcloud storage cp "gs://$(GCS_BUCKET)/$$file" "$(INDEX_DIR)/$$file"; \
	done
	@echo "Index downloaded to $(INDEX_DIR)/."

# Server
serve:
	$(RAGREP) serve --config $(CONFIG) --port 8321

# Pipeline commands
scrape:
ifdef SOURCE
	$(RAGREP) scrape --config $(CONFIG) --source $(SOURCE)
else
	$(RAGREP) scrape --config $(CONFIG)
endif

INGEST_ARGS :=
ifdef FORCE
INGEST_ARGS += --force
endif
ifdef SOURCE
INGEST_ARGS += --source $(SOURCE)
endif

ingest:
	$(RAGREP) ingest --config $(CONFIG) $(INGEST_ARGS)

query:
ifdef Q
	$(RAGREP) "$(Q)" --config $(CONFIG)
else
	$(RAGREP)
endif

stats:
	$(RAGREP) stats --config $(CONFIG)

eval:
	$(RAGREP) eval --config $(CONFIG)

INSPECT_ARGS :=
ifdef SOURCE
INSPECT_ARGS += --source $(SOURCE)
endif
ifdef GREP
INSPECT_ARGS += --grep "$(GREP)"
endif
ifdef N
INSPECT_ARGS += -n $(N)
endif
ifdef FULL
INSPECT_ARGS += --full
endif

inspect:
	$(RAGREP) inspect $(MODE) --config $(CONFIG) $(INSPECT_ARGS)

check:
	cargo fmt --manifest-path rust/Cargo.toml --check
	cargo clippy --manifest-path rust/Cargo.toml --all-targets --no-deps

test:
	cargo test --manifest-path rust/Cargo.toml

clean:
	@echo "Removing data/index/..."
	rm -rf data/index/
	@echo "Clean complete"
