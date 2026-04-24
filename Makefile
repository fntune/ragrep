SHELL := /usr/bin/env bash
-include .env
export
.PHONY: help install scrape ingest query stats eval inspect check test clean serve \
        upload-index download-index requirements.txt

CONFIG ?= config.toml
LOG_LEVEL ?= INFO
GCS_BUCKET ?= $(RAGREP_GCS_BUCKET)
INDEX_DIR ?= data/index
INDEX_FILES := faiss.index chunks.pkl bm25.pkl

help:
	@echo "ragrep — hybrid FAISS + BM25 RAG pipeline"
	@echo ""
	@echo "Setup:"
	@echo "  install          Create venv and install dependencies"
	@echo ""
	@echo "Pipeline:"
	@echo "  scrape           Scrape data from sources (SOURCE=slack|git|atlassian|gdrive|bitbucket|files)"
	@echo "  ingest           Build FAISS + BM25 index from raw data"
	@echo "  query            Query the index (Q='question' or interactive REPL)"
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
	@echo "Search (activate venv first):"
	@echo "  ragrep <term> [-n 5] [-s git] [--full]"
	@echo "  ragrep <term> --server http://localhost:8321"

install:
	@echo "Creating virtual environment with uv..."
	uv sync --extra full --extra dev --extra serve
	@echo "Done. Activate with: source .venv/bin/activate"

# GCS index management (requires RAGREP_GCS_BUCKET env var)
upload-index:
	@test -n "$(GCS_BUCKET)" || (echo "Error: RAGREP_GCS_BUCKET not set" && exit 1)
	@echo "Uploading index to gs://$(GCS_BUCKET)/ ..."
	gcloud storage cp $(INDEX_DIR)/faiss.index gs://$(GCS_BUCKET)/
	gcloud storage cp $(INDEX_DIR)/chunks.pkl gs://$(GCS_BUCKET)/
	gcloud storage cp $(INDEX_DIR)/bm25.pkl gs://$(GCS_BUCKET)/
	@echo "Upload complete. embed_cache.pkl excluded (ingestion only)."

download-index:
	@test -n "$(GCS_BUCKET)" || (echo "Error: RAGREP_GCS_BUCKET not set" && exit 1)
	@mkdir -p $(INDEX_DIR)
	@echo "Downloading index from gs://$(GCS_BUCKET)/ ..."
	gcloud storage cp gs://$(GCS_BUCKET)/faiss.index $(INDEX_DIR)/
	gcloud storage cp gs://$(GCS_BUCKET)/chunks.pkl $(INDEX_DIR)/
	gcloud storage cp gs://$(GCS_BUCKET)/bm25.pkl $(INDEX_DIR)/
	@echo "Index downloaded to $(INDEX_DIR)/."

# Server
serve:
	uv run uvicorn ragrep.server:app --port 8321 --reload

# Cloud Run deployment helpers
requirements.txt: pyproject.toml uv.lock
	@uv export --format requirements-txt --no-hashes --no-dev --no-emit-project --no-header \
		--extra serve 2>/dev/null > requirements.txt
	@echo "Generated requirements.txt"

# Pipeline commands
scrape:
ifdef SOURCE
	uv run python -m ragrep.cli --config $(CONFIG) --log-level $(LOG_LEVEL) scrape --source $(SOURCE)
else
	uv run python -m ragrep.cli --config $(CONFIG) --log-level $(LOG_LEVEL) scrape
endif

INGEST_ARGS :=
ifdef FORCE
INGEST_ARGS += --force
endif
ifdef SOURCE
INGEST_ARGS += --source $(SOURCE)
endif

ingest:
	uv run python -m ragrep.cli --config $(CONFIG) --log-level $(LOG_LEVEL) ingest $(INGEST_ARGS)

query:
ifdef Q
	uv run python -m ragrep.cli --config $(CONFIG) --log-level $(LOG_LEVEL) query -q "$(Q)"
else
	uv run python -m ragrep.cli --config $(CONFIG) --log-level $(LOG_LEVEL) query
endif

stats:
	uv run python -m ragrep.cli --config $(CONFIG) --log-level $(LOG_LEVEL) stats

eval:
	uv run python -m ragrep.cli --config $(CONFIG) --log-level $(LOG_LEVEL) eval

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
	uv run python -m ragrep.cli --config $(CONFIG) --log-level $(LOG_LEVEL) inspect $(MODE) $(INSPECT_ARGS)

check:
	uv run ruff check src/ tests/
	uv run ruff format --check src/ tests/
	uv run mypy src/

test:
	uv run pytest tests/ -v

clean:
	@echo "Removing data/index/..."
	rm -rf data/index/
	@echo "Clean complete"
