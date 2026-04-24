"""Configuration loading from TOML."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "data/raw"
    index_dir: str = "data/index"


@dataclass(frozen=True)
class IngestConfig:
    batch_size: int = 32
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64


@dataclass(frozen=True)
class EmbeddingModelConfig:
    """Single embedding model specification."""

    provider: str
    model_name: str
    device: str = "mps"


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "voyage"
    model_name: str = "voyage-code-3"
    device: str = "mps"  # only used by sentence-transformers provider
    models: tuple[EmbeddingModelConfig, ...] = ()

    def get_models(self) -> list[EmbeddingModelConfig]:
        """Return model list. Falls back to single provider/model_name if models is empty."""
        if self.models:
            return list(self.models)
        return [EmbeddingModelConfig(self.provider, self.model_name, self.device)]


@dataclass(frozen=True)
class RerankerConfig:
    provider: str = "voyage"
    model_name: str = "rerank-2.5"
    device: str = "mps"  # only used by sentence-transformers provider


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60
    dedup_threshold: float = 0.0


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = "gemma3:4b"
    ollama_url: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 1024


@dataclass(frozen=True)
class ScrapeConfig:
    """Scrape settings per source. Secrets come from env vars, not config."""
    slack: dict = field(default_factory=dict)
    atlassian: dict = field(default_factory=dict)
    gdrive: dict = field(default_factory=dict)
    git: dict = field(default_factory=dict)
    bitbucket: dict = field(default_factory=dict)
    code: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    data: DataConfig
    ingest: IngestConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig
    scrape: ScrapeConfig

    @property
    def raw_dir(self) -> Path:
        return Path(self.data.raw_dir)

    @property
    def index_dir(self) -> Path:
        return Path(self.data.index_dir)


def load_env_files() -> None:
    """Populate os.environ from .env files. Searches CWD then ~/.config/ragrep/.env. Existing vars win."""
    candidates = [Path.cwd() / ".env", Path.home() / ".config" / "ragrep" / ".env"]
    for env_file in candidates:
        if not env_file.exists():
            continue
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


def find_config_path(explicit: Path | None = None) -> Path | None:
    """Resolve config.toml location: explicit arg, RAGREP_CONFIG env, ./config.toml, ~/.config/ragrep/config.toml."""
    if explicit is not None:
        return explicit
    env = os.environ.get("RAGREP_CONFIG")
    if env:
        return Path(env)
    cwd_path = Path("config.toml")
    if cwd_path.exists():
        return cwd_path
    xdg_path = Path.home() / ".config" / "ragrep" / "config.toml"
    if xdg_path.exists():
        return xdg_path
    return None


def load_config(path: Path | None = None) -> Config:
    """Load config from TOML file. Searches RAGREP_CONFIG, CWD, ~/.config/ragrep/ if path not given."""
    resolved = find_config_path(path)
    raw: dict = {}
    if resolved is not None and resolved.exists():
        with open(resolved, "rb") as f:
            raw = tomllib.load(f)

    scrape_raw = raw.get("scrape", {})

    embedding_raw = raw.get("embedding", {})
    embedding_models_raw = embedding_raw.pop("models", None)
    embedding_kwargs = dict(embedding_raw)
    if embedding_models_raw:
        embedding_kwargs["models"] = tuple(EmbeddingModelConfig(**m) for m in embedding_models_raw)

    return Config(
        data=DataConfig(**raw.get("data", {})),
        ingest=IngestConfig(**raw.get("ingest", {})),
        embedding=EmbeddingConfig(**embedding_kwargs),
        reranker=RerankerConfig(**raw.get("reranker", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        generation=GenerationConfig(**raw.get("generation", {})),
        scrape=ScrapeConfig(
            slack=scrape_raw.get("slack", {}),
            atlassian=scrape_raw.get("atlassian", {}),
            gdrive=scrape_raw.get("gdrive", {}),
            git=scrape_raw.get("git", {}),
            bitbucket=scrape_raw.get("bitbucket", {}),
            code=scrape_raw.get("code", {}),
        ),
    )
