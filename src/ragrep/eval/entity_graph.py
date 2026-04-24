"""Cross-source entity graph for evaluation ground truth.

Extracts Jira IDs from chunks and builds a graph of which entities appear
across multiple sources. Cross-source clusters serve as automatic ground
truth for retrieval evaluation.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ragrep.models import Chunk

# Regex for Jira ticket IDs — exclude common false positives
_JIRA_RE = re.compile(r"\b([A-Z]{2,10}-\d{1,6})\b")
_JIRA_EXCLUDE = {"UTF-8", "ISO-8859", "US-ASCII", "X-HTTP", "X-API", "X-CSRF"}


@dataclass
class Entity:
    """An entity that appears across chunks and sources."""

    etype: str  # "jira"
    eid: str  # "PLAT-984"
    # source → list of chunk indices into the chunks array
    chunks_by_source: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))

    @property
    def sources(self) -> set[str]:
        return set(self.chunks_by_source.keys())

    @property
    def n_sources(self) -> int:
        return len(self.chunks_by_source)

    @property
    def all_chunk_indices(self) -> set[int]:
        return {idx for indices in self.chunks_by_source.values() for idx in indices}

    @property
    def total_mentions(self) -> int:
        return sum(len(v) for v in self.chunks_by_source.values())


@dataclass
class EvalCase:
    """A retrieval evaluation case derived from a cross-source entity."""

    entity_key: str  # "jira:PLAT-984"
    query: str
    expected_sources: set[str]
    ground_truth_indices: set[int]  # chunk indices that should appear


@dataclass
class EntityGraph:
    """Cross-source entity graph built from chunks."""

    entities: dict[str, Entity]  # "jira:PLAT-984" → Entity
    chunks: list[Chunk]

    @classmethod
    def build(cls, chunks: list[Chunk]) -> "EntityGraph":
        """Extract entities from all chunks and build the graph."""
        entities: dict[str, Entity] = {}

        for i, chunk in enumerate(chunks):
            text = chunk.content

            # Jira IDs
            for m in _JIRA_RE.finditer(text):
                jid = m.group(1)
                if jid in _JIRA_EXCLUDE or jid.startswith(("HTTP-", "SHA-")):
                    continue
                key = f"jira:{jid}"
                if key not in entities:
                    entities[key] = Entity(etype="jira", eid=jid)
                entities[key].chunks_by_source[chunk.source].append(i)

        return cls(entities=entities, chunks=chunks)

    def cross_source_entities(self, min_sources: int = 2) -> list[Entity]:
        """Entities that appear across multiple sources, sorted by source count."""
        cross = [e for e in self.entities.values() if e.n_sources >= min_sources]
        cross.sort(key=lambda e: (-e.n_sources, -e.total_mentions))
        return cross

    def eval_cases(self, min_sources: int = 2, max_cases: int = 50) -> list[EvalCase]:
        """Generate eval cases from cross-source entities."""
        cases: list[EvalCase] = []

        for entity in self.cross_source_entities(min_sources):
            query = self._make_query(entity)
            if not query:
                continue

            cases.append(EvalCase(
                entity_key=f"{entity.etype}:{entity.eid}",
                query=query,
                expected_sources=entity.sources,
                ground_truth_indices=entity.all_chunk_indices,
            ))

            if len(cases) >= max_cases:
                break

        return cases

    def _make_query(self, entity: Entity) -> str | None:
        """Generate a natural language query for an entity."""
        if entity.etype == "jira":
            # Try to find the Jira ticket's title from an atlassian chunk
            for idx in entity.chunks_by_source.get("atlassian", []):
                chunk = self.chunks[idx]
                title = chunk.title.strip()
                if title and entity.eid in title:
                    # Strip the Jira ID prefix to get the description
                    clean = re.sub(r"^[A-Z]{2,10}-\d{1,6}\s*[-:]\s*", "", title).strip()
                    if clean and len(clean) > 10:
                        return clean

            # Fallback: use the first atlassian chunk's title
            for idx in entity.chunks_by_source.get("atlassian", []):
                title = self.chunks[idx].title.strip()
                if title and len(title) > 10:
                    return title

            # Fallback: use git commit message
            for idx in entity.chunks_by_source.get("git", []):
                title = self.chunks[idx].title.strip()
                if title and entity.eid in title and len(title) > 10:
                    # Clean up common commit message prefixes
                    return re.sub(r"^(feat|fix|chore|refactor|docs)\s*[:(]\s*", "", title).strip()

        return None
