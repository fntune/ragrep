//! Cross-source entity graph for evaluation ground truth.
//!
//! Extracts Jira IDs from chunks and builds retrieval cases from entities
//! that appear across multiple sources.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use regex::Regex;

use crate::models::Chunk;

const JIRA_EXCLUDE: &[&str] = &["UTF-8", "ISO-8859", "US-ASCII", "X-HTTP", "X-API", "X-CSRF"];

fn jira_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"\b([A-Z]{2,10}-\d{1,6})\b").unwrap())
}

fn jira_title_prefix_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^[A-Z]{2,10}-\d{1,6}\s*[-:]\s*").unwrap())
}

fn commit_prefix_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^(feat|fix|chore|refactor|docs)\s*[:(]\s*").unwrap())
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub kind: String,
    pub id: String,
    pub chunks_by_source: BTreeMap<String, Vec<usize>>,
}

impl Entity {
    pub fn sources(&self) -> BTreeSet<String> {
        self.chunks_by_source.keys().cloned().collect()
    }

    pub fn source_count(&self) -> usize {
        self.chunks_by_source.len()
    }

    pub fn all_chunk_indices(&self) -> BTreeSet<usize> {
        self.chunks_by_source
            .values()
            .flat_map(|indices| indices.iter().copied())
            .collect()
    }

    pub fn total_mentions(&self) -> usize {
        self.chunks_by_source.values().map(Vec::len).sum()
    }
}

#[derive(Debug, Clone)]
pub struct EvalCase {
    pub entity_key: String,
    pub query: String,
    pub expected_sources: BTreeSet<String>,
    pub ground_truth_indices: BTreeSet<usize>,
}

#[derive(Debug, Clone)]
pub struct EntityGraph<'a> {
    pub entities: BTreeMap<String, Entity>,
    pub chunks: &'a [Chunk],
}

impl<'a> EntityGraph<'a> {
    pub fn build(chunks: &'a [Chunk]) -> Self {
        let mut entities: BTreeMap<String, Entity> = BTreeMap::new();

        for (i, chunk) in chunks.iter().enumerate() {
            for caps in jira_re().captures_iter(&chunk.content) {
                let id = caps[1].to_string();
                if is_excluded_jira(&id) {
                    continue;
                }
                let key = format!("jira:{id}");
                entities
                    .entry(key)
                    .or_insert_with(|| Entity {
                        kind: "jira".to_string(),
                        id,
                        chunks_by_source: BTreeMap::new(),
                    })
                    .chunks_by_source
                    .entry(chunk.source.clone())
                    .or_default()
                    .push(i);
            }
        }

        Self { entities, chunks }
    }

    pub fn cross_source_entities(&self, min_sources: usize) -> Vec<&Entity> {
        let mut out: Vec<&Entity> = self
            .entities
            .values()
            .filter(|entity| entity.source_count() >= min_sources)
            .collect();
        out.sort_by(|a, b| {
            b.source_count()
                .cmp(&a.source_count())
                .then_with(|| b.total_mentions().cmp(&a.total_mentions()))
                .then_with(|| a.id.cmp(&b.id))
        });
        out
    }

    pub fn eval_cases(&self, min_sources: usize, max_cases: usize) -> Vec<EvalCase> {
        let mut cases = Vec::new();
        for entity in self.cross_source_entities(min_sources) {
            let Some(query) = self.make_query(entity) else {
                continue;
            };
            cases.push(EvalCase {
                entity_key: format!("{}:{}", entity.kind, entity.id),
                query,
                expected_sources: entity.sources(),
                ground_truth_indices: entity.all_chunk_indices(),
            });
            if cases.len() >= max_cases {
                break;
            }
        }
        cases
    }

    fn make_query(&self, entity: &Entity) -> Option<String> {
        if entity.kind != "jira" {
            return None;
        }

        for idx in entity
            .chunks_by_source
            .get("atlassian")
            .into_iter()
            .flatten()
        {
            let title = self.chunks[*idx].title.trim();
            if title.contains(&entity.id) {
                let clean = jira_title_prefix_re().replace(title, "").trim().to_string();
                if clean.len() > 10 {
                    return Some(clean);
                }
            }
        }

        for idx in entity
            .chunks_by_source
            .get("atlassian")
            .into_iter()
            .flatten()
        {
            let title = self.chunks[*idx].title.trim();
            if title.len() > 10 {
                return Some(title.to_string());
            }
        }

        for idx in entity.chunks_by_source.get("git").into_iter().flatten() {
            let title = self.chunks[*idx].title.trim();
            if title.contains(&entity.id) && title.len() > 10 {
                return Some(commit_prefix_re().replace(title, "").trim().to_string());
            }
        }

        None
    }
}

fn is_excluded_jira(id: &str) -> bool {
    JIRA_EXCLUDE.contains(&id) || id.starts_with("HTTP-") || id.starts_with("SHA-")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::MetaValue;

    fn chunk(source: &str, title: &str, content: &str) -> Chunk {
        Chunk {
            id: format!("{source}:{title}"),
            doc_id: format!("{source}:doc"),
            content: content.to_string(),
            title: title.to_string(),
            source: source.to_string(),
            metadata: BTreeMap::<String, MetaValue>::new(),
        }
    }

    #[test]
    fn builds_cross_source_jira_cases() {
        let chunks = vec![
            chunk(
                "atlassian",
                "PLAT-123: Fix access token refresh",
                "Ticket PLAT-123",
            ),
            chunk(
                "git",
                "fix: PLAT-123 wire access token refresh",
                "Refs PLAT-123",
            ),
            chunk("git", "chore", "Encoding UTF-8 should not count"),
        ];

        let graph = EntityGraph::build(&chunks);
        let cases = graph.eval_cases(2, 10);

        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].entity_key, "jira:PLAT-123");
        assert_eq!(cases[0].query, "Fix access token refresh");
        assert_eq!(
            cases[0].expected_sources,
            ["atlassian".to_string(), "git".to_string()]
                .into_iter()
                .collect()
        );
    }

    #[test]
    fn excludes_known_false_positive_ids() {
        let chunks = vec![chunk("git", "encoding", "UTF-8 ISO-8859 HTTP-404 SHA-256")];
        let graph = EntityGraph::build(&chunks);
        assert!(graph.entities.is_empty());
    }
}
