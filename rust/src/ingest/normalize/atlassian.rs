//! Confluence pages + Jira issues → Document.
//!
//! Both source types live in the same `atlassian.jsonl`; the `type` field
//! discriminates. Mirrors `normalize_atlassian` in `normalize.py:187`.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use super::{read_jsonl, short_hash};
use crate::models::{Document, MetaValue};

#[derive(Deserialize, Default)]
struct Raw {
    #[serde(default)]
    content: String,
    #[serde(default, rename = "type")]
    source_type: String,
    #[serde(default)]
    key: String,
    #[serde(default)]
    summary: String,
    #[serde(default)]
    project: String,
    #[serde(default)]
    title: String,
    #[serde(default)]
    space: String,
}

pub fn normalize(raw_dir: &Path) -> Result<Vec<Document>> {
    let records = read_jsonl(&raw_dir.join("atlassian.jsonl"))?;
    let mut out = Vec::with_capacity(records.len());
    for v in records {
        let r: Raw = serde_json::from_value(v).unwrap_or_default();
        if r.content.len() < 20 {
            continue;
        }

        let source_type = if r.source_type.is_empty() {
            "atlassian".to_string()
        } else {
            r.source_type.clone()
        };

        let (title, space) = if source_type == "jira" {
            (format!("{}: {}", r.key, r.summary), r.project)
        } else {
            (r.title, r.space)
        };

        let title_truncated: String = title.chars().take(120).collect();
        let id = format!("atlassian:{}", short_hash(&title));

        let mut metadata = BTreeMap::new();
        metadata.insert("source_type".into(), MetaValue::Str(source_type));
        metadata.insert("space".into(), MetaValue::Str(space));

        out.push(Document {
            id,
            source: "atlassian".into(),
            content: r.content,
            title: title_truncated,
            metadata,
        });
    }
    Ok(out)
}
