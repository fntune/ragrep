//! Bitbucket PR descriptions + comments → Document.
//!
//! Mirrors `normalize_bitbucket` in `normalize.py:447`.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use super::read_jsonl;
use crate::models::{Document, MetaValue};

#[derive(Deserialize, Default)]
struct Raw {
    #[serde(default)]
    repo: String,
    #[serde(default)]
    pr_id: u64,
    #[serde(default)]
    title: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    author: String,
    #[serde(default)]
    state: String,
    #[serde(default)]
    source_branch: String,
    #[serde(default)]
    target_branch: String,
    #[serde(default)]
    comments: Vec<Comment>,
    #[serde(default)]
    approvals: Vec<String>,
}

#[derive(Deserialize, Default)]
struct Comment {
    #[serde(default)]
    author: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    path: Option<String>,
}

pub fn normalize(raw_dir: &Path) -> Result<Vec<Document>> {
    let records = read_jsonl(&raw_dir.join("bitbucket.jsonl"))?;
    let mut out = Vec::with_capacity(records.len());
    for v in records {
        let r: Raw = serde_json::from_value(v).unwrap_or_default();

        let mut parts = Vec::new();
        let description = r.description.trim();
        if !description.is_empty() {
            parts.push(format!("{}: {}", r.author, description));
        }
        for c in &r.comments {
            if c.content.is_empty() {
                continue;
            }
            let prefix = match c.path.as_deref() {
                Some(p) if !p.is_empty() => format!("{} on {p}", c.author),
                _ => c.author.clone(),
            };
            parts.push(format!("{prefix}: {}", c.content));
        }

        let content = parts.join("\n");
        if content.len() < 20 {
            continue;
        }

        let title_full = format!("{} PR#{}: {}", r.repo, r.pr_id, r.title);
        let title: String = title_full.chars().take(120).collect();

        let mut metadata = BTreeMap::new();
        metadata.insert("repo".into(), MetaValue::Str(r.repo.clone()));
        metadata.insert("state".into(), MetaValue::Str(r.state));
        metadata.insert("source_branch".into(), MetaValue::Str(r.source_branch));
        metadata.insert("target_branch".into(), MetaValue::Str(r.target_branch));
        metadata.insert(
            "comment_count".into(),
            MetaValue::Int(r.comments.len() as i64),
        );
        metadata.insert("approvals".into(), MetaValue::Str(r.approvals.join(", ")));

        out.push(Document {
            id: format!("bitbucket:{}:{}", r.repo, r.pr_id),
            source: "bitbucket".into(),
            content,
            title,
            metadata,
        });
    }
    Ok(out)
}
