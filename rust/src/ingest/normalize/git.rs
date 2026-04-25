//! Git commit log → Document.
//!
//! Mirrors `normalize_git` in `normalize.py:251`. Includes commit message,
//! author/date, PR + branch, files changed, and diff (or diff_stat as
//! fallback) so the chunker has rich content to embed.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use super::{read_jsonl, short_hash};
use crate::models::{Document, MetaValue};

#[derive(Deserialize, Default)]
struct Raw {
    #[serde(default)]
    subject: String,
    #[serde(default)]
    body: String,
    #[serde(default)]
    repo: String,
    #[serde(default)]
    hash: String,
    #[serde(default)]
    author: String,
    #[serde(default)]
    date: String,
    #[serde(default)]
    diff: String,
    #[serde(default)]
    diff_stat: String,
    #[serde(default)]
    files_changed: Vec<FileChange>,
    #[serde(default)]
    pr_number: Option<u64>,
    #[serde(default)]
    branch: String,
}

#[derive(Deserialize, Default)]
struct FileChange {
    #[serde(default)]
    status: String,
    #[serde(default)]
    path: String,
}

pub fn normalize(raw_dir: &Path) -> Result<Vec<Document>> {
    let records = read_jsonl(&raw_dir.join("git.jsonl"))?;
    let mut out = Vec::with_capacity(records.len());
    for v in records {
        let r: Raw = serde_json::from_value(v).unwrap_or_default();
        let subject = r.subject.trim();
        if subject.is_empty() && r.body.trim().is_empty() && r.diff.is_empty() {
            continue;
        }

        let mut parts: Vec<String> = Vec::new();
        if !subject.is_empty() {
            parts.push(subject.to_string());
        }
        let body = r.body.trim();
        if !body.is_empty() {
            parts.push(body.to_string());
        }
        parts.push(format!("Author: {}", r.author));
        parts.push(format!("Date: {}", r.date));
        if let Some(pr) = r.pr_number {
            parts.push(format!("PR: #{pr}"));
        }
        if !r.branch.is_empty() {
            parts.push(format!("Branch: {}", r.branch));
        }
        if !r.files_changed.is_empty() {
            let files: Vec<String> = r
                .files_changed
                .iter()
                .map(|f| format!("  {} {}", f.status, f.path))
                .collect();
            parts.push(format!("Files changed:\n{}", files.join("\n")));
        }
        if !r.diff.is_empty() {
            parts.push(format!("Diff:\n{}", r.diff));
        } else if !r.diff_stat.is_empty() {
            parts.push(format!("Diff stat:\n{}", r.diff_stat));
        }

        let content = parts.join("\n");
        if content.len() < 10 {
            continue;
        }

        let title = if let Some(pr) = r.pr_number {
            let subj_prefix: String = subject.chars().take(70).collect();
            format!("{} PR#{}: {}", r.repo, pr, subj_prefix)
        } else {
            let subj_prefix: String = subject.chars().take(80).collect();
            format!("{}: {}", r.repo, subj_prefix)
        };

        let id = if !r.hash.is_empty() {
            format!("git:{}:{}", r.repo, r.hash)
        } else {
            format!("git:{}", short_hash(&content))
        };

        let mut metadata = BTreeMap::new();
        metadata.insert("repo".into(), MetaValue::Str(r.repo.clone()));
        metadata.insert("author".into(), MetaValue::Str(r.author));
        metadata.insert("date".into(), MetaValue::Str(r.date));
        if let Some(pr) = r.pr_number {
            metadata.insert("pr_number".into(), MetaValue::Int(pr as i64));
        }
        if !r.branch.is_empty() {
            metadata.insert("branch".into(), MetaValue::Str(r.branch));
        }
        if !r.files_changed.is_empty() {
            metadata.insert(
                "files_count".into(),
                MetaValue::Int(r.files_changed.len() as i64),
            );
        }

        out.push(Document {
            id,
            source: "git".into(),
            content,
            title,
            metadata,
        });
    }
    Ok(out)
}
