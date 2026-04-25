//! Source code files → Document.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use super::read_jsonl;
use crate::models::{Document, MetaValue};

#[derive(Deserialize)]
struct Raw {
    #[serde(default)]
    content: String,
    #[serde(default)]
    repo: String,
    #[serde(default)]
    path: String,
    #[serde(default)]
    language: String,
    #[serde(default)]
    size_bytes: u64,
}

pub fn normalize(raw_dir: &Path) -> Result<Vec<Document>> {
    let records = read_jsonl(&raw_dir.join("code.jsonl"))?;
    let mut out = Vec::with_capacity(records.len());
    for v in records {
        let r: Raw = match serde_json::from_value(v) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if r.content.len() < 100 {
            continue;
        }
        let prefixed = format!("# {}/{}\n\n{}", r.repo, r.path, r.content);
        let title = format!("{}/{}", r.repo, r.path);
        let id = format!("code:{}:{}", r.repo, r.path);
        let mut metadata = BTreeMap::new();
        metadata.insert("repo".into(), MetaValue::Str(r.repo));
        metadata.insert("path".into(), MetaValue::Str(r.path));
        metadata.insert("language".into(), MetaValue::Str(r.language));
        metadata.insert("size_bytes".into(), MetaValue::Int(r.size_bytes as i64));
        out.push(Document {
            id,
            source: "code".into(),
            content: prefixed,
            title,
            metadata,
        });
    }
    Ok(out)
}
