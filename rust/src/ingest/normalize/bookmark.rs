//! Slack bookmarks → Document.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use super::{read_jsonl, short_hash};
use crate::models::{Document, MetaValue};

#[derive(Deserialize)]
struct Raw {
    #[serde(default)]
    title: String,
    #[serde(default)]
    link: String,
    #[serde(default)]
    channel_name: String,
}

pub fn normalize(raw_dir: &Path) -> Result<Vec<Document>> {
    let records = read_jsonl(&raw_dir.join("bookmarks.jsonl"))?;
    let mut out = Vec::with_capacity(records.len());
    for v in records {
        let r: Raw = match serde_json::from_value(v) {
            Ok(r) => r,
            Err(_) => continue,
        };
        if r.title.is_empty() || r.link.is_empty() {
            continue;
        }
        let content = format!("{}: {}", r.title, r.link);
        let mut metadata = BTreeMap::new();
        metadata.insert("channel_name".into(), MetaValue::Str(r.channel_name));
        metadata.insert("link".into(), MetaValue::Str(r.link.clone()));
        out.push(Document {
            id: format!("bookmark:{}", short_hash(&r.link)),
            source: "bookmark".into(),
            content,
            title: r.title,
            metadata,
        });
    }
    Ok(out)
}
