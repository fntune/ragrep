//! Pinned Slack messages → Document.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;

use super::{clean_text, read_jsonl, UserMap};
use crate::models::{Document, MetaValue};

#[derive(Deserialize)]
struct Raw {
    #[serde(default)]
    text: String,
    #[serde(default)]
    channel_name: String,
    #[serde(default = "default_ts")]
    ts: String,
}

fn default_ts() -> String {
    "0".into()
}

pub fn normalize(raw_dir: &Path, users: &UserMap) -> Result<Vec<Document>> {
    let records = read_jsonl(&raw_dir.join("pins.jsonl"))?;
    let mut out = Vec::with_capacity(records.len());
    for v in records {
        let r: Raw = match serde_json::from_value(v) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let text = clean_text(&r.text, users);
        if text.len() < 20 {
            continue;
        }
        let title = format!("Pinned in #{}: {}", r.channel_name, preview(&text, 60));
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "channel_name".into(),
            MetaValue::Str(r.channel_name.clone()),
        );
        out.push(Document {
            id: format!("pin:{}:{}", r.channel_name, r.ts),
            source: "pin".into(),
            content: text,
            title,
            metadata,
        });
    }
    Ok(out)
}

fn preview(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}
