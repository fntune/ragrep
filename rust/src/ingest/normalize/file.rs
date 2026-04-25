//! Extracted file content (PDF/DOCX/PPTX/XLSX → text) → Document.

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
    file_id: String,
    #[serde(default = "default_name")]
    file_name: String,
    #[serde(default)]
    file_type: String,
    #[serde(default)]
    extraction_method: String,
    #[serde(default)]
    channel_name: String,
}

fn default_name() -> String {
    "unknown".into()
}

pub fn normalize(raw_dir: &Path) -> Result<Vec<Document>> {
    let records = read_jsonl(&raw_dir.join("files_extracted.jsonl"))?;
    let mut out = Vec::with_capacity(records.len());
    for v in records {
        let r: Raw = match serde_json::from_value(v) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let content = r.content.trim().to_string();
        if content.len() < 20 {
            continue;
        }
        let mut metadata = BTreeMap::new();
        metadata.insert("file_type".into(), MetaValue::Str(r.file_type));
        metadata.insert(
            "extraction_method".into(),
            MetaValue::Str(r.extraction_method),
        );
        metadata.insert("channel_name".into(), MetaValue::Str(r.channel_name));
        out.push(Document {
            id: format!("file:{}", r.file_id),
            source: "file".into(),
            content,
            title: r.file_name,
            metadata,
        });
    }
    Ok(out)
}
