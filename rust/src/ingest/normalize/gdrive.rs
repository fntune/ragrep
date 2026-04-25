//! Google Drive docs/sheets/slides → Document. Skips spreadsheets.
//!
//! Mirrors `normalize_gdrive` in `normalize.py:219`.

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
    #[serde(default)]
    name: String,
    #[serde(default)]
    file_id: String,
    #[serde(default)]
    file_type: String,
    #[serde(default)]
    mime_type: String,
    #[serde(default)]
    path: String,
}

pub fn normalize(raw_dir: &Path) -> Result<Vec<Document>> {
    let records = read_jsonl(&raw_dir.join("gdrive.jsonl"))?;
    let mut out = Vec::with_capacity(records.len());
    for v in records {
        let r: Raw = serde_json::from_value(v).unwrap_or_default();

        // Skip spreadsheets / Google Sheets.
        if r.mime_type.contains("spreadsheet") || r.file_type == "Google Sheet" {
            continue;
        }
        let content = r.content.trim().to_string();
        if content.len() < 20 {
            continue;
        }

        let title: String = r.name.chars().take(120).collect();
        let file_id = if r.file_id.is_empty() {
            short_hash(&title)
        } else {
            r.file_id
        };
        let doc_type = r.file_type.to_lowercase().replace("google ", "");

        let mut metadata = BTreeMap::new();
        metadata.insert("path".into(), MetaValue::Str(r.path));
        metadata.insert("doc_type".into(), MetaValue::Str(doc_type));

        out.push(Document {
            id: format!("gdrive:{file_id}"),
            source: "gdrive".into(),
            content,
            title,
            metadata,
        });
    }
    Ok(out)
}
