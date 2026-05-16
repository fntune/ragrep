//! YouTube support videos → Document.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use super::read_jsonl;
use crate::models::{Document, MetaValue};

pub fn normalize(raw_dir: &Path) -> Result<Vec<Document>> {
    let records = read_jsonl(&raw_dir.join("youtube.jsonl"))?;
    let mut out = Vec::with_capacity(records.len());
    for record in records {
        let video_id = non_empty_string(record.get("video_id"))
            .or_else(|| non_empty_string(record.get("id")))
            .unwrap_or_default();
        let title = scalar_string(record.get("title"));
        let description = scalar_string(record.get("description"));
        let content = non_empty_string(record.get("content")).unwrap_or_else(|| {
            if title.is_empty() {
                description.clone()
            } else if description.is_empty() {
                title.clone()
            } else {
                format!("{title}. {description}")
            }
        });
        if video_id.is_empty() || content.is_empty() {
            continue;
        }

        let mut metadata = BTreeMap::new();
        metadata.insert("video_id".into(), MetaValue::Str(video_id.clone()));
        metadata.insert("title".into(), MetaValue::Str(title.clone()));
        insert_str(&mut metadata, "description", record.get("description"));
        insert_str(&mut metadata, "video_url", record.get("video_url"));
        insert_str(&mut metadata, "thumbnail_url", record.get("thumbnail_url"));
        insert_str(&mut metadata, "published_at", record.get("published_at"));
        insert_str(&mut metadata, "playlist_id", record.get("playlist_id"));
        insert_str(&mut metadata, "channel_id", record.get("channel_id"));
        insert_str(&mut metadata, "updated_at", record.get("updated_at"));

        out.push(Document {
            id: format!("youtube:{video_id}"),
            source: "youtube".into(),
            content,
            title,
            metadata,
        });
    }
    Ok(out)
}

fn insert_str(metadata: &mut BTreeMap<String, MetaValue>, key: &str, value: Option<&Value>) {
    if let Some(value) = non_empty_string(value) {
        metadata.insert(key.to_string(), MetaValue::Str(value));
    }
}

fn scalar_string(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => value.trim().to_string(),
        Some(Value::Number(value)) => value.to_string(),
        Some(Value::Bool(value)) => value.to_string(),
        _ => String::new(),
    }
}

fn non_empty_string(value: Option<&Value>) -> Option<String> {
    let value = scalar_string(value);
    (!value.is_empty()).then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_video_records() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("youtube.jsonl"),
            r#"{"video_id":"v1","title":"Deposits","description":"How deposits work","video_url":"https://youtu.be/v1","thumbnail_url":"https://img/v1.jpg","playlist_id":"p1"}"#,
        )
        .unwrap();

        let docs = normalize(dir.path()).unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, "youtube:v1");
        assert_eq!(docs[0].source, "youtube");
        assert_eq!(docs[0].content, "Deposits. How deposits work");
        assert_eq!(
            docs[0].metadata.get("video_id"),
            Some(&MetaValue::Str("v1".to_string()))
        );
        assert_eq!(
            docs[0].metadata.get("playlist_id"),
            Some(&MetaValue::Str("p1".to_string()))
        );
    }
}
