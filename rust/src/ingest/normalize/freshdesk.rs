//! Freshdesk support articles → Document.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use serde_json::Value;

use super::read_jsonl;
use crate::models::{Document, MetaValue};

pub fn normalize(raw_dir: &Path) -> Result<Vec<Document>> {
    let records = read_jsonl(&raw_dir.join("freshdesk.jsonl"))?;
    let mut out = Vec::with_capacity(records.len());
    for record in records {
        let article_id = scalar_string(record.get("id"));
        let content = scalar_string(record.get("content"));
        if article_id.is_empty() || content.is_empty() {
            continue;
        }

        let title = scalar_string(record.get("title"));
        let mut metadata = BTreeMap::new();
        metadata.insert("article_id".into(), MetaValue::Str(article_id.clone()));
        insert_str(&mut metadata, "category_id", record.get("category_id"));
        insert_str(&mut metadata, "category_name", record.get("category_name"));
        insert_str(&mut metadata, "folder_id", record.get("folder_id"));
        insert_str(&mut metadata, "folder_name", record.get("folder_name"));
        insert_str(&mut metadata, "portal_id", record.get("portal_id"));
        insert_i64(&mut metadata, "status", record.get("status"));
        insert_i64(&mut metadata, "thumbs_up", record.get("thumbs_up"));
        insert_i64(&mut metadata, "thumbs_down", record.get("thumbs_down"));
        insert_i64(&mut metadata, "hits", record.get("hits"));
        insert_str(&mut metadata, "created_at", record.get("created_at"));
        insert_str(&mut metadata, "updated_at", record.get("updated_at"));
        insert_str(&mut metadata, "link", record.get("link"));

        out.push(Document {
            id: format!("freshdesk:{article_id}"),
            source: "freshdesk".into(),
            content,
            title,
            metadata,
        });
    }
    Ok(out)
}

fn insert_str(metadata: &mut BTreeMap<String, MetaValue>, key: &str, value: Option<&Value>) {
    let value = scalar_string(value);
    if !value.is_empty() {
        metadata.insert(key.to_string(), MetaValue::Str(value));
    }
}

fn insert_i64(metadata: &mut BTreeMap<String, MetaValue>, key: &str, value: Option<&Value>) {
    if let Some(value) = scalar_i64(value) {
        metadata.insert(key.to_string(), MetaValue::Int(value));
    }
}

fn scalar_i64(value: Option<&Value>) -> Option<i64> {
    match value {
        Some(Value::Number(number)) => number.as_i64(),
        Some(Value::String(value)) => value.parse().ok(),
        _ => None,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_article_records() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("freshdesk.jsonl"),
            r#"{"id":123,"title":"KYC","content":"How KYC works","status":2,"portal_id":"800","folder_name":"Account","updated_at":"2026-05-16","link":"https://support/123"}"#,
        )
        .unwrap();

        let docs = normalize(dir.path()).unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, "freshdesk:123");
        assert_eq!(docs[0].source, "freshdesk");
        assert_eq!(docs[0].title, "KYC");
        assert_eq!(
            docs[0].metadata.get("article_id"),
            Some(&MetaValue::Str("123".to_string()))
        );
        assert_eq!(docs[0].metadata.get("status"), Some(&MetaValue::Int(2)));
        assert_eq!(
            docs[0].metadata.get("portal_id"),
            Some(&MetaValue::Str("800".to_string()))
        );
    }
}
