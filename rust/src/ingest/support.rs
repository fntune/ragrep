//! Support knowledge records stored as raw JSONL.
//!
//! Freshdesk articles and YouTube videos are support-owned sources. This module
//! owns their raw record files so HTTP writes and source-scoped ingest mutate the
//! same data that the normalizers consume.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use fs2::FileExt;
use serde_json::{Map, Value};

const LOCK_FILE: &str = ".support-records.lock";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Freshdesk,
    Youtube,
}

impl Source {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "freshdesk" => Ok(Self::Freshdesk),
            "youtube" => Ok(Self::Youtube),
            other => bail!("invalid support source: {other}"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Freshdesk => "freshdesk",
            Self::Youtube => "youtube",
        }
    }

    fn file_name(self) -> &'static str {
        match self {
            Self::Freshdesk => "freshdesk.jsonl",
            Self::Youtube => "youtube.jsonl",
        }
    }

    fn id_field(self) -> &'static str {
        match self {
            Self::Freshdesk => "id",
            Self::Youtube => "video_id",
        }
    }

    fn id_fields(self) -> &'static [&'static str] {
        match self {
            Self::Freshdesk => &["id"],
            Self::Youtube => &["video_id", "id"],
        }
    }

    fn validate_record(self, id: &str, map: &Map<String, Value>) -> Result<()> {
        match self {
            Self::Freshdesk => {
                if !has_text(map, "content") {
                    bail!("invalid freshdesk record {id}: content is required");
                }
            }
            Self::Youtube => {
                if !has_text(map, "content")
                    && !has_text(map, "title")
                    && !has_text(map, "description")
                {
                    bail!(
                        "invalid youtube record {id}: content, title, or description is required"
                    );
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct ListFilter {
    pub playlist_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WriteResult {
    pub changed: bool,
    pub upserted: usize,
    pub deleted: usize,
    pub unchanged: usize,
    pub total_records: usize,
}

pub fn list(raw_dir: &Path, source: Source, filter: &ListFilter) -> Result<Vec<Value>> {
    let _lock = lock(raw_dir)?;
    let records = read_records(raw_dir, source)?;
    Ok(records
        .into_values()
        .filter(|record| matches_filter(record, filter))
        .collect())
}

pub fn fetch(raw_dir: &Path, source: Source, id: &str) -> Result<Option<Value>> {
    let id = normalize_id(id)?;
    let _lock = lock(raw_dir)?;
    let records = read_records(raw_dir, source)?;
    Ok(records.get(&id).cloned())
}

pub fn upsert(raw_dir: &Path, source: Source, id: &str, record: Value) -> Result<WriteResult> {
    apply(
        raw_dir,
        source,
        vec![record_with_path_id(source, id, record)?],
        Vec::new(),
    )
}

pub fn delete(raw_dir: &Path, source: Source, id: &str) -> Result<WriteResult> {
    apply(raw_dir, source, Vec::new(), vec![id.to_string()])
}

pub fn apply(
    raw_dir: &Path,
    source: Source,
    upserts: Vec<Value>,
    deletes: Vec<String>,
) -> Result<WriteResult> {
    let _lock = lock(raw_dir)?;
    let mut records = read_records(raw_dir, source)?;
    let mut changed = false;
    let mut upserted = 0;
    let mut deleted = 0;
    let mut unchanged = 0;

    for raw_id in deletes {
        let id = normalize_id(&raw_id)?;
        if records.remove(&id).is_some() {
            changed = true;
            deleted += 1;
        } else {
            unchanged += 1;
        }
    }

    for record in upserts {
        let (id, record) = normalize_record(source, None, record)?;
        if records.get(&id) == Some(&record) {
            unchanged += 1;
            continue;
        }
        records.insert(id, record);
        changed = true;
        upserted += 1;
    }

    if changed {
        write_records(raw_dir, source, &records)?;
    }

    Ok(WriteResult {
        changed,
        upserted,
        deleted,
        unchanged,
        total_records: records.len(),
    })
}

fn record_with_path_id(source: Source, id: &str, record: Value) -> Result<Value> {
    let (_, record) = normalize_record(source, Some(id), record)?;
    Ok(record)
}

fn normalize_record(
    source: Source,
    path_id: Option<&str>,
    record: Value,
) -> Result<(String, Value)> {
    let Value::Object(mut map) = record else {
        bail!("invalid {} record: expected JSON object", source.as_str());
    };

    let body_id = source
        .id_fields()
        .iter()
        .filter_map(|field| {
            map.get(*field)
                .and_then(scalar_string)
                .map(|id| (*field, id))
        })
        .next();

    let id = match (path_id, body_id.as_ref()) {
        (Some(path_id), Some((_, body_id))) => {
            let path_id = normalize_id(path_id)?;
            if &path_id != body_id {
                bail!(
                    "invalid {} record id: path id {} does not match body id {}",
                    source.as_str(),
                    path_id,
                    body_id
                );
            }
            path_id
        }
        (Some(path_id), None) => normalize_id(path_id)?,
        (None, Some((_, body_id))) => normalize_id(body_id)?,
        (None, None) => bail!(
            "invalid {} record: {} is required",
            source.as_str(),
            source.id_field()
        ),
    };

    for field in source.id_fields() {
        if let Some(value) = map.get(*field).and_then(scalar_string) {
            if value != id {
                bail!(
                    "invalid {} record id: field {} has {}, expected {}",
                    source.as_str(),
                    field,
                    value,
                    id
                );
            }
        }
    }

    for field in source.id_fields() {
        if *field != source.id_field() {
            map.remove(*field);
        }
    }
    map.insert(source.id_field().to_string(), Value::String(id.clone()));
    source.validate_record(&id, &map)?;

    Ok((id, Value::Object(map)))
}

fn read_records(raw_dir: &Path, source: Source) -> Result<BTreeMap<String, Value>> {
    let path = raw_dir.join(source.file_name());
    if !path.exists() {
        return Ok(BTreeMap::new());
    }
    let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut records = BTreeMap::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {} of {}", i + 1, path.display()))?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: Value = serde_json::from_str(line)
            .with_context(|| format!("parsing JSON on line {} of {}", i + 1, path.display()))?;
        let (id, record) = normalize_record(source, None, record)
            .with_context(|| format!("validating line {} of {}", i + 1, path.display()))?;
        records.insert(id, record);
    }
    Ok(records)
}

fn write_records(raw_dir: &Path, source: Source, records: &BTreeMap<String, Value>) -> Result<()> {
    std::fs::create_dir_all(raw_dir).with_context(|| format!("creating {}", raw_dir.display()))?;
    let path = raw_dir.join(source.file_name());
    let tmp = temp_file_path(&path);
    let result = (|| -> Result<()> {
        let file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        let mut writer = BufWriter::new(file);
        for record in records.values() {
            serde_json::to_writer(&mut writer, record)
                .with_context(|| format!("serializing record for {}", tmp.display()))?;
            writer.write_all(b"\n")?;
        }
        writer.flush()?;
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("renaming {} -> {}", tmp.display(), path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

fn matches_filter(record: &Value, filter: &ListFilter) -> bool {
    if let Some(playlist_id) = filter.playlist_id.as_deref() {
        return record
            .get("playlist_id")
            .and_then(scalar_string)
            .is_some_and(|value| value == playlist_id);
    }
    true
}

fn normalize_id(id: &str) -> Result<String> {
    let id = id.trim();
    if id.is_empty() {
        bail!("invalid support record id: id is required");
    }
    Ok(id.to_string())
}

fn has_text(map: &Map<String, Value>, key: &str) -> bool {
    map.get(key)
        .and_then(scalar_string)
        .is_some_and(|value| !value.is_empty())
}

fn scalar_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => {
            let value = value.trim();
            (!value.is_empty()).then(|| value.to_string())
        }
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        _ => None,
    }
}

fn temp_file_path(path: &Path) -> PathBuf {
    path.with_extension(format!("tmp-{}-{}", std::process::id(), unique_suffix()))
}

fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default()
}

struct RawLock(File);

fn lock(raw_dir: &Path) -> Result<RawLock> {
    std::fs::create_dir_all(raw_dir).with_context(|| format!("creating {}", raw_dir.display()))?;
    let path = raw_dir.join(LOCK_FILE);
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    file.lock_exclusive()
        .with_context(|| format!("locking {}", path.display()))?;
    Ok(RawLock(file))
}

impl Drop for RawLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn upsert_fetch_list_and_delete_freshdesk_record() {
        let dir = tempfile::tempdir().unwrap();

        let result = upsert(
            dir.path(),
            Source::Freshdesk,
            "123",
            json!({"title":"KYC","content":"How KYC works","status":2}),
        )
        .unwrap();

        assert!(result.changed);
        assert_eq!(result.upserted, 1);
        assert_eq!(result.total_records, 1);
        assert_eq!(
            fetch(dir.path(), Source::Freshdesk, "123")
                .unwrap()
                .and_then(|record| record.get("id").cloned()),
            Some(json!("123"))
        );
        assert_eq!(
            list(dir.path(), Source::Freshdesk, &ListFilter::default())
                .unwrap()
                .len(),
            1
        );

        let result = delete(dir.path(), Source::Freshdesk, "123").unwrap();

        assert!(result.changed);
        assert_eq!(result.deleted, 1);
        assert_eq!(result.total_records, 0);
        assert!(fetch(dir.path(), Source::Freshdesk, "123")
            .unwrap()
            .is_none());
    }

    #[test]
    fn youtube_records_use_video_id_and_filter_by_playlist() {
        let dir = tempfile::tempdir().unwrap();

        apply(
            dir.path(),
            Source::Youtube,
            vec![
                json!({"id":"v1","title":"Deposits","playlist_id":"p1"}),
                json!({"video_id":"v2","title":"Withdrawals","playlist_id":"p2"}),
            ],
            Vec::new(),
        )
        .unwrap();

        let records = list(
            dir.path(),
            Source::Youtube,
            &ListFilter {
                playlist_id: Some("p1".into()),
            },
        )
        .unwrap();

        assert_eq!(records.len(), 1);
        assert_eq!(records[0].get("video_id"), Some(&json!("v1")));
        assert!(records[0].get("id").is_none());
    }

    #[test]
    fn unchanged_upsert_does_not_rewrite() {
        let dir = tempfile::tempdir().unwrap();
        let record = json!({"id":"123","title":"KYC","content":"How KYC works"});

        upsert(dir.path(), Source::Freshdesk, "123", record.clone()).unwrap();
        let result = upsert(dir.path(), Source::Freshdesk, "123", record).unwrap();

        assert!(!result.changed);
        assert_eq!(result.unchanged, 1);
        assert_eq!(result.total_records, 1);
    }

    #[test]
    fn rejects_conflicting_path_and_body_id() {
        let dir = tempfile::tempdir().unwrap();

        let err = upsert(
            dir.path(),
            Source::Freshdesk,
            "123",
            json!({"id":"456","title":"KYC","content":"How KYC works"}),
        )
        .unwrap_err();

        assert!(format!("{err:#}").contains("path id 123 does not match body id 456"));
    }

    #[test]
    fn rejects_records_without_indexable_text() {
        let dir = tempfile::tempdir().unwrap();

        let err = upsert(dir.path(), Source::Youtube, "v1", json!({"title":""})).unwrap_err();

        assert!(format!("{err:#}").contains("content, title, or description is required"));
    }
}
