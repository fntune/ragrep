//! Source-specific normalizers: raw JSONL → `Document`.
//!
//! Each submodule reads one source's JSONL file and emits `Vec<Document>`.
//! `normalize_all` runs them all and dedups by `Document.id`.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use regex::Regex;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::models::Document;

pub mod atlassian;
pub mod bitbucket;
pub mod bookmark;
pub mod code;
pub mod file;
pub mod gdrive;
pub mod git;
pub mod pin;
pub mod slack;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct UserInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub title: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ChannelInfo {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub topic: String,
    #[serde(default)]
    pub purpose: String,
}

pub type UserMap = HashMap<String, UserInfo>;
pub type ChannelMap = HashMap<String, ChannelInfo>;

pub fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let r = BufReader::new(f);
    let mut out = Vec::new();
    for (i, line) in r.lines().enumerate() {
        let line = line.with_context(|| format!("reading line {} of {}", i + 1, path.display()))?;
        let s = line.trim();
        if s.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(s)
            .with_context(|| format!("parsing JSON on line {} of {}", i + 1, path.display()))?;
        out.push(v);
    }
    Ok(out)
}

pub fn load_users(path: &Path) -> Result<UserMap> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let v: Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    let mut out = HashMap::new();
    match v {
        Value::Array(arr) => {
            for item in arr {
                if let Some(id) = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                {
                    let info: UserInfo = serde_json::from_value(item).unwrap_or_default();
                    out.insert(id, info);
                }
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                let info: UserInfo = serde_json::from_value(v).unwrap_or_default();
                out.insert(k, info);
            }
        }
        _ => {}
    }
    Ok(out)
}

pub fn load_channels(path: &Path) -> Result<ChannelMap> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let raw =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let v: Value =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", path.display()))?;
    let mut out = HashMap::new();
    match v {
        Value::Array(arr) => {
            for item in arr {
                if let Some(id) = item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                {
                    let info: ChannelInfo = serde_json::from_value(item).unwrap_or_default();
                    out.insert(id, info);
                }
            }
        }
        Value::Object(map) => {
            for (k, v) in map {
                let info: ChannelInfo = serde_json::from_value(v).unwrap_or_default();
                out.insert(k, info);
            }
        }
        _ => {}
    }
    Ok(out)
}

// Compiled-once regex patterns for slack mrkdwn cleaning.
// Matches the 10 patterns at the top of `normalize.py`.

macro_rules! re {
    ($name:ident, $pattern:expr) => {
        fn $name() -> &'static Regex {
            static R: OnceLock<Regex> = OnceLock::new();
            R.get_or_init(|| Regex::new($pattern).unwrap())
        }
    };
}

re!(channel_link_re, r"<#\w+\|([^>]+)>");
re!(labeled_link_re, r"<(https?://[^|>]+)\|([^>]+)>");
re!(bare_link_re, r"<(https?://[^>]+)>");
re!(user_mention_re, r"<@(\w+)>");
re!(emoji_re, r":[\w+\-]+:");
re!(bold_re, r"\*([^*]+)\*");
re!(italic_re, r"_([^_]+)_");
re!(strike_re, r"~([^~]+)~");
re!(mailto_re, r"<mailto:([^|>]+)\|([^>]+)>");
re!(tel_re, r"<tel:([^|>]+)\|([^>]+)>");
re!(spaces_re, r"[ \t]+");

pub fn clean_text(text: &str, users: &UserMap) -> String {
    let mut s = text.to_string();
    s = user_mention_re()
        .replace_all(&s, |caps: &regex::Captures| {
            let id = &caps[1];
            let name = users
                .get(id)
                .map(|u| u.name.as_str())
                .filter(|n| !n.is_empty())
                .unwrap_or(id);
            format!("@{name}")
        })
        .into_owned();
    s = mailto_re().replace_all(&s, "$2").into_owned();
    s = tel_re().replace_all(&s, "$2").into_owned();
    s = channel_link_re().replace_all(&s, "#$1").into_owned();
    s = labeled_link_re().replace_all(&s, "$2").into_owned();
    s = bare_link_re().replace_all(&s, "$1").into_owned();
    s = emoji_re().replace_all(&s, "").into_owned();
    s = bold_re().replace_all(&s, "$1").into_owned();
    s = italic_re().replace_all(&s, "$1").into_owned();
    s = strike_re().replace_all(&s, "$1").into_owned();
    s = spaces_re().replace_all(&s, " ").into_owned();
    s.trim().to_string()
}

pub fn user_label(uid: &str, users: &UserMap) -> String {
    match users.get(uid) {
        Some(u) if !u.title.is_empty() => format!("{} ({})", u.name, u.title),
        Some(u) if !u.name.is_empty() => u.name.clone(),
        _ => uid.to_string(),
    }
}

/// Short stable hash for synthesizing IDs (12 hex chars from sha256).
pub fn short_hash(s: &str) -> String {
    let h = Sha256::digest(s.as_bytes());
    let mut out = String::with_capacity(12);
    for b in &h[..6] {
        use std::fmt::Write;
        let _ = write!(out, "{b:02x}");
    }
    out
}

/// Normalize all sources from `raw_dir` and dedup by `Document.id`.
pub fn normalize_all(raw_dir: &Path) -> Result<Vec<Document>> {
    let users = load_users(&raw_dir.join("users.json"))?;
    let channels = load_channels(&raw_dir.join("channels.json"))?;

    let mut docs: Vec<Document> = Vec::new();
    docs.extend(slack::normalize(raw_dir, &users, &channels)?);
    docs.extend(atlassian::normalize(raw_dir)?);
    docs.extend(gdrive::normalize(raw_dir)?);
    docs.extend(git::normalize(raw_dir)?);
    docs.extend(file::normalize(raw_dir)?);
    docs.extend(bookmark::normalize(raw_dir)?);
    docs.extend(pin::normalize(raw_dir, &users)?);
    docs.extend(bitbucket::normalize(raw_dir)?);
    docs.extend(code::normalize(raw_dir)?);

    Ok(dedup(docs))
}

pub fn normalize_source(raw_dir: &Path, source: &str) -> Result<Vec<Document>> {
    let docs = match source {
        "slack" => {
            let users = load_users(&raw_dir.join("users.json"))?;
            let channels = load_channels(&raw_dir.join("channels.json"))?;
            slack::normalize(raw_dir, &users, &channels)?
        }
        "atlassian" => atlassian::normalize(raw_dir)?,
        "gdrive" => gdrive::normalize(raw_dir)?,
        "git" => git::normalize(raw_dir)?,
        "file" => file::normalize(raw_dir)?,
        "bookmark" => bookmark::normalize(raw_dir)?,
        "pin" => {
            let users = load_users(&raw_dir.join("users.json"))?;
            pin::normalize(raw_dir, &users)?
        }
        "bitbucket" => bitbucket::normalize(raw_dir)?,
        "code" => code::normalize(raw_dir)?,
        _ => Vec::new(),
    };
    Ok(dedup(docs))
}

fn dedup(mut docs: Vec<Document>) -> Vec<Document> {
    let mut seen: HashSet<String> = HashSet::with_capacity(docs.len());
    docs.retain(|d| seen.insert(d.id.clone()));
    docs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn users() -> UserMap {
        let mut m = HashMap::new();
        m.insert(
            "U123".into(),
            UserInfo {
                name: "alice".into(),
                title: "engineer".into(),
            },
        );
        m
    }

    #[test]
    fn user_label_with_title() {
        assert_eq!(user_label("U123", &users()), "alice (engineer)");
    }

    #[test]
    fn user_label_unknown_falls_back_to_id() {
        assert_eq!(user_label("U999", &users()), "U999");
    }

    #[test]
    fn clean_text_strips_mrkdwn() {
        let txt = "Hi <@U123>, see <https://example.com|the docs> *now* :smile:";
        let cleaned = clean_text(txt, &users());
        assert_eq!(cleaned, "Hi @alice, see the docs now");
    }

    #[test]
    fn clean_text_strips_channel_link_and_emoji_and_bold_and_italic() {
        let txt = "*bold* _italic_ ~strike~ in <#C1|general> :emoji:";
        let cleaned = clean_text(txt, &HashMap::new());
        assert_eq!(cleaned, "bold italic strike in #general");
    }

    #[test]
    fn clean_text_collapses_whitespace() {
        let txt = "a    b\tc";
        let cleaned = clean_text(txt, &HashMap::new());
        assert_eq!(cleaned, "a b c");
    }

    #[test]
    fn short_hash_is_stable_and_12_chars() {
        let a = short_hash("hello");
        let b = short_hash("hello");
        assert_eq!(a, b);
        assert_eq!(a.len(), 12);
        assert_ne!(short_hash("hello"), short_hash("world"));
    }

    #[test]
    fn normalize_source_ignores_unrelated_raw_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("git.jsonl"), "{not json").unwrap();
        std::fs::write(
            dir.path().join("bookmarks.jsonl"),
            r#"{"title":"Docs","link":"https://example.com","channel_name":"kb"}"#,
        )
        .unwrap();

        let docs = normalize_source(dir.path(), "bookmark").unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].source, "bookmark");
        assert_eq!(docs[0].title, "Docs");
    }
}
