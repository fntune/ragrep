//! Slack messages → Document.
//!
//! Mirrors `normalize_messages` in `normalize.py:97`. Threads are grouped by
//! `thread_ts`, sorted by message timestamp, and concatenated as
//! `"{user_label}: {clean_text}"` lines. Standalone messages get their own
//! Documents with `message_count=1`.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::Result;
use serde::Deserialize;
use time::macros::format_description;
use time::OffsetDateTime;

use super::{clean_text, read_jsonl, user_label, ChannelMap, UserMap};
use crate::models::{Document, MetaValue};

const SYSTEM_SUBTYPES: &[&str] = &[
    "channel_join",
    "channel_leave",
    "channel_topic",
    "channel_purpose",
    "channel_name",
    "bot_message",
    "file_comment",
    "tombstone",
];

#[derive(Deserialize, Default, Clone)]
struct Msg {
    #[serde(default)]
    subtype: String,
    #[serde(default)]
    thread_ts: String,
    #[serde(default)]
    ts: String,
    #[serde(default)]
    user_id: String,
    #[serde(default)]
    text: String,
    #[serde(default)]
    channel_name: String,
    #[serde(default)]
    channel_id: String,
    #[serde(default)]
    reaction_count: u32,
    #[serde(default)]
    reply_count: u32,
}

pub fn normalize(raw_dir: &Path, users: &UserMap, channels: &ChannelMap) -> Result<Vec<Document>> {
    let records = read_jsonl(&raw_dir.join("messages.jsonl"))?;
    let mut threads: HashMap<String, Vec<Msg>> = HashMap::new();
    let mut standalone: Vec<Msg> = Vec::new();

    for v in records {
        let m: Msg = match serde_json::from_value(v) {
            Ok(m) => m,
            Err(_) => continue,
        };
        if SYSTEM_SUBTYPES.contains(&m.subtype.as_str()) {
            continue;
        }
        if !m.thread_ts.is_empty() {
            threads.entry(m.thread_ts.clone()).or_default().push(m);
        } else if m.reply_count > 0 {
            // Thread parent without thread_ts in replies — key by its own ts.
            threads.entry(m.ts.clone()).or_default().push(m);
        } else {
            standalone.push(m);
        }
    }

    let mut out = Vec::with_capacity(threads.len() + standalone.len());

    // Threads
    for (thread_ts, mut msgs) in threads {
        msgs.sort_by(|a, b| {
            ts_to_f64(&a.ts)
                .partial_cmp(&ts_to_f64(&b.ts))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let channel = if msgs[0].channel_name.is_empty() {
            "unknown".to_string()
        } else {
            msgs[0].channel_name.clone()
        };
        let channel_id = msgs[0].channel_id.clone();

        let mut parts = Vec::new();
        for m in &msgs {
            let label = user_label(&m.user_id, users);
            let text = clean_text(&m.text, users);
            if !text.is_empty() {
                parts.push(format!("{label}: {text}"));
            }
        }
        let content = parts.join("\n");
        if content.len() < 20 {
            continue;
        }

        let first_text = clean_text(&msgs[0].text, users);
        let title_prefix: String = first_text.chars().take(80).collect();
        let title = format!("#{channel}: {title_prefix}");

        let mut metadata = BTreeMap::new();
        metadata.insert("channel_name".into(), MetaValue::Str(channel.clone()));
        metadata.insert("message_count".into(), MetaValue::Int(msgs.len() as i64));
        let reaction_total: u32 = msgs.iter().map(|m| m.reaction_count).sum();
        metadata.insert(
            "reaction_count".into(),
            MetaValue::Int(reaction_total as i64),
        );
        let first_ts = ts_to_f64(&msgs[0].ts);
        if let Some(date) = ts_to_iso_date(first_ts) {
            metadata.insert("date".into(), MetaValue::Str(date));
        }
        if let Some(ch) = channels.get(&channel_id) {
            if !ch.topic.is_empty() {
                metadata.insert("channel_topic".into(), MetaValue::Str(ch.topic.clone()));
            }
        }

        out.push(Document {
            id: format!("slack:{channel}:{thread_ts}"),
            source: "slack".into(),
            content,
            title,
            metadata,
        });
    }

    // Standalone messages
    for m in standalone {
        let text = clean_text(&m.text, users);
        if text.len() < 20 {
            continue;
        }
        let channel = if m.channel_name.is_empty() {
            "unknown".to_string()
        } else {
            m.channel_name.clone()
        };
        let label = user_label(&m.user_id, users);
        let ts = if m.ts.is_empty() {
            "0".to_string()
        } else {
            m.ts.clone()
        };
        let title_prefix: String = text.chars().take(80).collect();

        let mut metadata = BTreeMap::new();
        metadata.insert("channel_name".into(), MetaValue::Str(channel.clone()));
        metadata.insert("message_count".into(), MetaValue::Int(1));
        metadata.insert(
            "reaction_count".into(),
            MetaValue::Int(m.reaction_count as i64),
        );
        if let Some(date) = ts_to_iso_date(ts_to_f64(&ts)) {
            metadata.insert("date".into(), MetaValue::Str(date));
        }

        out.push(Document {
            id: format!("slack:{channel}:{ts}"),
            source: "slack".into(),
            content: format!("{label}: {text}"),
            title: format!("#{channel}: {title_prefix}"),
            metadata,
        });
    }

    Ok(out)
}

fn ts_to_f64(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or(0.0)
}

fn ts_to_iso_date(ts: f64) -> Option<String> {
    let secs = ts.trunc() as i64;
    let dt = OffsetDateTime::from_unix_timestamp(secs).ok()?;
    let fmt = format_description!("[year]-[month]-[day]");
    dt.date().format(&fmt).ok()
}
