//! Slack workspace scraper.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use regex::Regex;
use serde_json::{json, Value};
use time::macros::format_description;
use time::{Date, PrimitiveDateTime, Time};

use super::{bool_value, int_value, string_list, string_value, write_jsonl};

const BASE: &str = "https://slack.com/api";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const SKIP_FILETYPES: &[&str] = &[
    "gdoc", "gsheet", "gpres", "gform", "gslides", "gdraw", "mp4", "mov", "webm", "avi", "mkv",
    "flv",
];

pub fn scrape(raw_dir: &Path, config: &BTreeMap<String, toml::Value>) -> Result<usize> {
    let token = std::env::var("SLACK_TOKEN")
        .map_err(|_| anyhow!("SLACK_TOKEN environment variable is required"))?;
    std::fs::create_dir_all(raw_dir).with_context(|| format!("creating {}", raw_dir.display()))?;
    let files_dir = raw_dir.join("files");
    std::fs::create_dir_all(&files_dir)
        .with_context(|| format!("creating {}", files_dir.display()))?;

    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("building Slack HTTP client")?;

    let include_private = bool_value(config, "include_private").unwrap_or(false);
    let include_bots = bool_value(config, "include_bots").unwrap_or(false);
    let fetch_pins = bool_value(config, "fetch_pins").unwrap_or(true);
    let fetch_bookmarks = bool_value(config, "fetch_bookmarks").unwrap_or(true);
    let fetch_files = bool_value(config, "fetch_files").unwrap_or(true);
    let max_file_size = int_value(config, "max_file_size")
        .unwrap_or(50_000_000)
        .max(0) as u64;
    let max_messages = int_value(config, "max_messages").unwrap_or(0).max(0) as usize;
    let page_size = int_value(config, "page_size").unwrap_or(200).clamp(1, 999) as usize;
    let allowlist: BTreeSet<String> = string_list(config, "channel_allowlist")
        .into_iter()
        .collect();
    let denylist: BTreeSet<String> = string_list(config, "channel_denylist")
        .into_iter()
        .collect();
    let oldest = string_value(config, "date_cutoff")
        .as_deref()
        .and_then(parse_date_cutoff)
        .unwrap_or(0.0);

    let users = fetch_users(&client, &token)?;
    write_json(raw_dir.join("users.json").as_path(), &users)?;

    let channels = fetch_channels(&client, &token, include_private, &allowlist, &denylist)?;
    write_json(raw_dir.join("channels.json").as_path(), &channels)?;

    let mut messages = Vec::new();
    let mut pins = Vec::new();
    let mut bookmarks = Vec::new();
    let mut file_records = Vec::new();

    for channel in &channels {
        let channel_id = value_str(channel, "id");
        let channel_name = value_str(channel, "name");
        let raw_messages = fetch_all_pages(
            &client,
            &token,
            "conversations.history",
            vec![
                ("channel".to_string(), channel_id.clone()),
                ("limit".to_string(), page_size.to_string()),
                ("oldest".to_string(), oldest.to_string()),
            ],
            "messages",
        )?;

        if fetch_pins {
            pins.extend(fetch_pins_for_channel(
                &client,
                &token,
                &channel_id,
                &channel_name,
            )?);
        }
        if fetch_bookmarks {
            bookmarks.extend(fetch_bookmarks_for_channel(
                &client,
                &token,
                &channel_id,
                &channel_name,
            )?);
        }

        for msg in raw_messages {
            if skip_message(&msg, include_bots) {
                continue;
            }
            messages.push(message_to_record(&msg, &channel_id, &channel_name, false));
            if fetch_files {
                file_records.extend(download_msg_files(
                    &client,
                    &token,
                    &msg,
                    &files_dir,
                    max_file_size,
                    &channel_id,
                    &channel_name,
                )?);
            }

            if value_i64(&msg, "reply_count") > 0 {
                let ts = value_str(&msg, "ts");
                let replies = fetch_all_pages(
                    &client,
                    &token,
                    "conversations.replies",
                    vec![
                        ("channel".to_string(), channel_id.clone()),
                        ("ts".to_string(), ts),
                        ("limit".to_string(), "200".to_string()),
                    ],
                    "messages",
                )?;
                for reply in replies.into_iter().skip(1) {
                    if skip_message(&reply, include_bots) {
                        continue;
                    }
                    messages.push(message_to_record(&reply, &channel_id, &channel_name, true));
                    if fetch_files {
                        file_records.extend(download_msg_files(
                            &client,
                            &token,
                            &reply,
                            &files_dir,
                            max_file_size,
                            &channel_id,
                            &channel_name,
                        )?);
                    }
                }
            }

            if max_messages > 0 && messages.len() >= max_messages {
                break;
            }
        }

        eprintln!(
            "Scraped #{}: total messages={}, pins={}, bookmarks={}, files={}",
            channel_name,
            messages.len(),
            pins.len(),
            bookmarks.len(),
            file_records.len()
        );
        if max_messages > 0 && messages.len() >= max_messages {
            break;
        }
    }

    write_jsonl(&raw_dir.join("messages.jsonl"), &messages)?;
    write_jsonl(&raw_dir.join("pins.jsonl"), &pins)?;
    write_jsonl(&raw_dir.join("bookmarks.jsonl"), &bookmarks)?;
    write_jsonl(&raw_dir.join("files.jsonl"), &file_records)?;

    eprintln!(
        "Slack scrape complete: {} messages, {} channels, {} pins, {} bookmarks, {} files",
        messages.len(),
        channels.len(),
        pins.len(),
        bookmarks.len(),
        file_records.len()
    );
    Ok(messages.len())
}

fn fetch_users(client: &reqwest::blocking::Client, token: &str) -> Result<BTreeMap<String, Value>> {
    let members = fetch_all_pages(
        client,
        token,
        "users.list",
        vec![("limit".into(), "200".into())],
        "members",
    )?;
    let mut out = BTreeMap::new();
    for member in members {
        let id = value_str(&member, "id");
        if id.is_empty() {
            continue;
        }
        let profile = member.get("profile").unwrap_or(&Value::Null);
        let name = value_str(profile, "display_name")
            .if_empty_then(|| value_str(profile, "real_name"))
            .if_empty_then(|| value_str(&member, "name"))
            .if_empty_then(|| id.clone());
        out.insert(
            id,
            json!({
                "name": name,
                "title": value_str(profile, "title"),
                "department": profile
                    .get("fields")
                    .and_then(|fields| fields.get("department"))
                    .and_then(|department| department.get("value"))
                    .and_then(Value::as_str)
                    .unwrap_or(""),
                "is_bot": member.get("is_bot").and_then(Value::as_bool).unwrap_or(false),
            }),
        );
    }
    Ok(out)
}

fn fetch_channels(
    client: &reqwest::blocking::Client,
    token: &str,
    include_private: bool,
    allowlist: &BTreeSet<String>,
    denylist: &BTreeSet<String>,
) -> Result<Vec<Value>> {
    let types = if include_private {
        "public_channel,private_channel"
    } else {
        "public_channel"
    };
    let result = fetch_all_pages(
        client,
        token,
        "conversations.list",
        vec![
            ("types".into(), types.into()),
            ("exclude_archived".into(), "true".into()),
            ("limit".into(), "200".into()),
        ],
        "channels",
    );
    let raw = match result {
        Ok(channels) => channels,
        Err(err) if include_private && format!("{err:#}").contains("missing_scope") => {
            return fetch_channels(client, token, false, allowlist, denylist);
        }
        Err(err) => return Err(err),
    };

    Ok(raw
        .into_iter()
        .filter(|channel| {
            let name = value_str(channel, "name");
            (allowlist.is_empty() || allowlist.contains(&name)) && !denylist.contains(&name)
        })
        .map(|channel| {
            json!({
                "id": value_str(&channel, "id"),
                "name": value_str(&channel, "name"),
                "topic": channel.get("topic").map(|v| value_str(v, "value")).unwrap_or_default(),
                "purpose": channel.get("purpose").map(|v| value_str(v, "value")).unwrap_or_default(),
                "is_private": channel.get("is_private").and_then(Value::as_bool).unwrap_or(false),
                "num_members": value_i64(&channel, "num_members"),
            })
        })
        .collect())
}

fn fetch_pins_for_channel(
    client: &reqwest::blocking::Client,
    token: &str,
    channel_id: &str,
    channel_name: &str,
) -> Result<Vec<Value>> {
    let resp = match slack_get(
        client,
        token,
        "pins.list",
        &[("channel".to_string(), channel_id.to_string())],
    ) {
        Ok(resp) => resp,
        Err(err) if format!("{err:#}").contains("missing_scope") => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    Ok(resp
        .get("items")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|pin| pin.get("message").map(|message| (pin, message)))
        .map(|(pin, message)| {
            json!({
                "channel_id": channel_id,
                "channel_name": channel_name,
                "user_id": value_str(message, "user"),
                "ts": value_str(message, "ts"),
                "text": value_str(message, "text"),
                "pinned_by": value_str(pin, "created_by"),
            })
        })
        .collect())
}

fn fetch_bookmarks_for_channel(
    client: &reqwest::blocking::Client,
    token: &str,
    channel_id: &str,
    channel_name: &str,
) -> Result<Vec<Value>> {
    let resp = match slack_get(
        client,
        token,
        "bookmarks.list",
        &[("channel_id".to_string(), channel_id.to_string())],
    ) {
        Ok(resp) => resp,
        Err(err) if format!("{err:#}").contains("missing_scope") => return Ok(Vec::new()),
        Err(err) => return Err(err),
    };
    Ok(resp
        .get("bookmarks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|bookmark| {
            json!({
                "channel_id": channel_id,
                "channel_name": channel_name,
                "title": value_str(bookmark, "title"),
                "link": value_str(bookmark, "link"),
                "emoji": value_str(bookmark, "emoji"),
                "created_by": value_str(bookmark, "created_by"),
            })
        })
        .collect())
}

fn fetch_all_pages(
    client: &reqwest::blocking::Client,
    token: &str,
    method: &str,
    base_params: Vec<(String, String)>,
    result_key: &str,
) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let mut cursor = String::new();
    loop {
        let mut params = base_params.clone();
        if !cursor.is_empty() {
            params.push(("cursor".to_string(), cursor.clone()));
        }
        let resp = slack_get(client, token, method, &params)?;
        if let Some(items) = resp.get(result_key).and_then(Value::as_array) {
            out.extend(items.iter().cloned());
        }
        cursor = resp
            .get("response_metadata")
            .and_then(|m| m.get("next_cursor"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if cursor.is_empty() {
            break;
        }
    }
    Ok(out)
}

fn slack_get(
    client: &reqwest::blocking::Client,
    token: &str,
    method: &str,
    params: &[(String, String)],
) -> Result<Value> {
    let url = format!("{BASE}/{method}");
    for attempt in 1..=5 {
        let resp = client.get(&url).bearer_auth(token).query(params).send();
        match resp {
            Ok(resp) if resp.status().as_u16() == 429 => {
                let wait = resp
                    .headers()
                    .get("Retry-After")
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.parse::<u64>().ok())
                    .unwrap_or(attempt * 2);
                std::thread::sleep(Duration::from_secs(wait));
            }
            Ok(resp) => {
                let status = resp.status();
                let payload: Value = resp
                    .json()
                    .with_context(|| format!("parsing Slack {method} response"))?;
                if !status.is_success() {
                    bail!("Slack API {method} failed: HTTP {status} — {payload}");
                }
                if payload.get("ok").and_then(Value::as_bool) == Some(false) {
                    bail!(
                        "Slack API {method} failed: {}",
                        payload
                            .get("error")
                            .and_then(Value::as_str)
                            .unwrap_or("unknown_error")
                    );
                }
                return Ok(payload);
            }
            Err(err) if attempt < 5 => {
                std::thread::sleep(Duration::from_secs(attempt * 2));
                eprintln!("warning: Slack {method} failed transiently: {err}");
            }
            Err(err) => return Err(err).with_context(|| format!("calling Slack {method}")),
        }
    }
    bail!("Slack API {method} exhausted retries")
}

fn skip_message(msg: &Value, include_bots: bool) -> bool {
    let subtype = value_str(msg, "subtype");
    if matches!(
        subtype.as_str(),
        "channel_join" | "channel_leave" | "channel_topic" | "channel_purpose" | "channel_name"
    ) {
        return true;
    }
    !include_bots
        && (msg.get("bot_id").is_some()
            || msg.get("subtype").and_then(Value::as_str) == Some("bot_message"))
}

fn message_to_record(msg: &Value, channel_id: &str, channel_name: &str, is_reply: bool) -> Value {
    let text = value_str(msg, "text");
    json!({
        "channel_id": channel_id,
        "channel_name": channel_name,
        "user_id": value_str(msg, "user"),
        "ts": value_str(msg, "ts"),
        "thread_ts": value_str(msg, "thread_ts"),
        "text": text,
        "is_reply": is_reply,
        "reply_count": value_i64(msg, "reply_count"),
        "subtype": value_str(msg, "subtype"),
        "reactions": extract_reactions(msg),
        "reaction_count": msg
            .get("reactions")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|reaction| reaction.get("count").and_then(Value::as_i64))
            .sum::<i64>(),
        "files": extract_files(msg),
        "links": extract_links(&text),
    })
}

fn extract_reactions(msg: &Value) -> Vec<Value> {
    msg.get("reactions")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|reaction| {
            json!({
                "name": value_str(reaction, "name"),
                "count": value_i64(reaction, "count"),
            })
        })
        .collect()
}

fn extract_files(msg: &Value) -> Vec<Value> {
    msg.get("files")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|file| {
            json!({
                "id": value_str(file, "id"),
                "name": value_str(file, "name"),
                "filetype": value_str(file, "filetype"),
                "mimetype": value_str(file, "mimetype"),
                "size": value_i64(file, "size"),
                "url_private_download": value_str(file, "url_private_download"),
            })
        })
        .collect()
}

fn download_msg_files(
    client: &reqwest::blocking::Client,
    token: &str,
    msg: &Value,
    files_dir: &Path,
    max_file_size: u64,
    channel_id: &str,
    channel_name: &str,
) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    for file in msg
        .get("files")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let file_type = value_str(file, "filetype");
        if SKIP_FILETYPES.contains(&file_type.as_str()) {
            continue;
        }
        let size = value_i64(file, "size").max(0) as u64;
        if size > max_file_size {
            continue;
        }
        let file_id = value_str(file, "id");
        if file_id.is_empty() {
            continue;
        }
        let url = value_str(file, "url_private_download")
            .if_empty_then(|| value_str(file, "url_private"));
        if url.is_empty() {
            continue;
        }

        let local_name = format!(
            "{file_id}.{}",
            if file_type.is_empty() {
                "bin"
            } else {
                &file_type
            }
        );
        let local_path = files_dir.join(&local_name);
        if !local_path.exists() {
            let resp = client.get(&url).bearer_auth(token).send()?;
            if !resp.status().is_success() {
                continue;
            }
            let data = resp.bytes()?;
            std::fs::write(&local_path, &data)
                .with_context(|| format!("writing {}", local_path.display()))?;
        }

        out.push(json!({
            "channel_id": channel_id,
            "channel_name": channel_name,
            "user_id": value_str(msg, "user"),
            "ts": value_str(msg, "ts"),
            "file_id": file_id,
            "file_name": value_str(file, "name"),
            "file_type": file_type,
            "mime_type": value_str(file, "mimetype"),
            "size": size,
            "local_path": format!("files/{local_name}"),
        }));
    }
    Ok(out)
}

fn extract_links(text: &str) -> Vec<String> {
    static R: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let re = R.get_or_init(|| Regex::new(r"<(https?://[^|>]+)(?:\|[^>]*)?>").unwrap());
    re.captures_iter(text)
        .map(|caps| caps[1].to_string())
        .collect()
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let raw = serde_json::to_string_pretty(value)?;
    std::fs::write(path, raw).with_context(|| format!("writing {}", path.display()))
}

fn parse_date_cutoff(value: &str) -> Option<f64> {
    let fmt = format_description!("[year]-[month]-[day]");
    let date = Date::parse(value, &fmt).ok()?;
    let dt = PrimitiveDateTime::new(date, Time::MIDNIGHT).assume_utc();
    Some(dt.unix_timestamp() as f64)
}

fn value_str(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn value_i64(value: &Value, key: &str) -> i64 {
    value.get(key).and_then(Value::as_i64).unwrap_or(0)
}

trait EmptyStringExt {
    fn if_empty_then(self, f: impl FnOnce() -> String) -> String;
}

impl EmptyStringExt for String {
    fn if_empty_then(self, f: impl FnOnce() -> String) -> String {
        if self.is_empty() {
            f()
        } else {
            self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_slack_links() {
        assert_eq!(
            extract_links("See <https://example.com|docs> and <https://x.test>"),
            vec![
                "https://example.com".to_string(),
                "https://x.test".to_string()
            ]
        );
    }

    #[test]
    fn message_record_matches_normalizer_fields() {
        let msg = json!({
            "user": "U1",
            "ts": "1.2",
            "thread_ts": "1.0",
            "text": "hello <https://example.com|docs>",
            "reply_count": 2,
            "reactions": [{"name": "thumbsup", "count": 3}]
        });
        let record = message_to_record(&msg, "C1", "general", false);
        assert_eq!(record["channel_name"], "general");
        assert_eq!(record["reaction_count"], 3);
        assert_eq!(record["links"][0], "https://example.com");
    }
}
