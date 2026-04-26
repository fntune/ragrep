use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

pub mod atlassian;
pub mod bitbucket;
pub mod code;
pub mod gdrive;
pub mod git;
pub mod slack;

pub fn string_list(config: &BTreeMap<String, toml::Value>, key: &str) -> Vec<String> {
    match config.get(key) {
        Some(toml::Value::Array(values)) => values
            .iter()
            .filter_map(|value| value.as_str().map(ToString::to_string))
            .collect(),
        Some(toml::Value::String(value)) => vec![value.clone()],
        _ => Vec::new(),
    }
}

pub fn string_value(config: &BTreeMap<String, toml::Value>, key: &str) -> Option<String> {
    config
        .get(key)
        .and_then(toml::Value::as_str)
        .map(ToString::to_string)
}

pub fn int_value(config: &BTreeMap<String, toml::Value>, key: &str) -> Option<i64> {
    config.get(key).and_then(toml::Value::as_integer)
}

pub fn expand_repos(patterns: &[String]) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut seen = std::collections::BTreeSet::new();

    for pattern in patterns {
        let expanded = expand_home(pattern);
        let matches = glob::glob(&expanded)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .collect::<Vec<_>>();

        if matches.is_empty() {
            push_repo(PathBuf::from(expanded), &mut seen, &mut out);
            continue;
        }

        for path in matches {
            push_repo(path, &mut seen, &mut out);
        }
    }

    out
}

pub fn repo_name(repo: &git2::Repository, fallback_path: &Path) -> String {
    if let Ok(remote) = repo.find_remote("origin") {
        if let Some(url) = remote.url() {
            let tail = url
                .trim_end_matches('/')
                .rsplit(['/', ':'])
                .next()
                .unwrap_or(url)
                .trim_end_matches(".git");
            if !tail.is_empty() {
                return tail.to_string();
            }
        }
    }
    fallback_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("repo")
        .to_string()
}

pub fn write_jsonl<T: Serialize>(path: &Path, records: &[T]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for record in records {
        serde_json::to_writer(&mut writer, record)
            .with_context(|| format!("serializing record for {}", path.display()))?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    Ok(())
}

fn expand_home(path: &str) -> String {
    if path == "~" {
        return std::env::var("HOME").unwrap_or_else(|_| path.to_string());
    }
    if let Some(rest) = path.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/{rest}");
        }
    }
    path.to_string()
}

fn push_repo(
    path: PathBuf,
    seen: &mut std::collections::BTreeSet<PathBuf>,
    out: &mut Vec<PathBuf>,
) {
    let Ok(resolved) = path.canonicalize() else {
        return;
    };
    if !resolved.is_dir() || !resolved.join(".git").is_dir() {
        return;
    }
    if seen.insert(resolved.clone()) {
        out.push(resolved);
    }
}
