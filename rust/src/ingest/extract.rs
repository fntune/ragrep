//! Extract text content from downloaded files.
//!
//! Rust v1 keeps local reads for text-like files and uses Gemini multimodal
//! HTTP for PDFs, Office documents, and images.

use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use serde_json::{json, Value};

use crate::ingest::normalize::read_jsonl;
use crate::ingest::scrape::write_jsonl;

const DEFAULT_MODEL: &str = "gemini-2.0-flash";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const PROMPT: &str = "Extract the useful information from this file. If it contains text, transcribe all meaningful text. If it is an image, describe it in detail and transcribe visible text. If it is a spreadsheet, preserve rows in a readable tab-delimited form. Focus on information content.";

const IMAGE_TYPES: &[&str] = &[
    "png", "jpg", "jpeg", "gif", "webp", "heic", "bmp", "tiff", "svg",
];
const DOC_TYPES: &[&str] = &["pdf", "docx", "doc", "xlsx", "xls", "pptx", "ppt"];
const TEXT_TYPES: &[&str] = &[
    "text",
    "markdown",
    "python",
    "javascript",
    "typescript",
    "sql",
    "json",
    "yaml",
    "xml",
    "csv",
    "shell",
    "go",
    "rust",
    "java",
    "c",
    "cpp",
    "html",
    "css",
    "plain",
    "email",
];

#[derive(Debug, Default)]
struct Stats {
    vision: usize,
    text: usize,
    skipped: usize,
    failed: usize,
}

pub fn extract_all(raw_dir: &Path) -> Result<usize> {
    let files_jsonl = raw_dir.join("files.jsonl");
    let output_path = raw_dir.join("files_extracted.jsonl");
    let records = read_jsonl(&files_jsonl)?;
    if records.is_empty() {
        eprintln!("No file records found at {}", files_jsonl.display());
        write_jsonl::<Value>(&output_path, &[])?;
        return Ok(0);
    }

    let model = std::env::var("GEMINI_VISION_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let api_key = std::env::var("GEMINI_API_KEY").ok();
    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("building Gemini HTTP client")?;

    let mut stats = Stats::default();
    let mut out = Vec::new();

    for mut record in records {
        let file_type = value_str(&record, "file_type").to_lowercase();
        let local_path = raw_dir.join(value_str(&record, "local_path"));
        if !local_path.exists() {
            stats.skipped += 1;
            continue;
        }

        let extracted = if TEXT_TYPES.contains(&file_type.as_str()) {
            stats.text += 1;
            extract_text(&local_path).map(|content| (content, "text".to_string()))
        } else if IMAGE_TYPES.contains(&file_type.as_str())
            || DOC_TYPES.contains(&file_type.as_str())
        {
            let Some(key) = api_key.as_deref() else {
                stats.failed += 1;
                eprintln!(
                    "warning: GEMINI_API_KEY is not set; cannot extract {}",
                    local_path.display()
                );
                continue;
            };
            stats.vision += 1;
            describe_file(&client, key, &model, &local_path, &file_type)
                .map(|content| (content, "vision".to_string()))
        } else {
            stats.skipped += 1;
            continue;
        };

        match extracted {
            Ok((Some(content), method)) => {
                if let Some(obj) = record.as_object_mut() {
                    obj.insert("content".to_string(), Value::String(content));
                    obj.insert("extraction_method".to_string(), Value::String(method));
                    out.push(record);
                }
            }
            Ok((None, _)) => stats.failed += 1,
            Err(err) => {
                stats.failed += 1;
                eprintln!(
                    "warning: extraction failed for {}: {err:#}",
                    local_path.display()
                );
            }
        }
    }

    write_jsonl(&output_path, &out)?;
    eprintln!(
        "Extraction complete: {} records written. vision={}, text={}, skipped={}, failed={}",
        out.len(),
        stats.vision,
        stats.text,
        stats.skipped,
        stats.failed
    );
    Ok(out.len())
}

fn extract_text(path: &Path) -> Result<Option<String>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("reading text file {}", path.display()))?
        .trim()
        .to_string();
    Ok((content.len() >= 50).then_some(content))
}

fn describe_file(
    client: &reqwest::blocking::Client,
    api_key: &str,
    model: &str,
    path: &Path,
    file_type: &str,
) -> Result<Option<String>> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if data.is_empty() {
        return Ok(None);
    }
    let encoded = base64::engine::general_purpose::STANDARD.encode(data);
    let mime = mime_for(file_type);
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    );
    let resp = client
        .post(url)
        .json(&json!({
            "contents": [{
                "parts": [
                    { "text": PROMPT },
                    { "inline_data": { "mime_type": mime, "data": encoded } }
                ]
            }]
        }))
        .send()
        .context("sending Gemini extraction request")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        bail!("Gemini extraction failed: HTTP {status} — {body}");
    }

    let payload: Value = resp.json().context("parsing Gemini extraction response")?;
    let text = payload
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first())
        .and_then(|candidate| candidate.get("content"))
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|part| part.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string();

    Ok((text.len() >= 20).then_some(text))
}

fn mime_for(file_type: &str) -> &'static str {
    match file_type {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "heic" => "image/heic",
        "bmp" => "image/bmp",
        "tiff" => "image/tiff",
        "svg" => "image/svg+xml",
        "pdf" => "application/pdf",
        "docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc" => "application/msword",
        "xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls" => "application/vnd.ms-excel",
        "pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "ppt" => "application/vnd.ms-powerpoint",
        _ => "application/octet-stream",
    }
}

fn value_str(record: &Value, key: &str) -> String {
    record
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

#[allow(dead_code)]
fn _path_for(raw_dir: &Path, relative: &str) -> Result<PathBuf> {
    if relative.is_empty() {
        return Err(anyhow!("missing local_path"));
    }
    Ok(raw_dir.join(relative))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_mime_types() {
        assert_eq!(mime_for("pdf"), "application/pdf");
        assert_eq!(mime_for("png"), "image/png");
    }
}
