//! Scrape Google Drive files.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::StatusCode;
use serde::{Deserialize, Serialize};

use super::int_value;

const BASE: &str = "https://www.googleapis.com/drive/v3";
const DRIVE_SCOPE: &str = "https://www.googleapis.com/auth/drive.readonly";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const DEFAULT_PAGE_SIZE: usize = 100;
const DEFAULT_MAX_DOWNLOAD_BYTES: u64 = 1_000_000;

pub fn scrape(raw_dir: &Path, config: &BTreeMap<String, toml::Value>) -> Result<usize> {
    std::fs::create_dir_all(raw_dir).with_context(|| format!("creating {}", raw_dir.display()))?;

    let token = access_token().context(
        "getting Google Drive credentials; set GOOGLE_APPLICATION_CREDENTIALS for a service \
         account/ADC environment, or GOOGLE_ACCESS_TOKEN for an already-scoped token",
    )?;
    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("building Google Drive HTTP client")?;

    let page_size = int_value(config, "page_size")
        .unwrap_or(DEFAULT_PAGE_SIZE as i64)
        .clamp(1, 1000) as usize;
    let max_files = int_value(config, "max_files").unwrap_or(0).max(0) as usize;
    let max_download_bytes = int_value(config, "max_download_bytes")
        .unwrap_or(DEFAULT_MAX_DOWNLOAD_BYTES as i64)
        .max(0) as u64;

    let output_path = raw_dir.join("gdrive.jsonl");
    let already_done = load_existing_ids(&output_path)?;

    eprintln!("Listing Google Drive files...");
    let all_files = list_all_files(&client, &token, page_size, max_files)?;
    let files = all_files
        .into_iter()
        .filter(|file| !already_done.contains(&file.id))
        .collect::<Vec<_>>();

    if files.is_empty() {
        eprintln!(
            "Google Drive scrape complete: 0 new records ({} already present)",
            already_done.len()
        );
        return Ok(0);
    }

    let folder_cache = resolve_folder_paths(&client, &token, &files);
    let mut writer = append_jsonl(&output_path)?;
    let mut written = 0usize;
    let mut skipped = 0usize;

    for file in files {
        match fetch_file(&client, &token, &file, &folder_cache, max_download_bytes) {
            Ok(Some(record)) => {
                serde_json::to_writer(&mut writer, &record)
                    .with_context(|| format!("serializing record for {}", output_path.display()))?;
                writer.write_all(b"\n")?;
                written += 1;
                if written.is_multiple_of(200) {
                    eprintln!("Fetched {written} new Google Drive records...");
                }
            }
            Ok(None) => skipped += 1,
            Err(err) => {
                skipped += 1;
                eprintln!(
                    "warning: failed to fetch Google Drive file {} ({}): {err:#}",
                    file.name, file.id
                );
            }
        }
    }
    writer.flush()?;

    eprintln!(
        "Google Drive scrape complete: {written} new records, {} resumed, {skipped} skipped",
        already_done.len()
    );
    Ok(written)
}

fn access_token() -> Result<String> {
    if let Ok(token) = std::env::var("GOOGLE_ACCESS_TOKEN") {
        let token = token.trim();
        if !token.is_empty() {
            return Ok(token.to_string());
        }
    }

    let runtime = tokio::runtime::Runtime::new().context("creating Tokio runtime")?;
    runtime.block_on(async {
        use yup_oauth2::authenticator::ApplicationDefaultCredentialsTypes;

        let opts = yup_oauth2::ApplicationDefaultCredentialsFlowOpts::default();
        let auth = match yup_oauth2::ApplicationDefaultCredentialsAuthenticator::builder(opts).await
        {
            ApplicationDefaultCredentialsTypes::ServiceAccount(builder) => builder
                .build()
                .await
                .context("building service-account authenticator")?,
            ApplicationDefaultCredentialsTypes::InstanceMetadata(builder) => builder
                .build()
                .await
                .context("building instance-metadata authenticator")?,
        };
        let token = auth
            .token(&[DRIVE_SCOPE])
            .await
            .context("requesting Google Drive access token")?;
        token
            .token()
            .map(ToString::to_string)
            .ok_or_else(|| anyhow!("Google auth response did not include an access token"))
    })
}

fn list_all_files(
    client: &reqwest::blocking::Client,
    token: &str,
    page_size: usize,
    max_files: usize,
) -> Result<Vec<DriveFile>> {
    let mime_filter = export_mimes()
        .keys()
        .chain(download_mimes().keys())
        .map(|mime| format!("mimeType = '{mime}'"))
        .collect::<Vec<_>>()
        .join(" or ");
    let query = format!("({mime_filter}) and trashed = false");
    let fields = "nextPageToken, files(id, name, mimeType, size, parents, owners)";

    let mut files = Vec::new();
    let mut page_token = String::new();
    loop {
        let mut params = vec![
            ("q".to_string(), query.clone()),
            ("pageSize".to_string(), page_size.to_string()),
            ("fields".to_string(), fields.to_string()),
            ("includeItemsFromAllDrives".to_string(), "true".to_string()),
            ("supportsAllDrives".to_string(), "true".to_string()),
            ("corpora".to_string(), "allDrives".to_string()),
        ];
        if !page_token.is_empty() {
            params.push(("pageToken".to_string(), page_token.clone()));
        }

        let resp: ListResponse = get_json(client, token, &format!("{BASE}/files"), &params)?;
        files.extend(resp.files);
        if max_files > 0 && files.len() >= max_files {
            files.truncate(max_files);
            break;
        }
        if files.len().is_multiple_of(500) && !files.is_empty() {
            eprintln!("Listed {} Google Drive files...", files.len());
        }
        page_token = resp.next_page_token;
        if page_token.is_empty() {
            break;
        }
    }
    Ok(files)
}

fn resolve_folder_paths(
    client: &reqwest::blocking::Client,
    token: &str,
    files: &[DriveFile],
) -> BTreeMap<String, String> {
    let mut cache = BTreeMap::new();
    let parent_ids = files
        .iter()
        .filter_map(|file| file.parents.first().cloned())
        .collect::<BTreeSet<_>>();

    eprintln!(
        "Resolving {} Google Drive folder paths...",
        parent_ids.len()
    );
    for folder_id in parent_ids {
        let path = resolve_folder_path(client, token, &folder_id, &mut cache);
        cache.insert(folder_id, path);
    }
    cache
}

fn resolve_folder_path(
    client: &reqwest::blocking::Client,
    token: &str,
    folder_id: &str,
    cache: &mut BTreeMap<String, String>,
) -> String {
    if let Some(path) = cache.get(folder_id) {
        return path.clone();
    }

    let mut parts = Vec::new();
    let mut current = folder_id.to_string();
    for _ in 0..5 {
        if let Some(path) = cache.get(&current) {
            if !path.is_empty() {
                parts.push(path.clone());
            }
            break;
        }

        let params = vec![
            ("fields".to_string(), "name, parents".to_string()),
            ("supportsAllDrives".to_string(), "true".to_string()),
        ];
        let folder: Folder =
            match get_json(client, token, &format!("{BASE}/files/{current}"), &params) {
                Ok(folder) => folder,
                Err(_) => break,
            };
        if !folder.name.is_empty() {
            parts.push(folder.name);
        }
        let Some(parent) = folder.parents.first() else {
            break;
        };
        current = parent.clone();
    }

    parts.reverse();
    parts.join(" > ")
}

fn fetch_file(
    client: &reqwest::blocking::Client,
    token: &str,
    file: &DriveFile,
    folder_cache: &BTreeMap<String, String>,
    max_download_bytes: u64,
) -> Result<Option<Record>> {
    let (content, file_type) = if let Some((_, export_mime)) = export_mimes()
        .iter()
        .find(|(mime, _)| **mime == file.mime_type.as_str())
    {
        let bytes = get_bytes(
            client,
            token,
            &format!("{BASE}/files/{}/export", file.id),
            &[("mimeType".to_string(), (*export_mime).to_string())],
        )?;
        (
            String::from_utf8_lossy(&bytes).into_owned(),
            file_type_label(&file.mime_type).to_string(),
        )
    } else if let Some((_, file_type)) = download_mimes()
        .iter()
        .find(|(mime, _)| **mime == file.mime_type.as_str())
    {
        if file.size.unwrap_or(0) > max_download_bytes {
            return Ok(None);
        }
        let bytes = get_bytes(
            client,
            token,
            &format!("{BASE}/files/{}", file.id),
            &[("alt".to_string(), "media".to_string())],
        )?;
        (
            String::from_utf8_lossy(&bytes).into_owned(),
            (*file_type).to_string(),
        )
    } else {
        return Ok(None);
    };

    if content.trim().len() < 50 {
        return Ok(None);
    }

    let path = file
        .parents
        .first()
        .and_then(|parent| folder_cache.get(parent))
        .cloned()
        .unwrap_or_default();
    let owner = file
        .owners
        .first()
        .map(|owner| owner.display_name.clone())
        .unwrap_or_default();

    Ok(Some(Record {
        file_id: file.id.clone(),
        name: file.name.clone(),
        owner,
        content,
        mime_type: file.mime_type.clone(),
        file_type,
        path,
    }))
}

fn get_json<T: for<'de> Deserialize<'de>>(
    client: &reqwest::blocking::Client,
    token: &str,
    url: &str,
    params: &[(String, String)],
) -> Result<T> {
    for attempt in 1..=3 {
        let resp = client
            .get(url)
            .bearer_auth(token)
            .query(params)
            .header("Accept", "application/json")
            .send();
        match resp {
            Ok(resp) if resp.status().is_success() => return resp.json().context("parsing JSON"),
            Ok(resp) if should_retry(resp.status()) && attempt < 3 => {
                sleep_retry(resp.headers(), attempt);
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                anyhow::bail!("Google Drive request failed: HTTP {status} - {body}");
            }
            Err(err) if attempt < 3 => {
                eprintln!("warning: Google Drive request failed, retrying: {err}");
                std::thread::sleep(Duration::from_secs(attempt as u64 * 2));
            }
            Err(err) => return Err(err).context("sending Google Drive request"),
        }
    }
    unreachable!()
}

fn get_bytes(
    client: &reqwest::blocking::Client,
    token: &str,
    url: &str,
    params: &[(String, String)],
) -> Result<Vec<u8>> {
    for attempt in 1..=3 {
        let resp = client.get(url).bearer_auth(token).query(params).send();
        match resp {
            Ok(resp) if resp.status().is_success() => {
                return Ok(resp.bytes().context("reading response bytes")?.to_vec());
            }
            Ok(resp) if should_retry(resp.status()) && attempt < 3 => {
                sleep_retry(resp.headers(), attempt);
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                anyhow::bail!("Google Drive download failed: HTTP {status} - {body}");
            }
            Err(err) if attempt < 3 => {
                eprintln!("warning: Google Drive download failed, retrying: {err}");
                std::thread::sleep(Duration::from_secs(attempt as u64 * 2));
            }
            Err(err) => return Err(err).context("downloading Google Drive file"),
        }
    }
    unreachable!()
}

fn should_retry(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

fn sleep_retry(headers: &reqwest::header::HeaderMap, attempt: usize) {
    let delay = headers
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(attempt as u64 * 2);
    std::thread::sleep(Duration::from_secs(delay));
}

fn load_existing_ids(path: &Path) -> Result<BTreeSet<String>> {
    let mut out = BTreeSet::new();
    if !path.exists() {
        return Ok(out);
    }

    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(&line)
            .with_context(|| format!("parsing existing {}", path.display()))?;
        if let Some(file_id) = value.get("file_id").and_then(serde_json::Value::as_str) {
            out.insert(file_id.to_string());
        }
    }
    Ok(out)
}

fn append_jsonl(path: &Path) -> Result<BufWriter<std::fs::File>> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    Ok(BufWriter::new(file))
}

fn export_mimes() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        ("application/vnd.google-apps.document", "text/plain"),
        ("application/vnd.google-apps.spreadsheet", "text/csv"),
        ("application/vnd.google-apps.presentation", "text/plain"),
    ])
}

fn download_mimes() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        ("text/plain", "text"),
        ("text/csv", "csv"),
        ("text/markdown", "markdown"),
        ("text/html", "html"),
        ("application/json", "json"),
    ])
}

fn file_type_label(mime_type: &str) -> &'static str {
    match mime_type {
        "application/vnd.google-apps.document" => "Google Doc",
        "application/vnd.google-apps.spreadsheet" => "Google Sheet",
        "application/vnd.google-apps.presentation" => "Google Slides",
        _ => "document",
    }
}

#[derive(Debug, Deserialize)]
struct ListResponse {
    #[serde(rename = "nextPageToken", default)]
    next_page_token: String,
    #[serde(default)]
    files: Vec<DriveFile>,
}

#[derive(Debug, Clone, Deserialize)]
struct DriveFile {
    id: String,
    #[serde(default)]
    name: String,
    #[serde(rename = "mimeType", default)]
    mime_type: String,
    #[serde(default, deserialize_with = "deserialize_size")]
    size: Option<u64>,
    #[serde(default)]
    parents: Vec<String>,
    #[serde(default)]
    owners: Vec<Owner>,
}

#[derive(Debug, Clone, Deserialize)]
struct Owner {
    #[serde(rename = "displayName", default)]
    display_name: String,
}

#[derive(Debug, Deserialize)]
struct Folder {
    #[serde(default)]
    name: String,
    #[serde(default)]
    parents: Vec<String>,
}

#[derive(Debug, Serialize)]
struct Record {
    file_id: String,
    name: String,
    owner: String,
    content: String,
    mime_type: String,
    file_type: String,
    path: String,
}

fn deserialize_size<'de, D>(deserializer: D) -> std::result::Result<Option<u64>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    Ok(match value {
        Some(serde_json::Value::Number(number)) => number.as_u64(),
        Some(serde_json::Value::String(text)) => text.parse().ok(),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_google_workspace_types() {
        assert_eq!(
            file_type_label("application/vnd.google-apps.presentation"),
            "Google Slides"
        );
        assert_eq!(file_type_label("application/octet-stream"), "document");
    }

    #[test]
    fn parses_string_file_sizes() {
        let file: DriveFile = serde_json::from_value(serde_json::json!({
            "id": "abc",
            "size": "123"
        }))
        .unwrap();
        assert_eq!(file.size, Some(123));
    }

    #[test]
    fn loads_existing_file_ids() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gdrive.jsonl");
        std::fs::write(
            &path,
            r#"{"file_id":"one","name":"A"}"#.to_string()
                + "\n"
                + r#"{"file_id":"two","name":"B"}"#
                + "\n",
        )
        .unwrap();

        let ids = load_existing_ids(&path).unwrap();
        assert!(ids.contains("one"));
        assert!(ids.contains("two"));
    }
}
