//! Scrape Confluence pages and Jira issues.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use anyhow::{Context, Result};
use regex::Regex;
use serde_json::{json, Value};

use super::{int_value, string_list, string_value, write_jsonl};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

pub fn scrape(raw_dir: &Path, config: &BTreeMap<String, toml::Value>) -> Result<usize> {
    let confluence_url = std::env::var("CONFLUENCE_URL")
        .ok()
        .or_else(|| string_value(config, "confluence_url"))
        .unwrap_or_default();
    let jira_url = std::env::var("JIRA_URL")
        .ok()
        .or_else(|| string_value(config, "jira_url"))
        .unwrap_or_default();
    let username = std::env::var("ATLASSIAN_USERNAME")
        .ok()
        .or_else(|| string_value(config, "username"))
        .unwrap_or_default();
    let api_token = std::env::var("ATLASSIAN_API_TOKEN")
        .ok()
        .or_else(|| string_value(config, "api_token"))
        .unwrap_or_default();
    let spaces = string_list(config, "confluence_spaces");
    let projects = string_list(config, "jira_projects");
    let max_pages = int_value(config, "max_pages").unwrap_or(0).max(0) as usize;

    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("building Atlassian HTTP client")?;

    let mut records = Vec::new();
    records.extend(scrape_confluence(
        &client,
        &confluence_url,
        &username,
        &api_token,
        spaces,
        max_pages,
    )?);
    records.extend(scrape_jira(
        &client, &jira_url, &username, &api_token, projects, max_pages,
    )?);

    write_jsonl(&raw_dir.join("atlassian.jsonl"), &records)?;
    eprintln!(
        "Atlassian scrape complete: {} records written",
        records.len()
    );
    Ok(records.len())
}

fn scrape_confluence(
    client: &reqwest::blocking::Client,
    base_url: &str,
    username: &str,
    api_token: &str,
    mut spaces: Vec<String>,
    max_pages: usize,
) -> Result<Vec<Value>> {
    if base_url.is_empty() || username.is_empty() || api_token.is_empty() {
        eprintln!("Confluence not configured, skipping");
        return Ok(Vec::new());
    }

    if spaces.is_empty() {
        let resp = confluence_get(
            client,
            base_url,
            username,
            api_token,
            "/space",
            &[("limit", "100")],
        )?;
        spaces = resp
            .get("results")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|space| space.get("key").and_then(Value::as_str))
            .filter(|key| !key.starts_with('~'))
            .map(ToString::to_string)
            .collect();
    }

    let mut records = Vec::new();
    for space in spaces {
        let mut start = 0usize;
        loop {
            let start_s = start.to_string();
            let resp = confluence_get(
                client,
                base_url,
                username,
                api_token,
                "/content",
                &[
                    ("spaceKey", space.as_str()),
                    ("expand", "body.storage,version,ancestors"),
                    ("limit", "25"),
                    ("start", start_s.as_str()),
                ],
            )?;
            let pages = resp
                .get("results")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if pages.is_empty() {
                break;
            }

            for page in &pages {
                let title = value_str(page, "title");
                let html = page
                    .get("body")
                    .and_then(|body| body.get("storage"))
                    .and_then(|storage| storage.get("value"))
                    .and_then(Value::as_str)
                    .unwrap_or("");
                let content = html_to_text(html);
                if content.len() < 50 {
                    continue;
                }
                let ancestors = page
                    .get("ancestors")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(|ancestor| ancestor.get("title").and_then(Value::as_str))
                    .collect::<Vec<_>>()
                    .join(" > ");
                records.push(json!({
                    "type": "confluence",
                    "title": title,
                    "content": content,
                    "space": space,
                    "ancestors": ancestors,
                }));
                if max_pages > 0 && records.len() >= max_pages {
                    return Ok(records);
                }
            }

            start += pages.len();
        }
    }

    eprintln!("Scraped {} Confluence pages", records.len());
    Ok(records)
}

fn scrape_jira(
    client: &reqwest::blocking::Client,
    base_url: &str,
    username: &str,
    api_token: &str,
    projects: Vec<String>,
    max_issues: usize,
) -> Result<Vec<Value>> {
    if base_url.is_empty() || username.is_empty() || api_token.is_empty() {
        eprintln!("Jira not configured, skipping");
        return Ok(Vec::new());
    }

    let jql = if projects.is_empty() {
        "updated >= -730d ORDER BY updated DESC".to_string()
    } else {
        let clause = projects
            .iter()
            .map(|project| format!("project = {project}"))
            .collect::<Vec<_>>()
            .join(" OR ");
        format!("({clause}) ORDER BY updated DESC")
    };

    let mut records = Vec::new();
    let mut next_page_token = String::new();
    loop {
        let remaining = if max_issues == 0 {
            100
        } else {
            max_issues.saturating_sub(records.len()).min(100)
        };
        if remaining == 0 {
            break;
        }
        let max_results = remaining.to_string();
        let mut params = vec![
            ("jql", jql.as_str()),
            ("maxResults", max_results.as_str()),
            (
                "fields",
                "summary,description,comment,status,issuetype,priority,labels,project",
            ),
        ];
        if !next_page_token.is_empty() {
            params.push(("nextPageToken", next_page_token.as_str()));
        }

        let resp = jira_get(
            client,
            base_url,
            username,
            api_token,
            "/search/jql",
            &params,
        )?;
        let issues = resp
            .get("issues")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if issues.is_empty() {
            break;
        }

        for issue in issues {
            let fields = issue.get("fields").unwrap_or(&Value::Null);
            let key = value_str(&issue, "key");
            let summary = value_str(fields, "summary");
            let description = match fields.get("description") {
                Some(Value::String(s)) => s.trim().to_string(),
                Some(Value::Object(_)) => adf_to_text(fields.get("description").unwrap()),
                _ => String::new(),
            };
            let status = fields
                .get("status")
                .map(|status| value_str(status, "name"))
                .unwrap_or_default();
            let issue_type = fields
                .get("issuetype")
                .map(|issue_type| value_str(issue_type, "name"))
                .unwrap_or_default();
            let project = fields
                .get("project")
                .map(|project| value_str(project, "name"))
                .unwrap_or_default();

            let mut parts = Vec::new();
            if !description.is_empty() {
                parts.push(description);
            }
            parts.push(format!("Status: {status}"));
            parts.push(format!("Type: {issue_type}"));

            let comments = fields
                .get("comment")
                .and_then(|comment| comment.get("comments"))
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            if !comments.is_empty() {
                parts.push("\nComments:".to_string());
                for comment in comments {
                    let author = comment
                        .get("author")
                        .map(|author| value_str(author, "displayName"))
                        .unwrap_or_else(|| "Unknown".to_string());
                    let body = match comment.get("body") {
                        Some(Value::String(s)) => s.trim().to_string(),
                        Some(Value::Object(_)) => adf_to_text(comment.get("body").unwrap()),
                        Some(other) => other.to_string(),
                        None => String::new(),
                    };
                    if !body.is_empty() {
                        parts.push(format!("- {author}: {body}"));
                    }
                }
            }

            let content = parts.join("\n");
            if content.len() < 50 {
                continue;
            }
            records.push(json!({
                "type": "jira",
                "key": key,
                "summary": summary,
                "content": content,
                "project": project,
            }));
            if max_issues > 0 && records.len() >= max_issues {
                break;
            }
        }

        next_page_token = value_str(&resp, "nextPageToken");
        if next_page_token.is_empty()
            || resp.get("isLast").and_then(Value::as_bool).unwrap_or(false)
        {
            break;
        }
    }

    eprintln!("Scraped {} Jira issues", records.len());
    Ok(records)
}

fn confluence_get(
    client: &reqwest::blocking::Client,
    base_url: &str,
    username: &str,
    api_token: &str,
    path: &str,
    params: &[(&str, &str)],
) -> Result<Value> {
    get_json(
        client,
        &format!("{}/rest/api{}", base_url.trim_end_matches('/'), path),
        username,
        api_token,
        params,
    )
}

fn jira_get(
    client: &reqwest::blocking::Client,
    base_url: &str,
    username: &str,
    api_token: &str,
    path: &str,
    params: &[(&str, &str)],
) -> Result<Value> {
    get_json(
        client,
        &format!("{}/rest/api/3{}", base_url.trim_end_matches('/'), path),
        username,
        api_token,
        params,
    )
}

fn get_json(
    client: &reqwest::blocking::Client,
    url: &str,
    username: &str,
    api_token: &str,
    params: &[(&str, &str)],
) -> Result<Value> {
    for attempt in 1..=3 {
        let resp = client
            .get(url)
            .basic_auth(username, Some(api_token))
            .header("Accept", "application/json")
            .query(params)
            .send();
        match resp {
            Ok(resp) if resp.status().is_success() => return resp.json().context("parsing JSON"),
            Ok(resp) if attempt == 3 => {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                anyhow::bail!("Atlassian request failed: HTTP {status} — {body}");
            }
            Ok(_) | Err(_) => std::thread::sleep(Duration::from_secs(3 * attempt)),
        }
    }
    unreachable!()
}

fn html_to_text(html: &str) -> String {
    let mut text = html.to_string();
    for (pattern, replacement) in [
        (r"<br\s*/?>", "\n"),
        (r"<li[^>]*>", "- "),
        (r"<h[1-6][^>]*>", "\n## "),
        (r"</h[1-6]>", "\n"),
        (r"<p[^>]*>", "\n"),
        (r"<[^>]+>", ""),
        (r"&nbsp;", " "),
        (r"&amp;", "&"),
        (r"&lt;", "<"),
        (r"&gt;", ">"),
        (r"\n{3,}", "\n\n"),
    ] {
        text = regex_replace(&text, pattern, replacement);
    }
    text.trim().to_string()
}

fn regex_replace(text: &str, pattern: &str, replacement: &str) -> String {
    static CACHE: OnceLock<std::sync::Mutex<BTreeMap<String, Regex>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(BTreeMap::new()));
    let mut cache = cache.lock().unwrap();
    let re = cache
        .entry(pattern.to_string())
        .or_insert_with(|| Regex::new(pattern).unwrap());
    re.replace_all(text, replacement).into_owned()
}

fn adf_to_text(adf: &Value) -> String {
    let mut parts = Vec::new();
    if let Some(content) = adf.get("content") {
        walk_adf(content, &mut parts);
    }
    parts.join("").trim().to_string()
}

fn walk_adf(node: &Value, parts: &mut Vec<String>) {
    match node {
        Value::Array(items) => {
            for item in items {
                walk_adf(item, parts);
            }
        }
        Value::Object(map) => {
            let node_type = map.get("type").and_then(Value::as_str).unwrap_or("");
            match node_type {
                "text" => parts.push(
                    map.get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                ),
                "hardBreak" => parts.push("\n".to_string()),
                "heading" | "paragraph" => {
                    if let Some(content) = map.get("content") {
                        walk_adf(content, parts);
                    }
                    parts.push("\n".to_string());
                }
                "bulletList" => {
                    if let Some(items) = map.get("content").and_then(Value::as_array) {
                        for item in items {
                            parts.push("- ".to_string());
                            if let Some(content) = item.get("content") {
                                walk_adf(content, parts);
                            }
                        }
                    }
                }
                "codeBlock" => {
                    parts.push("```\n".to_string());
                    if let Some(content) = map.get("content") {
                        walk_adf(content, parts);
                    }
                    parts.push("\n```\n".to_string());
                }
                _ => {
                    if let Some(content) = map.get("content") {
                        walk_adf(content, parts);
                    }
                }
            }
        }
        _ => {}
    }
}

fn value_str(value: &Value, key: &str) -> String {
    value
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_basic_html() {
        assert_eq!(
            html_to_text("<p>Hello&nbsp;<b>world</b></p>"),
            "Hello world"
        );
    }

    #[test]
    fn converts_adf_text() {
        let adf = json!({
            "content": [
                {"type": "paragraph", "content": [{"type": "text", "text": "Hello"}]},
                {"type": "bulletList", "content": [{"content": [{"type": "text", "text": "Item"}]}]}
            ]
        });
        let text = adf_to_text(&adf);
        assert!(text.contains("Hello"));
        assert!(text.contains("- Item"));
    }
}
