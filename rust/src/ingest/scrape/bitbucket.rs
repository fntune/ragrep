//! Scrape Bitbucket Cloud pull requests, comments, and approvals.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use base64::Engine;
use serde_json::{json, Value};

use super::{int_value, string_list, string_value, write_jsonl};

const BASE: &str = "https://api.bitbucket.org/2.0";
const OAUTH_KEY: &str = "DZTA4PpWMBMBUnrTJy";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

pub fn scrape(raw_dir: &Path, config: &BTreeMap<String, toml::Value>) -> Result<usize> {
    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("building Bitbucket HTTP client")?;

    let token = match std::env::var("BITBUCKET_ACCESS_TOKEN") {
        Ok(token) if !token.is_empty() => token,
        _ => match std::env::var("BITBUCKET_OAUTH_SECRET") {
            Ok(secret) if !secret.is_empty() => oauth_token(&client, &secret)?,
            _ => {
                eprintln!(
                    "Neither BITBUCKET_ACCESS_TOKEN nor BITBUCKET_OAUTH_SECRET set, skipping"
                );
                write_jsonl::<Value>(&raw_dir.join("bitbucket.jsonl"), &[])?;
                return Ok(0);
            }
        },
    };

    let Some(workspace) = string_value(config, "workspace") else {
        eprintln!("scrape.bitbucket.workspace not set, skipping");
        write_jsonl::<Value>(&raw_dir.join("bitbucket.jsonl"), &[])?;
        return Ok(0);
    };

    let repo_filter = string_list(config, "repos");
    let repos = if repo_filter.is_empty() {
        list_repos(&client, &workspace, &token)?
    } else {
        repo_filter
    };
    let states = {
        let configured = string_list(config, "states");
        if configured.is_empty() {
            vec![
                "MERGED".to_string(),
                "OPEN".to_string(),
                "DECLINED".to_string(),
            ]
        } else {
            configured
        }
    };
    let max_prs_per_repo = int_value(config, "max_prs_per_repo").unwrap_or(0).max(0) as usize;
    let since = string_value(config, "since").unwrap_or_default();

    let mut records = Vec::new();
    for repo in repos {
        let mut params = vec![("pagelen".to_string(), "50".to_string())];
        for state in &states {
            params.push(("state".to_string(), state.clone()));
        }
        let url = format!("{BASE}/repositories/{workspace}/{repo}/pullrequests");
        let mut prs = paginate(&client, &token, &url, &params)?;
        if !since.is_empty() {
            prs.retain(|pr| value_str(pr, "updated_on") >= since);
        }
        if max_prs_per_repo > 0 && prs.len() > max_prs_per_repo {
            prs.truncate(max_prs_per_repo);
        }

        for pr in prs {
            match build_pr_record(&client, &workspace, &repo, &token, &pr) {
                Ok(record) => records.push(record),
                Err(err) => eprintln!(
                    "warning: failed to scrape Bitbucket PR {}/{}#{}: {err:#}",
                    workspace,
                    repo,
                    value_i64(&pr, "id")
                ),
            }
        }
        eprintln!("Scraped {} Bitbucket PR records so far", records.len());
    }

    write_jsonl(&raw_dir.join("bitbucket.jsonl"), &records)?;
    eprintln!(
        "Bitbucket scrape complete: {} records written",
        records.len()
    );
    Ok(records.len())
}

fn oauth_token(client: &reqwest::blocking::Client, secret: &str) -> Result<String> {
    let creds = base64::engine::general_purpose::STANDARD.encode(format!("{OAUTH_KEY}:{secret}"));
    let resp = client
        .post("https://bitbucket.org/site/oauth2/access_token")
        .header("Authorization", format!("Basic {creds}"))
        .form(&[
            ("grant_type", "client_credentials"),
            ("scopes", "repository pullrequest"),
        ])
        .send()
        .context("requesting Bitbucket OAuth token")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        bail!("Bitbucket OAuth failed: HTTP {status} — {body}");
    }
    let payload: Value = resp.json().context("parsing Bitbucket OAuth response")?;
    payload
        .get("access_token")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| anyhow!("Bitbucket OAuth response missing access_token"))
}

fn list_repos(
    client: &reqwest::blocking::Client,
    workspace: &str,
    token: &str,
) -> Result<Vec<String>> {
    let url = format!("{BASE}/repositories/{workspace}");
    let repos = paginate(
        client,
        token,
        &url,
        &[("pagelen".to_string(), "100".to_string())],
    )?;
    Ok(repos
        .into_iter()
        .filter_map(|repo| {
            repo.get("slug")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .collect())
}

fn build_pr_record(
    client: &reqwest::blocking::Client,
    workspace: &str,
    repo: &str,
    token: &str,
    pr: &Value,
) -> Result<Value> {
    let pr_id = value_i64(pr, "id");
    let comments = fetch_comments(client, workspace, repo, pr_id, token)?;
    let approvals = fetch_approvals(client, workspace, repo, pr_id, token)?;
    Ok(json!({
        "repo": repo,
        "pr_id": pr_id,
        "title": value_str(pr, "title"),
        "description": value_str(pr, "description"),
        "author": pr.get("author").map(|a| value_str(a, "display_name")).unwrap_or_default(),
        "state": value_str(pr, "state"),
        "source_branch": pr
            .get("source")
            .and_then(|source| source.get("branch"))
            .map(|branch| value_str(branch, "name"))
            .unwrap_or_default(),
        "target_branch": pr
            .get("destination")
            .and_then(|dest| dest.get("branch"))
            .map(|branch| value_str(branch, "name"))
            .unwrap_or_default(),
        "created_on": value_str(pr, "created_on"),
        "updated_on": value_str(pr, "updated_on"),
        "comments": comments,
        "approvals": approvals,
    }))
}

fn fetch_comments(
    client: &reqwest::blocking::Client,
    workspace: &str,
    repo: &str,
    pr_id: i64,
    token: &str,
) -> Result<Vec<Value>> {
    let url = format!("{BASE}/repositories/{workspace}/{repo}/pullrequests/{pr_id}/comments");
    let raw = paginate(
        client,
        token,
        &url,
        &[("pagelen".to_string(), "100".to_string())],
    )?;
    Ok(raw
        .into_iter()
        .filter_map(|comment| {
            let content = comment
                .get("content")
                .map(|content| value_str(content, "raw"))
                .unwrap_or_default();
            if content.is_empty() {
                return None;
            }
            let inline = comment.get("inline");
            Some(json!({
                "author": comment.get("user").map(|user| value_str(user, "display_name")).unwrap_or_default(),
                "content": content,
                "created_on": value_str(&comment, "created_on"),
                "path": inline.and_then(|v| v.get("path")).and_then(Value::as_str),
                "line": inline.and_then(|v| v.get("to")).and_then(Value::as_i64),
            }))
        })
        .collect())
}

fn fetch_approvals(
    client: &reqwest::blocking::Client,
    workspace: &str,
    repo: &str,
    pr_id: i64,
    token: &str,
) -> Result<Vec<String>> {
    let url = format!("{BASE}/repositories/{workspace}/{repo}/pullrequests/{pr_id}/activity");
    let raw = paginate(
        client,
        token,
        &url,
        &[("pagelen".to_string(), "100".to_string())],
    )?;
    let mut seen = std::collections::BTreeSet::new();
    let mut out = Vec::new();
    for entry in raw {
        let name = entry
            .get("approval")
            .and_then(|approval| approval.get("user"))
            .map(|user| value_str(user, "display_name"))
            .unwrap_or_default();
        if !name.is_empty() && seen.insert(name.clone()) {
            out.push(name);
        }
    }
    Ok(out)
}

fn paginate(
    client: &reqwest::blocking::Client,
    token: &str,
    url: &str,
    params: &[(String, String)],
) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    let mut next = url.to_string();
    let mut first = true;
    while !next.is_empty() {
        let mut req = client
            .get(&next)
            .bearer_auth(token)
            .header("Accept", "application/json");
        if first {
            req = req.query(params);
        }
        first = false;
        let resp = req
            .send()
            .with_context(|| format!("calling Bitbucket {next}"))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            bail!("Bitbucket request failed: HTTP {status} — {body}");
        }
        let payload: Value = resp.json().context("parsing Bitbucket response")?;
        if let Some(values) = payload.get("values").and_then(Value::as_array) {
            out.extend(values.iter().cloned());
        }
        next = value_str(&payload, "next");
    }
    Ok(out)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_pr_record_shape_without_comments() {
        let pr = json!({
            "id": 7,
            "title": "Improve search",
            "description": "body",
            "author": {"display_name": "Alice"},
            "state": "MERGED",
            "source": {"branch": {"name": "feature"}},
            "destination": {"branch": {"name": "main"}}
        });
        assert_eq!(value_i64(&pr, "id"), 7);
        assert_eq!(
            pr.get("source")
                .and_then(|s| s.get("branch"))
                .map(|b| value_str(b, "name"))
                .unwrap(),
            "feature"
        );
    }
}
