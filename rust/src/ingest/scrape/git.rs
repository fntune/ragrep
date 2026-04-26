//! Scrape Git commit history with diffs.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use git2::{DiffFormat, DiffStatsFormat, Repository, Sort};
use regex::Regex;
use serde::Serialize;
use time::macros::format_description;
use time::{Date, OffsetDateTime, PrimitiveDateTime, Time, UtcOffset};

use super::{expand_repos, int_value, repo_name, string_list, string_value, write_jsonl};

const MAX_DIFF_BYTES: usize = 50_000;

fn pr_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"pull request #(\d+)").unwrap())
}

fn branch_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"Merged in (\S+)").unwrap())
}

fn diff_path_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^diff --git a/(\S+) b/").unwrap())
}

#[derive(Debug, Clone, Serialize)]
struct FileChange {
    status: String,
    path: String,
}

#[derive(Debug, Clone, Serialize)]
struct Record {
    repo: String,
    hash: String,
    author: String,
    date: String,
    subject: String,
    body: String,
    files_changed: Vec<FileChange>,
    diff_stat: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pr_number: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    diff: Option<String>,
}

pub fn scrape(raw_dir: &Path, config: &BTreeMap<String, toml::Value>) -> Result<usize> {
    let repos = string_list(config, "repos");
    if repos.is_empty() {
        eprintln!("No Git repos configured, skipping");
        return Ok(0);
    }

    let skip_repos: BTreeSet<String> = string_list(config, "skip_repos").into_iter().collect();
    let expanded = expand_repos(&repos)
        .into_iter()
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_none_or(|name| !skip_repos.contains(name))
        })
        .collect::<Vec<_>>();

    let max_commits = int_value(config, "max_commits").unwrap_or(0).max(0) as usize;
    let since = string_value(config, "since")
        .as_deref()
        .and_then(parse_since);
    let exclude_patterns = compile_patterns(string_list(config, "exclude_diff_patterns"))?;
    let exclude_authors: BTreeSet<String> =
        string_list(config, "exclude_authors").into_iter().collect();

    let mut records = Vec::new();
    for repo_path in expanded {
        match scrape_repo(
            &repo_path,
            max_commits,
            since,
            &exclude_patterns,
            &exclude_authors,
        ) {
            Ok(mut repo_records) => records.append(&mut repo_records),
            Err(err) => eprintln!(
                "warning: git scrape failed for {}: {err:#}",
                repo_path.display()
            ),
        }
    }

    let output = raw_dir.join("git.jsonl");
    write_jsonl(&output, &records)?;
    eprintln!("Git scrape complete: {} records written", records.len());
    Ok(records.len())
}

fn scrape_repo(
    repo_path: &Path,
    max_commits: usize,
    since: Option<i64>,
    exclude_patterns: &[Regex],
    exclude_authors: &BTreeSet<String>,
) -> Result<Vec<Record>> {
    let repo =
        Repository::open(repo_path).with_context(|| format!("opening {}", repo_path.display()))?;
    let name = repo_name(&repo, repo_path);
    let mut revwalk = repo.revwalk()?;
    revwalk.push_head()?;
    let _ = revwalk.set_sorting(Sort::TIME);

    let mut records = Vec::new();
    let mut skipped_large = 0usize;

    for (seen, oid_result) in revwalk.enumerate() {
        if max_commits > 0 && seen >= max_commits {
            break;
        }

        let oid = oid_result?;
        let commit = repo.find_commit(oid)?;
        let seconds = commit.time().seconds();
        if let Some(cutoff) = since {
            if seconds < cutoff {
                break;
            }
        }

        let subject = commit.summary().unwrap_or("").trim().to_string();
        if subject.is_empty() {
            continue;
        }

        let author = format_signature(&commit.author());
        if !exclude_authors.is_empty() {
            let author_name = author.split(" <").next().unwrap_or("");
            if exclude_authors.contains(author_name) {
                continue;
            }
        }

        let body = commit.body().unwrap_or("").trim().to_string();
        let (pr_number, branch) = parse_pr_info(&subject);
        let diff = commit_diff(&repo, &commit)?;
        let mut files_changed = files_changed(&diff);
        files_changed.retain(|file| !matches_any(&file.path, exclude_patterns));

        let mut patch = diff_patch(&diff)?;
        let diff_stat = diff_stat(&diff)?;
        if patch.len() > MAX_DIFF_BYTES {
            patch.clear();
        }
        if !patch.is_empty() {
            patch = filter_diff(&patch, exclude_patterns);
        }
        let diff = if patch.trim().is_empty() {
            if !files_changed.is_empty() {
                skipped_large += 1;
            }
            None
        } else {
            Some(patch)
        };

        records.push(Record {
            repo: name.clone(),
            hash: oid.to_string(),
            author,
            date: format_git_date(commit.time()),
            subject,
            body,
            files_changed,
            diff_stat,
            pr_number,
            branch,
            diff,
        });
    }

    eprintln!(
        "Scraped {} commits from {} ({} diffs skipped as >{}KB)",
        records.len(),
        name,
        skipped_large,
        MAX_DIFF_BYTES / 1024
    );
    Ok(records)
}

fn commit_diff<'repo>(
    repo: &'repo Repository,
    commit: &git2::Commit<'repo>,
) -> Result<git2::Diff<'repo>> {
    let tree = commit.tree()?;
    let parent_tree = if commit.parent_count() > 0 {
        Some(commit.parent(0)?.tree()?)
    } else {
        None
    };
    repo.diff_tree_to_tree(parent_tree.as_ref(), Some(&tree), None)
        .context("building commit diff")
}

fn files_changed(diff: &git2::Diff<'_>) -> Vec<FileChange> {
    diff.deltas()
        .filter_map(|delta| {
            let path = delta
                .new_file()
                .path()
                .or_else(|| delta.old_file().path())?
                .to_string_lossy()
                .to_string();
            let status = match delta.status() {
                git2::Delta::Added => "A",
                git2::Delta::Deleted => "D",
                git2::Delta::Renamed => "R",
                git2::Delta::Copied => "C",
                git2::Delta::Modified => "M",
                git2::Delta::Typechange => "T",
                _ => "M",
            };
            Some(FileChange {
                status: status.to_string(),
                path,
            })
        })
        .collect()
}

fn diff_patch(diff: &git2::Diff<'_>) -> Result<String> {
    let mut out = Vec::new();
    diff.print(DiffFormat::Patch, |_delta, _hunk, line| {
        out.extend_from_slice(line.content());
        true
    })?;
    Ok(String::from_utf8_lossy(&out).into_owned())
}

fn diff_stat(diff: &git2::Diff<'_>) -> Result<String> {
    let stats = diff.stats()?;
    let buf = stats.to_buf(DiffStatsFormat::FULL, 80)?;
    Ok(String::from_utf8_lossy(buf.as_ref()).trim().to_string())
}

fn filter_diff(diff: &str, exclude_patterns: &[Regex]) -> String {
    if exclude_patterns.is_empty() || diff.is_empty() {
        return diff.to_string();
    }

    let mut kept = Vec::new();
    for section in split_diff_sections(diff) {
        if section.trim().is_empty() {
            continue;
        }
        if let Some(caps) = diff_path_re().captures(&section) {
            if matches_any(&caps[1], exclude_patterns) {
                continue;
            }
        }
        kept.push(section);
    }
    kept.join("")
}

fn split_diff_sections(diff: &str) -> Vec<String> {
    let mut sections = Vec::new();
    let mut current = String::new();
    for line in diff.split_inclusive('\n') {
        if line.starts_with("diff --git a/") && !current.is_empty() {
            sections.push(std::mem::take(&mut current));
        }
        current.push_str(line);
    }
    if !current.is_empty() {
        sections.push(current);
    }
    sections
}

fn matches_any(path: &str, patterns: &[Regex]) -> bool {
    patterns.iter().any(|pattern| pattern.is_match(path))
}

fn compile_patterns(patterns: Vec<String>) -> Result<Vec<Regex>> {
    patterns
        .into_iter()
        .map(|pattern| Regex::new(&pattern).with_context(|| format!("invalid regex {pattern:?}")))
        .collect()
}

fn parse_pr_info(subject: &str) -> (Option<u64>, Option<String>) {
    let pr = pr_re()
        .captures(subject)
        .and_then(|caps| caps[1].parse::<u64>().ok());
    let branch = branch_re()
        .captures(subject)
        .map(|caps| caps[1].to_string());
    (pr, branch)
}

fn format_signature(sig: &git2::Signature<'_>) -> String {
    let name = sig.name().unwrap_or("");
    let email = sig.email().unwrap_or("");
    if email.is_empty() {
        name.to_string()
    } else {
        format!("{name} <{email}>")
    }
}

fn parse_since(since: &str) -> Option<i64> {
    let fmt = format_description!("[year]-[month]-[day]");
    let date = Date::parse(since, &fmt).ok()?;
    let dt = PrimitiveDateTime::new(date, Time::MIDNIGHT).assume_utc();
    Some(dt.unix_timestamp())
}

fn format_git_date(time: git2::Time) -> String {
    let seconds = time.seconds();
    let offset_seconds = time.offset_minutes() * 60;
    let Ok(offset) = UtcOffset::from_whole_seconds(offset_seconds) else {
        return seconds.to_string();
    };
    let Ok(dt) = OffsetDateTime::from_unix_timestamp(seconds) else {
        return seconds.to_string();
    };
    let dt = dt.to_offset(offset);
    let fmt = format_description!(
        "[year]-[month]-[day] [hour]:[minute]:[second] [offset_hour sign:mandatory][offset_minute]"
    );
    dt.format(&fmt).unwrap_or_else(|_| seconds.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bitbucket_merge_subject() {
        let (pr, branch) = parse_pr_info("Merged in feature/auth (pull request #123)");
        assert_eq!(pr, Some(123));
        assert_eq!(branch.as_deref(), Some("feature/auth"));
    }

    #[test]
    fn filters_diff_sections_by_path() {
        let diff = "diff --git a/keep.rs b/keep.rs\n+keep\ndiff --git a/generated/x.rs b/generated/x.rs\n+drop\n";
        let patterns = vec![Regex::new(r"generated/").unwrap()];
        let filtered = filter_diff(diff, &patterns);
        assert!(filtered.contains("keep.rs"));
        assert!(!filtered.contains("generated/x.rs"));
    }
}
