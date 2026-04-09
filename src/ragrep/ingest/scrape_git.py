"""Scrape Git commit history with diffs.

Extracts commit metadata, PR info, files changed, and full diffs.
Content-hash deduplication at the embedding layer means identical
code blocks across commits/files/repos are embedded only once.
"""

import json
import logging
import re
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_PR_RE = re.compile(r"pull request #(\d+)")
_BRANCH_RE = re.compile(r"Merged in (\S+)")

# Skip diffs larger than this (bulk auto-generated, data dumps)
_MAX_DIFF_BYTES = 50_000

# Split unified diff by file section: "diff --git a/... b/..."
_DIFF_SECTION_RE = re.compile(r"(?=^diff --git a/)", re.MULTILINE)

# Extract file path from "diff --git a/path b/path"
_DIFF_PATH_RE = re.compile(r"^diff --git a/(\S+) b/")


def _filter_diff(diff: str, exclude_patterns: list[re.Pattern]) -> str:
    """Strip file sections from unified diff matching exclude patterns."""
    if not exclude_patterns or not diff:
        return diff
    sections = _DIFF_SECTION_RE.split(diff)
    kept = []
    for section in sections:
        if not section.strip():
            continue
        m = _DIFF_PATH_RE.match(section)
        if m:
            path = m.group(1)
            if any(p.search(path) for p in exclude_patterns):
                continue
        kept.append(section)
    return "".join(kept)


def _filter_files(files: list[dict], exclude_patterns: list[re.Pattern]) -> list[dict]:
    """Strip files matching exclude patterns from files_changed list."""
    if not exclude_patterns:
        return files
    return [f for f in files if not any(p.search(f["path"]) for p in exclude_patterns)]


def _run_git(repo_path: str, args: list[str], timeout: int = 60) -> str:
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git", "-C", repo_path, *args],
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        log.warning("git command failed in %s: %s", repo_path, stderr)
        return ""
    return result.stdout.decode("utf-8", errors="replace")


def _parse_pr_info(subject: str) -> tuple[int | None, str | None]:
    """Extract PR number and branch name from Bitbucket merge commit subject."""
    pr_match = _PR_RE.search(subject)
    pr_number = int(pr_match.group(1)) if pr_match else None
    branch_match = _BRANCH_RE.search(subject)
    branch = branch_match.group(1) if branch_match else None
    return pr_number, branch


def _get_diff(repo_path: str, commit_hash: str) -> str | None:
    """Get the diff for a commit. Returns None if too large or unavailable."""
    diff = _run_git(repo_path, ["diff", f"{commit_hash}^..{commit_hash}"], timeout=30)
    if not diff:
        return None
    if len(diff) > _MAX_DIFF_BYTES:
        return None
    return diff


def _get_files_changed(repo_path: str, commit_hash: str) -> list[dict]:
    """Get list of files changed with status (A/M/D/R)."""
    output = _run_git(repo_path, ["diff", "--name-status", f"{commit_hash}^..{commit_hash}"])
    if not output:
        return []
    files = []
    for line in output.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t", 2)
        if len(parts) >= 2:
            status = parts[0][0]  # first char: A, M, D, R, C
            filepath = parts[-1]  # last element (handles renames: R100\told\tnew)
            files.append({"status": status, "path": filepath})
    return files


def _get_diff_stat(repo_path: str, commit_hash: str) -> str:
    """Get diff --stat summary."""
    return _run_git(repo_path, ["diff", "--stat", f"{commit_hash}^..{commit_hash}"]).strip()


def scrape_repo(
    repo_path: str,
    max_commits: int = 0,
    since: str | None = None,
    exclude_patterns: list[re.Pattern] | None = None,
    exclude_authors: set[str] | None = None,
) -> list[dict]:
    """Extract commits with metadata and diffs from a single repo."""
    remote = _run_git(repo_path, ["remote", "get-url", "origin"]).strip()
    repo_name = remote.split("/")[-1].replace(".git", "") if remote else Path(repo_path).name
    _exclude = exclude_patterns or []
    _skip_authors = exclude_authors or set()

    log_args = [
        "log",
        "--format=%H%n%an <%ae>%n%ai%n%s%n%b%n---END---",
    ]
    if max_commits:
        log_args.append(f"-n{max_commits}")
    if since:
        log_args.append(f"--since={since}")

    output = _run_git(repo_path, log_args, timeout=120)
    if not output:
        return []

    records: list[dict] = []
    skipped_large = 0
    commits = output.split("---END---\n")

    for commit_block in commits:
        commit_block = commit_block.strip()
        if not commit_block:
            continue

        lines = commit_block.split("\n", 4)
        if len(lines) < 4:
            continue

        commit_hash = lines[0]
        author = lines[1]
        date = lines[2]
        subject = lines[3]
        body = lines[4].strip() if len(lines) > 4 else ""

        if not subject.strip():
            continue

        # Skip excluded authors (match on name before <email>)
        if _skip_authors:
            author_name = author.split(" <")[0]
            if author_name in _skip_authors:
                continue

        pr_number, branch = _parse_pr_info(subject)

        # Get files changed (lightweight — always)
        files_changed = _get_files_changed(repo_path, commit_hash)
        files_changed = _filter_files(files_changed, _exclude)

        # Get full diff (skip if too large)
        diff = _get_diff(repo_path, commit_hash)
        if diff is not None:
            diff = _filter_diff(diff, _exclude)
            if not diff.strip():
                diff = None
        if diff is None and files_changed:
            skipped_large += 1

        # Get diff stat summary (always — useful even when diff skipped)
        diff_stat = _get_diff_stat(repo_path, commit_hash)

        record: dict = {
            "repo": repo_name,
            "hash": commit_hash,
            "author": author,
            "date": date,
            "subject": subject,
            "body": body,
            "files_changed": files_changed,
            "diff_stat": diff_stat,
        }

        if pr_number is not None:
            record["pr_number"] = pr_number
        if branch:
            record["branch"] = branch
        if diff:
            record["diff"] = diff

        records.append(record)

    log.info(
        "Scraped %d commits from %s (%d diffs skipped as >%dKB)",
        len(records), repo_name, skipped_large, _MAX_DIFF_BYTES // 1024,
    )
    return records


def _expand_repos(patterns: list[str]) -> list[str]:
    """Expand glob patterns in repo paths, filtering to git repos only."""
    import glob
    paths: list[str] = []
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser())
        matches = sorted(glob.glob(expanded))
        if not matches:
            paths.append(pattern)
            continue
        for m in matches:
            p = Path(m).resolve()
            if p.is_dir() and (p / ".git").is_dir():
                paths.append(str(p))
    # Deduplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in paths:
        resolved = str(Path(p).resolve())
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def scrape(raw_dir: Path, config: dict) -> None:
    """Scrape all configured Git repos, write to raw/git.jsonl."""
    repos = config.get("repos", [])
    if not repos:
        log.info("No Git repos configured, skipping")
        return

    expanded_repos = _expand_repos(repos)
    skip_repos = set(config.get("skip_repos", []))
    if skip_repos:
        before = len(expanded_repos)
        expanded_repos = [p for p in expanded_repos if Path(p).name not in skip_repos]
        log.info("Skipped %d repos: %s", before - len(expanded_repos), skip_repos)
    log.info("Resolved %d git repos from %d patterns", len(expanded_repos), len(repos))

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "git.jsonl"

    max_commits = config.get("max_commits", 0)
    since = config.get("since")

    # Compile diff path exclusion patterns
    exclude_patterns = [re.compile(p) for p in config.get("exclude_diff_patterns", [])]
    if exclude_patterns:
        log.info("Excluding diff paths matching: %s", config["exclude_diff_patterns"])

    exclude_authors = set(config.get("exclude_authors", []))
    if exclude_authors:
        log.info("Excluding authors: %s", exclude_authors)

    records: list[dict] = []
    for repo_path in expanded_repos:
        if not Path(repo_path).is_dir():
            log.warning("Repo not found: %s", repo_path)
            continue
        records.extend(scrape_repo(repo_path, max_commits, since, exclude_patterns, exclude_authors))

    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    log.info("Git scrape complete: %d records written", len(records))
