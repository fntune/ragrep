"""Scrape Bitbucket Cloud pull requests, comments, and reviews."""

import base64
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

_BASE_URL = "https://api.bitbucket.org/2.0"
_WORKERS = 10
_OAUTH_KEY = "DZTA4PpWMBMBUnrTJy"


def _get_oauth_token(oauth_secret: str) -> str:
    """Get a Bearer token via OAuth client_credentials grant."""
    url = "https://bitbucket.org/site/oauth2/access_token"
    data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "scopes": "repository",
    }).encode()
    req = urllib.request.Request(url, data=data)
    creds = base64.b64encode(f"{_OAUTH_KEY}:{oauth_secret}".encode()).decode()
    req.add_header("Authorization", f"Basic {creds}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    token = result["access_token"]
    log.info("Obtained Bitbucket OAuth token (expires in %ds)", result.get("expires_in", 0))
    return token


def _bb_request(url: str, token: str, retries: int = 3) -> dict:
    """Make an authenticated Bitbucket API request with retries."""
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    })
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError:
            raise  # don't retry HTTP errors (403, 404, etc.)
        except (TimeoutError, OSError) as e:
            if attempt == retries:
                raise
            log.warning("Request failed (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(3 * attempt)
    raise RuntimeError("unreachable")


def _paginate(url: str, token: str) -> list[dict]:
    """Follow Bitbucket pagination, collecting all values."""
    results: list[dict] = []
    while url:
        resp = _bb_request(url, token)
        results.extend(resp.get("values", []))
        url = resp.get("next", "")
    return results


def _fetch_pr_comments(workspace: str, repo: str, pr_id: int, token: str) -> list[dict]:
    """Fetch and normalize comments for a single PR."""
    url = f"{_BASE_URL}/repositories/{workspace}/{repo}/pullrequests/{pr_id}/comments?pagelen=100"
    raw_comments = _paginate(url, token)
    comments: list[dict] = []
    for c in raw_comments:
        content = c.get("content", {}).get("raw", "")
        if not content:
            continue
        inline = c.get("inline")
        comments.append({
            "author": c.get("user", {}).get("display_name", ""),
            "content": content,
            "created_on": c.get("created_on", ""),
            "path": inline.get("path") if inline else None,
            "line": inline.get("to") if inline else None,
        })
    return comments


def _fetch_pr_activity(workspace: str, repo: str, pr_id: int, token: str) -> list[str]:
    """Fetch PR activity and extract approver display names."""
    url = f"{_BASE_URL}/repositories/{workspace}/{repo}/pullrequests/{pr_id}/activity?pagelen=100"
    raw_activity = _paginate(url, token)
    approvers: list[str] = []
    seen: set[str] = set()
    for entry in raw_activity:
        approval = entry.get("approval")
        if approval:
            name = approval.get("user", {}).get("display_name", "")
            if name and name not in seen:
                approvers.append(name)
                seen.add(name)
    return approvers


def _build_pr_record(pr: dict, repo: str, workspace: str, token: str) -> dict:
    """Build a flat PR record with comments and approvals."""
    pr_id = pr["id"]
    comments = _fetch_pr_comments(workspace, repo, pr_id, token)
    approvals = _fetch_pr_activity(workspace, repo, pr_id, token)
    return {
        "repo": repo,
        "pr_id": pr_id,
        "title": pr.get("title", ""),
        "description": pr.get("description", "") or "",
        "author": pr.get("author", {}).get("display_name", ""),
        "state": pr.get("state", ""),
        "source_branch": pr.get("source", {}).get("branch", {}).get("name", ""),
        "target_branch": pr.get("destination", {}).get("branch", {}).get("name", ""),
        "created_on": pr.get("created_on", ""),
        "updated_on": pr.get("updated_on", ""),
        "comments": comments,
        "approvals": approvals,
    }


def _load_existing(output_path: Path) -> set[str]:
    """Load already-scraped PR keys (repo:pr_id) from existing output."""
    seen: set[str] = set()
    if not output_path.exists():
        return seen
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                key = f"{record['repo']}:{record['pr_id']}"
                seen.add(key)
    return seen


def _list_repos(workspace: str, token: str) -> list[str]:
    """List all repository slugs in a workspace."""
    url = f"{_BASE_URL}/repositories/{workspace}?pagelen=100"
    repos = _paginate(url, token)
    return [r["slug"] for r in repos]


def scrape(raw_dir: Path, config: dict) -> None:
    """Scrape Bitbucket Cloud PRs, write to raw/bitbucket.jsonl."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "bitbucket.jsonl"

    # Prefer workspace access token (Bearer), fall back to OAuth client_credentials
    token = os.environ.get("BITBUCKET_ACCESS_TOKEN", "")
    if not token:
        oauth_secret = os.environ.get("BITBUCKET_OAUTH_SECRET", "")
        if not oauth_secret:
            log.info("Neither BITBUCKET_ACCESS_TOKEN nor BITBUCKET_OAUTH_SECRET set, skipping")
            return
        token = _get_oauth_token(oauth_secret)

    workspace = config.get("workspace")
    if not workspace:
        log.warning("scrape.bitbucket.workspace not set in config, skipping")
        return
    repo_filter = config.get("repos", [])
    states = config.get("states", ["MERGED", "OPEN", "DECLINED"])
    max_prs_per_repo = config.get("max_prs_per_repo", 0)
    since = config.get("since", "")

    if repo_filter:
        repos = repo_filter
    else:
        log.info("Listing repos in workspace '%s'...", workspace)
        repos = _list_repos(workspace, token)
        log.info("Found %d repos", len(repos))

    already_done = _load_existing(output_path)
    if already_done:
        log.info("Resuming -- %d PRs already scraped", len(already_done))

    total_written = 0
    write_lock = threading.Lock()

    with open(output_path, "a") as out_f:
        for repo in repos:
            state_param = "&".join(f"state={s}" for s in states)
            url = f"{_BASE_URL}/repositories/{workspace}/{repo}/pullrequests?pagelen=50&{state_param}"
            log.info("Fetching PRs for %s/%s...", workspace, repo)
            try:
                all_prs = _paginate(url, token)
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    body = e.read().decode(errors="replace")
                    if "privilege scopes" in body:
                        log.error("OAuth token missing 'pullrequest' scope. "
                                  "Ask a workspace admin to update the OAuth consumer.")
                        return
                    log.warning("403 on %s/%s PRs: %s", workspace, repo, body[:200])
                    continue
                raise
            log.info("Found %d PRs in %s", len(all_prs), repo)

            # Filter by since date
            if since:
                all_prs = [
                    pr for pr in all_prs
                    if pr.get("updated_on", "") >= since
                ]

            # Filter already scraped
            prs = [
                pr for pr in all_prs
                if f"{repo}:{pr['id']}" not in already_done
            ]

            if max_prs_per_repo:
                prs = prs[:max_prs_per_repo]

            if not prs:
                log.info("No new PRs to scrape in %s", repo)
                continue

            done = 0
            with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
                futures = {
                    pool.submit(_build_pr_record, pr, repo, workspace, token): pr
                    for pr in prs
                }
                for fut in as_completed(futures):
                    done += 1
                    try:
                        record = fut.result()
                        with write_lock:
                            out_f.write(json.dumps(record) + "\n")
                            out_f.flush()
                        total_written += 1
                    except Exception as e:
                        pr = futures[fut]
                        log.warning("Failed to scrape PR %s/%s#%s: %s", workspace, repo, pr.get("id"), e)

                    if done % 50 == 0:
                        log.info("Progress: %d/%d PRs in %s", done, len(prs), repo)

    log.info("Bitbucket scrape complete: %d new PRs written (%d total on disk)",
             total_written, len(already_done) + total_written)
