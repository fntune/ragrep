"""Scrape Confluence pages and Jira issues."""

import json
import logging
import os
import re
from pathlib import Path

log = logging.getLogger(__name__)


def _http_request(url: str, username: str, api_token: str, retries: int = 3) -> dict:
    """Make an authenticated request with retries."""
    import base64
    import time
    import urllib.request

    credentials = base64.b64encode(f"{username}:{api_token}".encode()).decode()
    req = urllib.request.Request(url, headers={
        "Authorization": f"Basic {credentials}",
        "Accept": "application/json",
    })
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except (TimeoutError, OSError) as e:
            if attempt == retries:
                raise
            log.warning("Request failed (attempt %d/%d): %s", attempt, retries, e)
            time.sleep(3 * attempt)
    raise RuntimeError("unreachable")


def _confluence_request(base_url: str, path: str, username: str, api_token: str, params: dict | None = None) -> dict:
    import urllib.parse

    url = f"{base_url.rstrip('/')}/rest/api{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return _http_request(url, username, api_token)


def _jira_request(base_url: str, path: str, username: str, api_token: str, params: dict | None = None) -> dict:
    import urllib.parse

    url = f"{base_url.rstrip('/')}/rest/api/3{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return _http_request(url, username, api_token)


def _html_to_text(html: str) -> str:
    """Strip HTML tags to plain text."""
    text = re.sub(r"<br\s*/?>", "\n", html)
    text = re.sub(r"<li[^>]*>", "- ", text)
    text = re.sub(r"<h[1-6][^>]*>", "\n## ", text)
    text = re.sub(r"</h[1-6]>", "\n", text)
    text = re.sub(r"<p[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _adf_to_text(adf: dict) -> str:
    """Convert Atlassian Document Format to plain text."""
    if not adf or not isinstance(adf, dict):
        return ""

    parts: list[str] = []

    def walk(node: dict | list) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return

        node_type = node.get("type", "")
        if node_type == "text":
            parts.append(node.get("text", ""))
        elif node_type == "hardBreak":
            parts.append("\n")
        elif node_type in ("heading", "paragraph"):
            walk(node.get("content", []))
            parts.append("\n")
        elif node_type == "bulletList":
            for item in node.get("content", []):
                parts.append("- ")
                walk(item.get("content", []))
        elif node_type == "codeBlock":
            parts.append("```\n")
            walk(node.get("content", []))
            parts.append("\n```\n")
        else:
            walk(node.get("content", []))

    walk(adf.get("content", []))
    return "".join(parts).strip()


def scrape_confluence(
    confluence_url: str,
    username: str,
    api_token: str,
    spaces: list[str],
    max_pages: int = 0,
) -> list[dict]:
    """Scrape Confluence pages. Returns flat records."""
    if not confluence_url or not username or not api_token:
        log.info("Confluence not configured, skipping")
        return []

    records: list[dict] = []

    # Get spaces — skip personal spaces (~xxx)
    if not spaces:
        resp = _confluence_request(confluence_url, "/space", username, api_token, {"limit": 100})
        all_spaces = [s["key"] for s in resp.get("results", [])]
        spaces = [k for k in all_spaces if not k.startswith("~")]
        log.info("Found %d Confluence spaces (%d personal skipped)", len(spaces), len(all_spaces) - len(spaces))

    total_pages = 0
    for space_key in spaces:
        start = 0
        while True:
            params = {
                "spaceKey": space_key,
                "expand": "body.storage,version,ancestors",
                "limit": 25,
                "start": start,
            }
            resp = _confluence_request(confluence_url, "/content", username, api_token, params)
            pages = resp.get("results", [])
            if not pages:
                break

            for page in pages:
                title = page.get("title", "")
                html_body = page.get("body", {}).get("storage", {}).get("value", "")
                text = _html_to_text(html_body)

                if not text or len(text) < 50:
                    continue

                ancestors = " > ".join(a.get("title", "") for a in page.get("ancestors", []))

                records.append({
                    "type": "confluence",
                    "title": title,
                    "content": text,
                    "space": space_key,
                    "ancestors": ancestors,
                })
                total_pages += 1

                if max_pages and total_pages >= max_pages:
                    break

            start += len(pages)
            if max_pages and total_pages >= max_pages:
                break

        if max_pages and total_pages >= max_pages:
            break

    log.info("Scraped %d Confluence pages", total_pages)
    return records


def scrape_jira(
    jira_url: str,
    username: str,
    api_token: str,
    projects: list[str],
    max_issues: int = 0,
) -> list[dict]:
    """Scrape Jira issues. Returns flat records."""
    if not jira_url or not username or not api_token:
        log.info("Jira not configured, skipping")
        return []

    records: list[dict] = []

    if projects:
        project_clause = " OR ".join(f"project = {p}" for p in projects)
        jql = f"({project_clause}) ORDER BY updated DESC"
    else:
        jql = "updated >= -730d ORDER BY updated DESC"

    total_issues = 0
    next_page_token: str | None = None

    while True:
        batch_size = 100
        if max_issues:
            remaining = max_issues - total_issues
            if remaining <= 0:
                break
            batch_size = min(100, remaining)
        params: dict = {
            "jql": jql,
            "maxResults": batch_size,
            "fields": "summary,description,comment,status,issuetype,priority,labels,project",
        }
        if next_page_token:
            params["nextPageToken"] = next_page_token
        try:
            resp = _jira_request(jira_url, "/search/jql", username, api_token, params)
        except Exception as e:
            log.warning("Jira search failed (at %d issues): %s — retrying", total_issues, e)
            import time
            time.sleep(5)
            try:
                resp = _jira_request(jira_url, "/search/jql", username, api_token, params)
            except Exception as e2:
                log.warning("Jira search failed again: %s — stopping", e2)
                break
        issues = resp.get("issues", [])
        if not issues:
            break

        for issue in issues:
            key = issue.get("key", "")
            fields = issue.get("fields", {})
            summary = fields.get("summary", "")
            raw_desc = fields.get("description")
            if isinstance(raw_desc, dict):
                description = _adf_to_text(raw_desc)
            elif isinstance(raw_desc, str):
                description = raw_desc.strip()
            else:
                description = ""
            status = fields.get("status", {}).get("name", "")
            issue_type = fields.get("issuetype", {}).get("name", "")
            project_name = fields.get("project", {}).get("name", "")

            content_parts = []
            if description:
                content_parts.append(description)
            content_parts.append(f"Status: {status}")
            content_parts.append(f"Type: {issue_type}")

            # All comments, no cap
            comments = fields.get("comment", {}).get("comments", [])
            if comments:
                content_parts.append("\nComments:")
                for comment in comments:
                    author = comment.get("author", {}).get("displayName", "Unknown")
                    raw_body = comment.get("body", "")
                    body = _adf_to_text(raw_body) if isinstance(raw_body, dict) else str(raw_body).strip()
                    if body:
                        content_parts.append(f"- {author}: {body}")

            content = "\n".join(content_parts)
            if len(content) < 50:
                continue

            records.append({
                "type": "jira",
                "key": key,
                "summary": summary,
                "content": content,
                "project": project_name,
            })
            total_issues += 1

        next_page_token = resp.get("nextPageToken")
        if not next_page_token or resp.get("isLast", False):
            break

    log.info("Scraped %d Jira issues", total_issues)
    return records


def scrape(raw_dir: Path, config: dict) -> None:
    """Scrape Confluence and Jira, write to raw/atlassian.jsonl."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "atlassian.jsonl"

    confluence_url = os.environ.get("CONFLUENCE_URL", config.get("confluence_url", ""))
    jira_url = os.environ.get("JIRA_URL", config.get("jira_url", ""))
    username = os.environ.get("ATLASSIAN_USERNAME", config.get("username", ""))
    api_token = os.environ.get("ATLASSIAN_API_TOKEN", config.get("api_token", ""))
    spaces = config.get("confluence_spaces", [])
    projects = config.get("jira_projects", [])
    max_pages = config.get("max_pages", 0)

    records = []
    records.extend(scrape_confluence(confluence_url, username, api_token, spaces, max_pages))
    records.extend(scrape_jira(jira_url, username, api_token, projects, max_pages))

    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    log.info("Atlassian scrape complete: %d records written", len(records))
