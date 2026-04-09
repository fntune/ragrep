"""Normalize raw JSONL sources into Document objects."""

import hashlib
import json
import logging
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from ragrep.models import Document

log = logging.getLogger(__name__)

# Slack mrkdwn patterns (ported from slack-finetune/src/processor.py)
_CHANNEL_LINK = re.compile(r"<#\w+\|([^>]+)>")
_LABELED_LINK = re.compile(r"<(https?://[^|>]+)\|([^>]+)>")
_BARE_LINK = re.compile(r"<(https?://[^>]+)>")
_USER_MENTION = re.compile(r"<@(\w+)>")
_EMOJI = re.compile(r":[\w+\-]+:")
_BOLD = re.compile(r"\*([^*]+)\*")
_ITALIC = re.compile(r"_([^_]+)_")
_STRIKE = re.compile(r"~([^~]+)~")
_MAILTO = re.compile(r"<mailto:([^|>]+)\|([^>]+)>")
_TEL = re.compile(r"<tel:([^|>]+)\|([^>]+)>")

_SYSTEM_SUBTYPES = {
    "channel_join", "channel_leave", "channel_topic",
    "channel_purpose", "channel_name", "bot_message",
    "file_comment", "tombstone",
}


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def clean_text(text: str, user_map: dict[str, dict]) -> str:
    """Strip Slack formatting, replace user IDs with display names."""
    text = _USER_MENTION.sub(lambda m: f"@{user_map.get(m.group(1), {}).get('name', m.group(1))}", text)
    text = _MAILTO.sub(r"\2", text)
    text = _TEL.sub(r"\2", text)
    text = _CHANNEL_LINK.sub(r"#\1", text)
    text = _LABELED_LINK.sub(r"\2", text)
    text = _BARE_LINK.sub(r"\1", text)
    text = _EMOJI.sub("", text)
    text = _BOLD.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = _STRIKE.sub(r"\1", text)
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def _user_label(user_id: str, user_map: dict[str, dict]) -> str:
    info = user_map.get(user_id, {})
    name = info.get("name", user_id)
    title = info.get("title", "")
    return f"{name} ({title})" if title else name


def load_users(path: Path) -> dict[str, dict]:
    """Load users.json → {user_id: {name, title, ...}}."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    # Handle both list and dict formats
    if isinstance(data, list):
        return {u["id"]: u for u in data if "id" in u}
    return data


def load_channels(path: Path) -> dict[str, dict]:
    """Load channels.json → {channel_id: {name, topic, purpose, ...}}."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {c["id"]: c for c in data if "id" in c}
    return data


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        log.warning("File not found: %s", path)
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_messages(raw_dir: Path, user_map: dict[str, dict], channel_map: dict[str, dict]) -> list[Document]:
    """Normalize Slack messages. Group threads by thread_ts, concatenate with speaker labels."""
    records = _read_jsonl(raw_dir / "messages.jsonl")
    if not records:
        return []

    # Group by thread
    threads: dict[str, list[dict]] = defaultdict(list)
    standalone: list[dict] = []

    for msg in records:
        if msg.get("subtype", "") in _SYSTEM_SUBTYPES:
            continue
        thread_ts = msg.get("thread_ts", "")
        if thread_ts:
            threads[thread_ts].append(msg)
        elif msg.get("reply_count", 0) > 0:
            # Thread parent without thread_ts in replies — use its own ts
            threads[msg["ts"]].append(msg)
        else:
            standalone.append(msg)

    docs: list[Document] = []

    # Process threads
    for thread_ts, msgs in threads.items():
        msgs.sort(key=lambda m: float(m.get("ts", 0)))
        channel = msgs[0].get("channel_name", "unknown")
        channel_id = msgs[0].get("channel_id", "")

        parts = []
        for msg in msgs:
            user = _user_label(msg.get("user_id", ""), user_map)
            text = clean_text(msg.get("text", ""), user_map)
            if text:
                parts.append(f"{user}: {text}")

        content = "\n".join(parts)
        if len(content) < 20:
            continue

        first_text = clean_text(msgs[0].get("text", ""), user_map)
        title = f"#{channel}: {first_text[:80]}"

        ch_meta = channel_map.get(channel_id, {})
        first_ts = float(msgs[0].get("ts", 0))
        metadata: dict[str, str | int | float] = {
            "channel_name": channel,
            "message_count": len(msgs),
            "reaction_count": sum(m.get("reaction_count", 0) for m in msgs),
            "date": datetime.fromtimestamp(first_ts, tz=UTC).strftime("%Y-%m-%d"),
        }
        topic = ch_meta.get("topic", "")
        if topic:
            metadata["channel_topic"] = topic

        docs.append(Document(
            id=f"slack:{channel}:{thread_ts}",
            source="slack",
            content=content,
            title=title,
            metadata=metadata,
        ))

    # Process standalone messages
    for msg in standalone:
        text = clean_text(msg.get("text", ""), user_map)
        if len(text) < 20:
            continue
        channel = msg.get("channel_name", "unknown")
        user = _user_label(msg.get("user_id", ""), user_map)
        ts = msg.get("ts", "0")

        docs.append(Document(
            id=f"slack:{channel}:{ts}",
            source="slack",
            content=f"{user}: {text}",
            title=f"#{channel}: {text[:80]}",
            metadata={
                "channel_name": channel,
                "message_count": 1,
                "reaction_count": msg.get("reaction_count", 0),
                "date": datetime.fromtimestamp(float(ts), tz=UTC).strftime("%Y-%m-%d"),
            },
        ))

    log.info("Normalized %d Slack documents (%d threads, %d standalone)", len(docs), len(threads), len(standalone))
    return docs


def normalize_atlassian(raw_dir: Path) -> list[Document]:
    """Normalize Atlassian records from flat JSONL format."""
    records = _read_jsonl(raw_dir / "atlassian.jsonl")
    docs = []

    for rec in records:
        content = rec.get("content", "")
        if len(content) < 20:
            continue

        source_type = rec.get("type", "atlassian")

        if source_type == "jira":
            title = f"{rec.get('key', '')}: {rec.get('summary', '')}"
            space = rec.get("project", "")
        else:
            title = rec.get("title", "")
            space = rec.get("space", "")

        doc_id = f"atlassian:{_hash(title)}"
        docs.append(Document(
            id=doc_id,
            source="atlassian",
            content=content,
            title=title[:120],
            metadata={"source_type": source_type, "space": space},
        ))

    log.info("Normalized %d Atlassian documents", len(docs))
    return docs


def normalize_gdrive(raw_dir: Path) -> list[Document]:
    """Normalize Google Drive records from flat JSONL format. Skip spreadsheets."""
    records = _read_jsonl(raw_dir / "gdrive.jsonl")
    docs = []
    skipped_csv = 0

    for rec in records:
        if "spreadsheet" in rec.get("mime_type", "") or rec.get("file_type", "") == "Google Sheet":
            skipped_csv += 1
            continue

        content = rec.get("content", "").strip()
        if len(content) < 20:
            continue

        title = rec.get("name", "")
        file_id = rec.get("file_id", _hash(title))
        path = rec.get("path", "")
        doc_type = rec.get("file_type", "doc").lower().replace("google ", "")

        docs.append(Document(
            id=f"gdrive:{file_id}",
            source="gdrive",
            content=content,
            title=title[:120],
            metadata={"path": path, "doc_type": doc_type},
        ))

    log.info("Normalized %d GDrive documents (skipped %d CSVs)", len(docs), skipped_csv)
    return docs


def normalize_git(raw_dir: Path) -> list[Document]:
    """Normalize git commit records from flat JSONL format.

    Commits with diffs produce richer documents that include the actual
    code changes. The content-hash embedding cache deduplicates identical
    code blocks across commits, files, and repos.
    """
    records = _read_jsonl(raw_dir / "git.jsonl")
    docs = []

    for rec in records:
        subject = rec.get("subject", "").strip()
        body = rec.get("body", "").strip()
        repo = rec.get("repo", "")
        commit = rec.get("hash", "")
        author = rec.get("author", "")
        date = rec.get("date", "")
        diff = rec.get("diff", "")
        diff_stat = rec.get("diff_stat", "")
        files_changed = rec.get("files_changed", [])
        pr_number = rec.get("pr_number")
        branch = rec.get("branch", "")

        # Build content: commit message + files + diff
        parts = [subject]
        if body:
            parts.append(body)

        parts.append(f"Author: {author}")
        parts.append(f"Date: {date}")
        if pr_number:
            parts.append(f"PR: #{pr_number}")
        if branch:
            parts.append(f"Branch: {branch}")

        if files_changed:
            file_lines = [f"  {f['status']} {f['path']}" for f in files_changed]
            parts.append("Files changed:\n" + "\n".join(file_lines))

        if diff:
            parts.append(f"Diff:\n{diff}")
        elif diff_stat:
            parts.append(f"Diff stat:\n{diff_stat}")

        content = "\n".join(parts)

        if len(content) < 10:
            continue

        # Title: include PR# if available
        if pr_number:
            title = f"{repo} PR#{pr_number}: {subject[:70]}"
        else:
            title = f"{repo}: {subject[:80]}"

        metadata: dict[str, str | int | float] = {
            "repo": repo,
            "author": author,
            "date": date,
        }
        if pr_number:
            metadata["pr_number"] = pr_number
        if branch:
            metadata["branch"] = branch
        if files_changed:
            metadata["files_count"] = len(files_changed)

        doc_id = f"git:{repo}:{commit}" if commit else f"git:{_hash(content)}"
        docs.append(Document(
            id=doc_id,
            source="git",
            content=content,
            title=title,
            metadata=metadata,
        ))

    log.info("Normalized %d git documents", len(docs))
    return docs


def normalize_files(raw_dir: Path) -> list[Document]:
    """Normalize extracted file content."""
    records = _read_jsonl(raw_dir / "files_extracted.jsonl")
    docs = []

    for rec in records:
        content = rec.get("content", "").strip()
        if len(content) < 20:
            continue

        file_id = rec.get("file_id", "")
        file_name = rec.get("file_name", "unknown")
        channel = rec.get("channel_name", "")

        docs.append(Document(
            id=f"file:{file_id}",
            source="file",
            content=content,
            title=file_name,
            metadata={
                "file_type": rec.get("file_type", ""),
                "extraction_method": rec.get("extraction_method", ""),
                "channel_name": channel,
            },
        ))

    log.info("Normalized %d file documents", len(docs))
    return docs


def normalize_bookmarks(raw_dir: Path) -> list[Document]:
    """Normalize Slack bookmarks."""
    records = _read_jsonl(raw_dir / "bookmarks.jsonl")
    docs = []

    for rec in records:
        title = rec.get("title", "")
        link = rec.get("link", "")
        if not title or not link:
            continue

        channel = rec.get("channel_name", "")
        content = f"{title}: {link}"

        docs.append(Document(
            id=f"bookmark:{_hash(link)}",
            source="bookmark",
            content=content,
            title=title,
            metadata={"channel_name": channel, "link": link},
        ))

    log.info("Normalized %d bookmarks", len(docs))
    return docs


def normalize_pins(raw_dir: Path, user_map: dict[str, dict]) -> list[Document]:
    """Normalize pinned Slack messages."""
    records = _read_jsonl(raw_dir / "pins.jsonl")
    docs = []

    for rec in records:
        text = clean_text(rec.get("text", ""), user_map)
        if len(text) < 20:
            continue

        channel = rec.get("channel_name", "")
        ts = rec.get("ts", "0")

        docs.append(Document(
            id=f"pin:{channel}:{ts}",
            source="pin",
            content=text,
            title=f"Pinned in #{channel}: {text[:60]}",
            metadata={"channel_name": channel},
        ))

    log.info("Normalized %d pins", len(docs))
    return docs


def normalize_code(raw_dir: Path) -> list[Document]:
    """Normalize source code file records."""
    records = _read_jsonl(raw_dir / "code.jsonl")
    docs = []

    for rec in records:
        content = rec.get("content", "")
        if len(content) < 100:
            continue

        repo = rec.get("repo", "")
        filepath = rec.get("path", "")
        language = rec.get("language", "")

        # Prefix content with repo/path for context
        prefixed = f"# {repo}/{filepath}\n\n{content}"

        doc_id = f"code:{repo}:{filepath}"
        docs.append(Document(
            id=doc_id,
            source="code",
            content=prefixed,
            title=f"{repo}/{filepath}",
            metadata={
                "repo": repo,
                "path": filepath,
                "language": language,
                "size_bytes": rec.get("size_bytes", 0),
            },
        ))

    log.info("Normalized %d code files", len(docs))
    return docs


def normalize_bitbucket(raw_dir: Path) -> list[Document]:
    """Normalize Bitbucket PR records."""
    records = _read_jsonl(raw_dir / "bitbucket.jsonl")
    docs = []

    for rec in records:
        repo = rec.get("repo", "")
        pr_id = rec.get("pr_id", 0)
        title = rec.get("title", "")
        description = rec.get("description", "").strip()
        author = rec.get("author", "")
        state = rec.get("state", "")
        source_branch = rec.get("source_branch", "")
        target_branch = rec.get("target_branch", "")
        comments = rec.get("comments", [])
        approvals = rec.get("approvals", [])

        # Build content: PR description + all comments
        parts = []
        if description:
            parts.append(f"{author}: {description}")
        for c in comments:
            c_author = c.get("author", "")
            c_content = c.get("content", "")
            c_path = c.get("path")
            if not c_content:
                continue
            prefix = f"{c_author} on {c_path}" if c_path else c_author
            parts.append(f"{prefix}: {c_content}")

        content = "\n".join(parts)
        if len(content) < 20:
            continue

        doc_title = f"{repo} PR#{pr_id}: {title}"
        docs.append(Document(
            id=f"bitbucket:{repo}:{pr_id}",
            source="bitbucket",
            content=content,
            title=doc_title[:120],
            metadata={
                "repo": repo,
                "state": state,
                "source_branch": source_branch,
                "target_branch": target_branch,
                "comment_count": len(comments),
                "approvals": ", ".join(approvals),
            },
        ))

    log.info("Normalized %d Bitbucket PRs", len(docs))
    return docs


def normalize_all(raw_dir: Path) -> list[Document]:
    """Normalize all sources. Returns deduplicated list of Documents."""
    user_map = load_users(raw_dir / "users.json")
    channel_map = load_channels(raw_dir / "channels.json")

    docs: list[Document] = []
    docs.extend(normalize_messages(raw_dir, user_map, channel_map))
    docs.extend(normalize_atlassian(raw_dir))
    docs.extend(normalize_gdrive(raw_dir))
    docs.extend(normalize_git(raw_dir))
    docs.extend(normalize_files(raw_dir))
    docs.extend(normalize_bookmarks(raw_dir))
    docs.extend(normalize_pins(raw_dir, user_map))
    docs.extend(normalize_bitbucket(raw_dir))
    docs.extend(normalize_code(raw_dir))

    # Deduplicate by ID
    seen: set[str] = set()
    unique = []
    for doc in docs:
        if doc.id not in seen:
            seen.add(doc.id)
            unique.append(doc)

    log.info("Total: %d documents (%d unique, %d duplicates removed)", len(docs), len(unique), len(docs) - len(unique))
    return unique
