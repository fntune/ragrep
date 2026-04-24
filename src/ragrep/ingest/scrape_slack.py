"""Slack workspace scraper with async concurrency and adaptive rate limiting.

Adapted from slack-finetune/src/scraper.py for the ragrep ingest pipeline.
Interface: scrape(raw_dir, config) matching the other scrapers in src/ingest/.
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

log = logging.getLogger(__name__)

# File types that are Slack-hosted links (no downloadable content)
_SKIP_FILETYPES = {"gdoc", "gsheet", "gpres", "gform", "gslides", "gdraw"}
# Video types — too large, no text content
_SKIP_FILETYPES |= {"mp4", "mov", "webm", "avi", "mkv", "flv"}


class TokenBucketLimiter:
    """Token bucket rate limiter that adapts to Slack's actual 429 responses.

    Starts fast (INITIAL_RPS), backs off on 429s, ramps up when successful.
    """

    INITIAL_RPS = 8.0  # requests per second
    MIN_RPS = 1.0
    MAX_RPS = 20.0
    RAMP_UP_FACTOR = 1.05  # multiply after N successes
    BACK_OFF_FACTOR = 0.4  # multiply on 429
    RAMP_UP_INTERVAL = 30  # successes before ramping up

    def __init__(self) -> None:
        self._rps = self.INITIAL_RPS
        self._lock = asyncio.Lock()
        self._last_call = 0.0
        self._success_count = 0
        self._total_calls = 0
        self._rate_limited = 0

    @property
    def rps(self) -> float:
        return round(self._rps, 1)

    @property
    def stats(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "rate_limited": self._rate_limited,
            "final_rps": self.rps,
        }

    async def acquire(self) -> None:
        """Wait until we can make the next API call."""
        async with self._lock:
            now = time.monotonic()
            min_interval = 1.0 / self._rps
            elapsed = now - self._last_call
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
            self._last_call = time.monotonic()

    async def on_success(self) -> None:
        self._total_calls += 1
        self._success_count += 1
        if self._success_count >= self.RAMP_UP_INTERVAL:
            self._success_count = 0
            self._rps = min(self.MAX_RPS, self._rps * self.RAMP_UP_FACTOR)

    async def on_rate_limited(self, retry_after: float) -> None:
        self._total_calls += 1
        self._rate_limited += 1
        self._success_count = 0
        old = self._rps
        self._rps = max(self.MIN_RPS, self._rps * self.BACK_OFF_FACTOR)
        log.warning("429 rate limited (Retry-After=%.0fs), rps %.1f -> %.1f", retry_after, old, self._rps)
        await asyncio.sleep(retry_after)


_limiter = TokenBucketLimiter()

# Concurrency semaphore — limits how many API calls are in-flight at once
_concurrency = asyncio.Semaphore(15)


async def _api_call(func, **kwargs) -> dict:  # type: ignore[no-untyped-def]
    """Call Slack async API with token bucket rate limiting."""
    for attempt in range(1, 6):
        async with _concurrency:
            await _limiter.acquire()
            try:
                result = await func(**kwargs)
                await _limiter.on_success()
                return result
            except SlackApiError as e:
                if e.response.status_code == 429:
                    retry_after = float(e.response.headers.get("Retry-After", attempt * 2))
                else:
                    raise
        # Rate limited — sleep OUTSIDE the semaphore
        await _limiter.on_rate_limited(retry_after)
    raise RuntimeError("Exceeded retries for API call")


@dataclass
class ScrapeState:
    """Tracks last scraped timestamp per channel for incremental scraping."""

    channels: dict[str, dict] = field(default_factory=dict)
    last_run: str = ""

    @classmethod
    def load(cls, path: Path) -> "ScrapeState":
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            return cls(
                channels=data.get("channels", {}),
                last_run=data.get("last_run", ""),
            )
        return cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "last_run": datetime.now(UTC).isoformat(),
            "channels": self.channels,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


async def fetch_users(client: AsyncWebClient) -> dict[str, dict]:
    """Fetch all users. Returns {user_id: {name, title, department, is_bot}}."""
    user_map: dict[str, dict] = {}
    cursor = None

    while True:
        kwargs: dict = {"limit": 200}
        if cursor:
            kwargs["cursor"] = cursor

        resp = await _api_call(client.users_list, **kwargs)
        for member in resp["members"]:
            profile = member.get("profile", {})
            name = (
                profile.get("display_name")
                or profile.get("real_name")
                or member.get("name", member["id"])
            )
            user_map[member["id"]] = {
                "name": name,
                "title": profile.get("title", ""),
                "department": profile.get("fields", {}).get("department", {}).get("value", "") if profile.get("fields") else "",
                "is_bot": member.get("is_bot", False),
            }

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    log.info("Fetched %s users", str(len(user_map)))
    return user_map


async def fetch_channels(
    client: AsyncWebClient,
    include_private: bool,
    channel_allowlist: list[str],
    channel_denylist: list[str],
) -> list[dict]:
    """Fetch channels with metadata, applying allowlist/denylist."""
    channels: list[dict] = []
    cursor = None

    types = "public_channel"
    if include_private:
        types = "public_channel,private_channel"

    while True:
        kwargs: dict = {"types": types, "exclude_archived": True, "limit": 200}
        if cursor:
            kwargs["cursor"] = cursor

        try:
            resp = await _api_call(client.conversations_list, **kwargs)
        except SlackApiError as e:
            if "missing_scope" in str(e) and include_private:
                log.warning("Missing groups:read scope, falling back to public channels only")
                return await fetch_channels(
                    client,
                    include_private=False,
                    channel_allowlist=channel_allowlist,
                    channel_denylist=channel_denylist,
                )
            raise

        for ch in resp["channels"]:
            name = ch["name"]
            if channel_allowlist and name not in channel_allowlist:
                continue
            if name in channel_denylist:
                continue
            channels.append({
                "id": ch["id"],
                "name": name,
                "topic": ch.get("topic", {}).get("value", ""),
                "purpose": ch.get("purpose", {}).get("value", ""),
                "is_private": ch.get("is_private", False),
                "num_members": ch.get("num_members", 0),
            })

        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    log.info("Found %d channels after filtering (%d private)", len(channels), sum(1 for c in channels if c["is_private"]))
    return channels


async def _fetch_all_pages(client: AsyncWebClient, method, base_kwargs: dict, result_key: str = "messages") -> list[dict]:
    """Generic cursor-paginated fetcher."""
    all_items: list[dict] = []
    cursor = None
    while True:
        kwargs = {**base_kwargs}
        if cursor:
            kwargs["cursor"] = cursor
        resp = await _api_call(method, **kwargs)
        all_items.extend(resp.get(result_key, []))
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break
    return all_items


async def fetch_pins(client: AsyncWebClient, channel_id: str) -> list[dict]:
    """Fetch pinned messages for a channel."""
    try:
        resp = await _api_call(client.pins_list, channel=channel_id)
        return resp.get("items", [])
    except SlackApiError as e:
        if "missing_scope" in str(e):
            return []
        log.warning("Failed to fetch pins for %s: %s", channel_id, e.response["error"])
        return []


async def fetch_bookmarks(client: AsyncWebClient, channel_id: str) -> list[dict]:
    """Fetch bookmarks for a channel."""
    try:
        resp = await _api_call(client.bookmarks_list, channel_id=channel_id)
        return resp.get("bookmarks", [])
    except SlackApiError as e:
        if "missing_scope" in str(e):
            return []
        log.warning("Failed to fetch bookmarks for %s: %s", channel_id, e.response["error"])
        return []


async def download_file(
    session: aiohttp.ClientSession,
    file_info: dict,
    files_dir: Path,
    max_size: int,
    download_sem: asyncio.Semaphore,
) -> dict | None:
    """Download a Slack file to disk. Returns metadata dict or None on skip/failure."""
    filetype = file_info.get("filetype", "")
    if filetype in _SKIP_FILETYPES:
        return None
    size = file_info.get("size", 0)
    if size > max_size:
        return None
    url = file_info.get("url_private_download") or file_info.get("url_private")
    if not url:
        return None
    file_id = file_info.get("id", "")
    if not file_id:
        return None

    ext = filetype or "bin"
    local_name = f"{file_id}.{ext}"
    local_path = files_dir / local_name

    # Skip if already downloaded (incremental)
    if local_path.exists():
        return {"file_id": file_id, "local_path": f"files/{local_name}"}

    try:
        async with download_sem:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    log.debug("HTTP %d downloading %s", resp.status, file_info.get("name", "?"))
                    return None
                data = await resp.read()
    except Exception as e:
        log.debug("Failed to download file %s: %s", file_info.get("name", "?"), e)
        return None

    local_path.write_bytes(data)
    return {"file_id": file_id, "local_path": f"files/{local_name}"}


def _extract_reactions(msg: dict) -> list[dict]:
    reactions = msg.get("reactions", [])
    return [{"name": r["name"], "count": r["count"]} for r in reactions]


def _extract_files(msg: dict) -> list[dict]:
    files = msg.get("files", [])
    return [{
        "id": f.get("id", ""),
        "name": f.get("name", ""),
        "filetype": f.get("filetype", ""),
        "mimetype": f.get("mimetype", ""),
        "size": f.get("size", 0),
        "url_private_download": f.get("url_private_download", ""),
    } for f in files]


def _extract_links(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"<(https?://[^|>]+)(?:\|[^>]*)?>", text)]


def _message_to_record(msg: dict, channel_id: str, channel_name: str, is_reply: bool = False) -> dict:
    text = msg.get("text", "")
    return {
        "channel_id": channel_id,
        "channel_name": channel_name,
        "user_id": msg.get("user", ""),
        "ts": msg.get("ts", ""),
        "thread_ts": msg.get("thread_ts", ""),
        "text": text,
        "is_reply": is_reply,
        "reply_count": msg.get("reply_count", 0),
        "subtype": msg.get("subtype", ""),
        "reactions": _extract_reactions(msg),
        "reaction_count": sum(r["count"] for r in msg.get("reactions", [])),
        "files": _extract_files(msg),
        "links": _extract_links(text),
    }


@dataclass
class ChannelResult:
    channel_id: str
    channel_name: str
    messages: list[dict] = field(default_factory=list)
    pins: list[dict] = field(default_factory=list)
    bookmarks: list[dict] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    max_ts: float = 0.0
    msg_count: int = 0
    thread_count: int = 0


async def _download_msg_files(
    msg: dict,
    session: aiohttp.ClientSession,
    fetch_files: bool,
    max_file_size: int,
    files_dir: Path,
    download_sem: asyncio.Semaphore,
    ch_id: str,
    ch_name: str,
) -> list[dict]:
    """Download files from a message, return metadata records."""
    if not fetch_files or not msg.get("files"):
        return []
    records = []
    for file_info in msg["files"]:
        dl = await download_file(session, file_info, files_dir, max_file_size, download_sem)
        if dl:
            records.append({
                "channel_id": ch_id, "channel_name": ch_name,
                "user_id": msg.get("user", ""), "ts": msg.get("ts", ""),
                "file_id": file_info.get("id", ""),
                "file_name": file_info.get("name", ""),
                "file_type": file_info.get("filetype", ""),
                "mime_type": file_info.get("mimetype", ""),
                "size": file_info.get("size", 0),
                "local_path": dl["local_path"],
            })
    return records


# Alias module-level functions to avoid name collision with bool params in scrape_channel
_fetch_pins = fetch_pins
_fetch_bookmarks = fetch_bookmarks


async def scrape_channel(
    client: AsyncWebClient,
    ch: dict,
    oldest: float,
    session: aiohttp.ClientSession,
    files_dir: Path,
    download_sem: asyncio.Semaphore,
    *,
    include_bots: bool,
    fetch_pins: bool,
    fetch_bookmarks: bool,
    fetch_files: bool,
    max_file_size: int,
    page_size: int,
) -> ChannelResult:
    """Scrape a single channel: history + pins + bookmarks concurrently, then threads."""
    ch_id, ch_name = ch["id"], ch["name"]
    result = ChannelResult(channel_id=ch_id, channel_name=ch_name, max_ts=oldest)

    # Fire history, pins, bookmarks concurrently
    coros = [
        _fetch_all_pages(client, client.conversations_history, {"channel": ch_id, "limit": page_size, **({"oldest": str(oldest)} if oldest else {})}),
    ]
    if fetch_pins:
        coros.append(_fetch_pins(client, ch_id))
    if fetch_bookmarks:
        coros.append(_fetch_bookmarks(client, ch_id))

    gathered = await asyncio.gather(*coros, return_exceptions=True)

    # Unpack
    idx = 0
    raw_messages = gathered[idx] if not isinstance(gathered[idx], BaseException) else []
    idx += 1

    if fetch_pins:
        raw_pins = gathered[idx] if not isinstance(gathered[idx], BaseException) else []
        idx += 1
        for pin in raw_pins:
            pin_msg = pin.get("message", {})
            if pin_msg:
                result.pins.append({
                    "channel_id": ch_id, "channel_name": ch_name,
                    "user_id": pin_msg.get("user", ""), "ts": pin_msg.get("ts", ""),
                    "text": pin_msg.get("text", ""), "pinned_by": pin.get("created_by", ""),
                })

    if fetch_bookmarks:
        raw_bookmarks = gathered[idx] if not isinstance(gathered[idx], BaseException) else []
        for bm in raw_bookmarks:
            result.bookmarks.append({
                "channel_id": ch_id, "channel_name": ch_name,
                "title": bm.get("title", ""), "link": bm.get("link", ""),
                "emoji": bm.get("emoji", ""), "created_by": bm.get("created_by", ""),
            })

    # Process messages, collect thread tasks
    thread_coros = []
    for msg in raw_messages:
        subtype = msg.get("subtype", "")
        if subtype in ("channel_join", "channel_leave", "channel_topic", "channel_purpose", "channel_name"):
            continue
        if not include_bots and (msg.get("bot_id") or subtype == "bot_message"):
            continue

        result.messages.append(_message_to_record(msg, ch_id, ch_name))
        result.msg_count += 1
        ts = float(msg.get("ts", 0))
        result.max_ts = max(result.max_ts, ts)

        # File downloads
        file_records = await _download_msg_files(msg, session, fetch_files, max_file_size, files_dir, download_sem, ch_id, ch_name)
        result.files.extend(file_records)

        # Queue thread fetch
        if msg.get("reply_count", 0) > 0 and msg.get("ts"):
            result.thread_count += 1
            thread_coros.append(
                _fetch_all_pages(client, client.conversations_replies, {"channel": ch_id, "ts": msg["ts"], "limit": 200})
            )

    # Fetch all threads concurrently
    if thread_coros:
        thread_results = await asyncio.gather(*thread_coros, return_exceptions=True)
        for replies in thread_results:
            if isinstance(replies, BaseException):
                continue
            for reply in replies[1:]:  # skip parent
                if not include_bots and (reply.get("bot_id") or reply.get("subtype") == "bot_message"):
                    continue
                result.messages.append(_message_to_record(reply, ch_id, ch_name, is_reply=True))
                result.msg_count += 1

                # File downloads from thread replies
                file_records = await _download_msg_files(reply, session, fetch_files, max_file_size, files_dir, download_sem, ch_id, ch_name)
                result.files.extend(file_records)

    return result


async def scrape_async(raw_dir: Path, config: dict) -> None:
    """Main async scrape orchestrator."""
    global _limiter, _concurrency
    _limiter = TokenBucketLimiter()
    _concurrency = asyncio.Semaphore(15)

    token = os.environ.get("SLACK_TOKEN", "")
    if not token:
        raise ValueError("SLACK_TOKEN environment variable is required")

    # Read config
    date_cutoff = config.get("date_cutoff", "")
    include_bots = config.get("include_bots", False)
    include_private = config.get("include_private", False)
    do_fetch_pins = config.get("fetch_pins", True)
    do_fetch_bookmarks = config.get("fetch_bookmarks", True)
    do_fetch_files = config.get("fetch_files", True)
    max_file_size = config.get("max_file_size", 50_000_000)
    max_messages = config.get("max_messages", 0)
    page_size = config.get("page_size", 200)
    channel_allowlist = config.get("channel_allowlist", [])
    channel_denylist = config.get("channel_denylist", [])

    raw_dir.mkdir(parents=True, exist_ok=True)
    files_dir = raw_dir / "files"
    files_dir.mkdir(exist_ok=True)

    state_path = raw_dir / "state.json"
    messages_path = raw_dir / "messages.jsonl"
    users_path = raw_dir / "users.json"
    channels_path = raw_dir / "channels.json"
    pins_path = raw_dir / "pins.jsonl"
    bookmarks_path = raw_dir / "bookmarks.jsonl"
    files_path = raw_dir / "files.jsonl"

    state = ScrapeState.load(state_path)
    client = AsyncWebClient(token=token)

    # Fetch users and channels (sequential setup)
    user_map = await fetch_users(client)
    with open(users_path, "w") as f:
        json.dump(user_map, f, indent=2)

    channels = await fetch_channels(client, include_private, channel_allowlist, channel_denylist)
    with open(channels_path, "w") as f:
        json.dump(channels, f, indent=2)

    # Compute date cutoff
    oldest_ts = 0.0
    if date_cutoff:
        oldest_ts = datetime.fromisoformat(date_cutoff).replace(tzinfo=UTC).timestamp()

    # Build channel args
    channel_args: list[tuple[dict, float]] = []
    for ch in channels:
        ch_id = ch["id"]
        ch_oldest = oldest_ts
        ch_state = state.channels.get(ch_id, {})
        if ch_state.get("last_ts"):
            ch_oldest = max(ch_oldest, float(ch_state["last_ts"]))
        channel_args.append((ch, ch_oldest))

    start_time = time.monotonic()
    totals = {"messages": 0, "threads": 0, "pins": 0, "bookmarks": 0, "files": 0}
    completed = 0

    is_incremental = state_path.exists() and state.channels
    mode = "a" if is_incremental else "w"

    # Concurrency controls
    channel_sem = asyncio.Semaphore(20)
    download_sem = asyncio.Semaphore(10)  # file downloads hit CDN, separate from API

    async def bounded_scrape(
        ch: dict, ch_oldest: float, http_session: aiohttp.ClientSession
    ) -> ChannelResult:
        async with channel_sem:
            return await scrape_channel(
                client, ch, ch_oldest, http_session, files_dir, download_sem,
                include_bots=include_bots,
                fetch_pins=do_fetch_pins,
                fetch_bookmarks=do_fetch_bookmarks,
                fetch_files=do_fetch_files,
                max_file_size=max_file_size,
                page_size=page_size,
            )

    headers = {"Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession(headers=headers) as http_session:
        with (
            open(messages_path, mode) as msg_out,
            open(pins_path, mode) as pins_out,
            open(bookmarks_path, mode) as bm_out,
            open(files_path, mode) as files_out,
        ):
            # Fire ALL channels as tasks, bounded by channel_sem
            tasks = [bounded_scrape(ch, ch_oldest, http_session) for ch, ch_oldest in channel_args]

            for coro in asyncio.as_completed(tasks):
                try:
                    res: ChannelResult = await coro
                except Exception as e:
                    log.warning("Channel failed: %s", e)
                    completed += 1
                    continue

                # Write results
                for record in res.messages:
                    msg_out.write(json.dumps(record) + "\n")
                for pin in res.pins:
                    pins_out.write(json.dumps(pin) + "\n")
                for bm in res.bookmarks:
                    bm_out.write(json.dumps(bm) + "\n")
                for file_rec in res.files:
                    files_out.write(json.dumps(file_rec) + "\n")

                totals["messages"] += res.msg_count
                totals["threads"] += res.thread_count
                totals["pins"] += len(res.pins)
                totals["bookmarks"] += len(res.bookmarks)
                totals["files"] += len(res.files)

                # Update state
                old_count = state.channels.get(res.channel_id, {}).get("message_count", 0)
                state.channels[res.channel_id] = {
                    "name": res.channel_name,
                    "last_ts": str(res.max_ts),
                    "message_count": old_count + res.msg_count,
                }

                completed += 1

                # Log progress every 50 channels
                if completed % 50 == 0 or completed == len(channels):
                    elapsed = time.monotonic() - start_time
                    log.info(
                        "[%d/%d] %s msgs, %d threads, %d pins, %d bm, %d files (%.0fs, %.1f rps)",
                        completed, len(channels), str(totals["messages"]),
                        totals["threads"], totals["pins"], totals["bookmarks"], totals["files"],
                        elapsed, _limiter.rps,
                    )

                if max_messages and totals["messages"] >= max_messages:
                    log.info("Reached max_messages limit (%d)", max_messages)
                    break

    state.save(state_path)
    elapsed = time.monotonic() - start_time
    log.info(
        "Scrape complete: %s messages, %d channels, %s threads, %d pins, %d bookmarks, %d files (%.0fs)",
        str(totals["messages"]), len(channels), str(totals["threads"]),
        totals["pins"], totals["bookmarks"], totals["files"], elapsed,
    )
    log.info("Rate limiter stats: %s", _limiter.stats)


def scrape(raw_dir: Path, config: dict) -> None:
    """Scrape a Slack workspace. Writes messages.jsonl, users.json, channels.json, etc."""
    asyncio.run(scrape_async(raw_dir, config))


async def resolve_users_async(raw_dir: Path, config: dict) -> None:
    """Re-fetch user list and resolve any IDs found in messages but missing from users.json."""
    global _limiter, _concurrency
    _limiter = TokenBucketLimiter()
    _concurrency = asyncio.Semaphore(15)

    token = os.environ.get("SLACK_TOKEN", "")
    if not token:
        raise ValueError("SLACK_TOKEN environment variable is required")

    users_path = raw_dir / "users.json"
    messages_path = raw_dir / "messages.jsonl"

    client = AsyncWebClient(token=token)

    # Step 1: Re-fetch full user list (including deleted)
    user_map = await fetch_users(client)
    log.info("Fetched %d users from users_list", len(user_map))

    # Step 2: Find IDs referenced in messages but not in user_map
    missing: set[str] = set()
    mention_re = re.compile(r"<@(\w+)>")
    with open(messages_path) as f:
        for line in f:
            if not line.strip():
                continue
            msg = json.loads(line)
            uid = msg.get("user_id", "")
            if uid and uid not in user_map:
                missing.add(uid)
            for m in mention_re.findall(msg.get("text", "")):
                if m not in user_map:
                    missing.add(m)

    if not missing:
        log.info("All user IDs already resolved")
    else:
        log.info("Resolving %d missing user IDs via users.info...", len(missing))
        resolved = 0
        failed = 0
        for uid in missing:
            try:
                resp = await _api_call(client.users_info, user=uid)
                member = resp["user"]
                profile = member.get("profile", {})
                name = (
                    profile.get("display_name")
                    or profile.get("real_name")
                    or member.get("name", uid)
                )
                user_map[uid] = {
                    "name": name,
                    "title": profile.get("title", ""),
                    "department": "",
                    "is_bot": member.get("is_bot", False),
                }
                resolved += 1
            except Exception as e:
                log.debug("Failed to resolve %s: %s", uid, e)
                failed += 1
        log.info("Resolved %d, failed %d", resolved, failed)

    with open(users_path, "w") as f:
        json.dump(user_map, f, indent=2)
    log.info("Wrote %d users to %s", len(user_map), users_path)


def resolve_users(raw_dir: Path, config: dict) -> None:
    """Re-fetch missing users from Slack and update users.json."""
    asyncio.run(resolve_users_async(raw_dir, config))


async def download_files_async(raw_dir: Path, config: dict) -> None:
    """Download files from already-scraped messages in messages.jsonl."""
    token = os.environ.get("SLACK_TOKEN", "")
    if not token:
        raise ValueError("SLACK_TOKEN environment variable is required")

    max_file_size = config.get("max_file_size", 50_000_000)

    messages_path = raw_dir / "messages.jsonl"
    files_dir = raw_dir / "files"
    files_dir.mkdir(exist_ok=True)
    files_path = raw_dir / "files.jsonl"

    if not messages_path.exists():
        log.error("No messages.jsonl found -- run scrape first")
        return

    # Collect all file_info dicts with message context
    file_tasks: list[tuple[dict, dict]] = []  # (file_info, msg_context)
    seen_ids: set[str] = set()

    with open(messages_path) as f:
        for line in f:
            if not line.strip():
                continue
            msg = json.loads(line)
            for file_info in msg.get("files", []):
                fid = file_info.get("id", "")
                if not fid or fid in seen_ids:
                    continue
                seen_ids.add(fid)
                file_tasks.append((file_info, {
                    "channel_id": msg.get("channel_id", ""),
                    "channel_name": msg.get("channel_name", ""),
                    "user_id": msg.get("user_id", ""),
                    "ts": msg.get("ts", ""),
                }))

    log.info("Found %d unique files in messages.jsonl", len(file_tasks))

    download_sem = asyncio.Semaphore(10)
    results: list[dict | None] = []
    start = time.monotonic()

    headers = {"Authorization": f"Bearer {token}"}
    async with aiohttp.ClientSession(headers=headers) as session:

        async def _download_one(file_info: dict, ctx: dict) -> dict | None:
            dl = await download_file(session, file_info, files_dir, max_file_size, download_sem)
            if not dl:
                return None
            return {
                **ctx,
                "file_id": file_info.get("id", ""),
                "file_name": file_info.get("name", ""),
                "file_type": file_info.get("filetype", ""),
                "mime_type": file_info.get("mimetype", ""),
                "size": file_info.get("size", 0),
                "local_path": dl["local_path"],
            }

        # Fire all downloads concurrently (bounded by download_sem)
        coros = [_download_one(fi, ctx) for fi, ctx in file_tasks]
        for i, coro in enumerate(asyncio.as_completed(coros)):
            try:
                result = await coro
                results.append(result)
            except Exception as e:
                log.debug("File download error: %s", e)
                results.append(None)
            if (i + 1) % 100 == 0:
                elapsed = time.monotonic() - start
                done = sum(1 for r in results if r is not None)
                log.info("[%d/%d] downloaded=%d (%.0fs)", i + 1, len(file_tasks), done, elapsed)

    # Write all records
    with open(files_path, "w") as out:
        for rec in results:
            if rec:
                out.write(json.dumps(rec) + "\n")

    downloaded = sum(1 for r in results if r is not None)
    elapsed = time.monotonic() - start
    log.info("File download complete: %d downloaded, %d skipped (%.0fs)", downloaded, len(file_tasks) - downloaded, elapsed)


def download_files(raw_dir: Path, config: dict) -> None:
    """Download files from already-scraped messages."""
    asyncio.run(download_files_async(raw_dir, config))
