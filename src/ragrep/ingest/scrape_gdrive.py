"""Scrape Google Drive files (parallelized, resumable)."""

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

log = logging.getLogger(__name__)

# Google Workspace export mime types -> plain text
_EXPORT_MIMES: dict[str, str] = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}

_FILE_TYPE_LABELS: dict[str, str] = {
    "application/vnd.google-apps.document": "Google Doc",
    "application/vnd.google-apps.spreadsheet": "Google Sheet",
    "application/vnd.google-apps.presentation": "Google Slides",
}

# Binary files we can download and read
_DOWNLOAD_MIMES: dict[str, str] = {
    "text/plain": "text",
    "text/csv": "csv",
    "text/markdown": "markdown",
    "text/html": "html",
    "application/json": "json",
}

_DRIVE_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
_WORKERS = 20


def _get_credentials():
    """Get Google credentials with Drive scope."""
    from google.auth import default
    from google.auth.exceptions import DefaultCredentialsError
    from google.auth.transport.requests import Request

    try:
        creds, _ = default(scopes=[_DRIVE_SCOPE])
        creds.refresh(Request())
        return creds
    except DefaultCredentialsError:
        raise SystemExit(
            "No Google credentials found. Run:\n"
            "  gcloud auth application-default login "
            f"--scopes={_DRIVE_SCOPE},https://www.googleapis.com/auth/cloud-platform"
        )
    except Exception as e:
        if "scope" in str(e).lower() or "unauthorized" in str(e).lower() or "403" in str(e):
            raise SystemExit(
                "Current credentials lack Drive scope. Re-authenticate:\n"
                "  gcloud auth application-default login "
                f"--scopes={_DRIVE_SCOPE},https://www.googleapis.com/auth/cloud-platform"
            )
        raise


def _build_service(creds):
    """Build Drive API service (NOT thread-safe -- create one per thread)."""
    from googleapiclient.discovery import build
    return build("drive", "v3", credentials=creds, cache_discovery=False)


_thread_local = threading.local()
_creds_ref = None


def _get_thread_service():
    if not hasattr(_thread_local, "service"):
        _thread_local.service = _build_service(_creds_ref)
    return _thread_local.service


def _list_all_files(service) -> list[dict]:
    """List all files matching query, paginating through results."""
    files = []
    page_token = None

    mime_clauses = [f"mimeType = '{m}'" for m in _EXPORT_MIMES]
    mime_clauses.extend(f"mimeType = '{m}'" for m in _DOWNLOAD_MIMES)
    mime_filter = " or ".join(mime_clauses)
    q = f"({mime_filter}) and trashed = false"

    while True:
        resp = service.files().list(
            q=q,
            pageSize=100,
            fields="nextPageToken, files(id, name, mimeType, size, parents, owners)",
            pageToken=page_token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            corpora="allDrives",
        ).execute()

        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

        if len(files) % 500 == 0:
            log.info("Listed %d files so far...", len(files))

    return files


def _resolve_folder_paths(files: list[dict]) -> dict[str, str]:
    """Pre-resolve all unique parent folder IDs to paths."""
    parent_ids = {f["parents"][0] for f in files if f.get("parents")}
    log.info("Resolving %d unique folder paths...", len(parent_ids))

    cache: dict[str, str] = {}

    def _resolve_one(folder_id: str) -> str:
        if folder_id in cache:
            return cache[folder_id]
        parts = []
        current = folder_id
        for _ in range(5):
            if current in cache:
                parts.append(cache[current])
                break
            try:
                svc = _get_thread_service()
                folder = svc.files().get(
                    fileId=current, fields="name, parents", supportsAllDrives=True
                ).execute()
                parts.append(folder.get("name", ""))
                folder_parents = folder.get("parents", [])
                current = folder_parents[0] if folder_parents else None
                if not current:
                    break
            except Exception:
                break
        path = " > ".join(reversed(parts))
        cache[folder_id] = path
        return path

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        futures = {pool.submit(_resolve_one, pid): pid for pid in parent_ids}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                pass

    return cache


def _fetch_file(f: dict, folder_cache: dict[str, str]) -> dict | None:
    """Fetch a single file's content. Returns flat record or None."""
    svc = _get_thread_service()
    file_id = f["id"]
    name = f.get("name", "")
    mime = f.get("mimeType", "")

    try:
        if mime in _EXPORT_MIMES:
            export_mime = _EXPORT_MIMES[mime]
            raw = svc.files().export(fileId=file_id, mimeType=export_mime).execute()
            content = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            file_type = _FILE_TYPE_LABELS.get(mime, "document")
        elif mime in _DOWNLOAD_MIMES:
            size = int(f.get("size", 0) or 0)
            if size > 1_000_000:
                return None
            raw = svc.files().get_media(fileId=file_id).execute()
            content = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
            file_type = _DOWNLOAD_MIMES[mime]
        else:
            return None
    except Exception as e:
        log.debug("Failed to fetch %s (%s): %s", name, file_id, e)
        return None

    if not content or len(content.strip()) < 50:
        return None

    # No truncation -- chunking handles long content

    folder_path = ""
    parents = f.get("parents", [])
    if parents:
        folder_path = folder_cache.get(parents[0], "")

    owner = ""
    owners = f.get("owners", [])
    if owners:
        owner = owners[0].get("displayName", "")

    return {
        "file_id": file_id,
        "name": name,
        "owner": owner,
        "content": content,
        "mime_type": mime,
        "file_type": file_type,
        "path": folder_path,
    }


def _load_existing(output_path: Path) -> set[str]:
    """Load file IDs already scraped from existing output."""
    seen: set[str] = set()
    if not output_path.exists():
        return seen
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                fid = record.get("file_id", "")
                if fid:
                    seen.add(fid)
    return seen


def scrape(raw_dir: Path, config: dict) -> None:
    """Scrape Google Drive, write to raw/gdrive.jsonl."""
    global _creds_ref

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / "gdrive.jsonl"

    creds = _get_credentials()
    _creds_ref = creds
    service = _build_service(creds)

    log.info("Listing Drive files...")
    all_files = _list_all_files(service)
    log.info("Found %d files", len(all_files))

    already_done = _load_existing(output_path)
    if already_done:
        log.info("Resuming -- %d files already scraped, %d remaining",
                 len(already_done), len(all_files) - len(already_done))
    files = [f for f in all_files if f["id"] not in already_done]

    if not files:
        log.info("All files already scraped -- nothing to do")
        return

    folder_cache = _resolve_folder_paths(files)

    log.info("Fetching %d files with %d workers...", len(files), _WORKERS)
    done = 0
    new_records = 0
    skipped = 0
    _write_lock = threading.Lock()

    with open(output_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
            futures = {
                pool.submit(_fetch_file, f, folder_cache): f
                for f in files
            }
            for fut in as_completed(futures):
                done += 1
                try:
                    result = fut.result()
                    if result:
                        with _write_lock:
                            out_f.write(json.dumps(result) + "\n")
                            out_f.flush()
                        new_records += 1
                    else:
                        skipped += 1
                except Exception:
                    skipped += 1

                if done % 200 == 0:
                    log.info("Fetched %d/%d files (%d new, %d skipped)",
                             done, len(files), new_records, skipped)

    total = len(already_done) + new_records
    log.info("Done: %d new + %d resumed = %d total records (%d skipped)",
             new_records, len(already_done), total, skipped)
