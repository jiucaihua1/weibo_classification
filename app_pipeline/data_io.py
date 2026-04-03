"""Load training/infer records from unified jsonl or crawl bundle JSON."""
import json
from typing import Any, Dict, Iterator, List


def load_unified_jsonl_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(data)
    return records


def iter_unified_jsonl_records(path: str) -> Iterator[Dict[str, Any]]:
    """Stream unified jsonl lines (one JSON object per line)."""
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                yield data


def load_crawl_bundle_records(path: str) -> List[Dict[str, Any]]:
    with open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    users = data.get("users") if isinstance(data, dict) else data
    if not isinstance(users, list):
        raise ValueError("crawl bundle must contain a 'users' array")
    rows: List[Dict[str, Any]] = []
    for block in users:
        for r in block.get("records") or []:
            rows.append(r)
    return rows


def load_training_records(path: str) -> List[Dict[str, Any]]:
    """Records with user_id + text for TF-IDF (same semantics as former load_unified_records)."""
    if path.lower().endswith(".jsonl"):
        out = []
        for data in load_unified_jsonl_records(path):
            if data.get("user_id") and data.get("text"):
                out.append(data)
        return out
    if path.lower().endswith(".json"):
        out = []
        for data in load_crawl_bundle_records(path):
            if data.get("user_id") and data.get("text"):
                out.append(data)
        return out
    raise ValueError(f"Unsupported input format: {path}")


def load_infer_records(path: str) -> List[Dict[str, Any]]:
    """All rows with user_id (infer counts source_type even without text)."""
    if path.lower().endswith(".jsonl"):
        rows = []
        for data in load_unified_jsonl_records(path):
            if data.get("user_id"):
                rows.append(data)
        return rows
    if path.lower().endswith(".json"):
        rows = []
        for data in load_crawl_bundle_records(path):
            if data.get("user_id"):
                rows.append(data)
        return rows
    raise ValueError(f"Unsupported input format: {path}")


def iter_crawl_bundle_users(path: str) -> Iterator[Dict[str, Any]]:
    """
    Stream `users` entries from a crawl bundle JSON without loading the whole file into memory.
    Requires `ijson` (pip install ijson). Falls back to full json.load if ijson is missing.
    """
    try:
        import ijson  # type: ignore
    except ImportError:
        with open(path, "rt", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        users = data.get("users") if isinstance(data, dict) else data
        if not isinstance(users, list):
            raise ValueError("crawl bundle must contain a 'users' array")
        for block in users:
            yield block
        return

    with open(path, "rb") as f:
        for item in ijson.items(f, "users.item"):
            if isinstance(item, dict):
                yield item
