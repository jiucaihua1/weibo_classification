"""Load training/infer records from unified jsonl or crawl bundle JSON."""
import json
from typing import Any, Dict, List


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
