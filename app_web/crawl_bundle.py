import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def merge_unified_jsonl_by_user(paths: list[str]) -> dict[str, list[dict[str, Any]]]:
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        if not path or not Path(path).is_file():
            continue
        with open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                uid = str(row.get("user_id", "")).strip()
                if uid:
                    by_user[uid].append(row)
    return dict(by_user)


def build_crawl_bundle(
    *,
    job_id: str,
    user_ids_ordered: list[str],
    unified_paths: list[str],
) -> dict[str, Any]:
    by_user = merge_unified_jsonl_by_user(unified_paths)
    users = [{"user_id": uid, "records": by_user.get(uid, [])} for uid in user_ids_ordered]
    return {
        "schema": "weibo_crawl_bundle_v1",
        "job_id": job_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "user_ids_order": list(user_ids_ordered),
        "users": users,
    }


def write_crawl_bundle(output_dir: str, job_id: str, payload: dict[str, Any]) -> tuple[str, str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    latest = Path(output_dir) / "weibo_crawl_latest.json"
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    archived = Path(output_dir) / f"weibo_crawl_{ts}_{job_id[:8]}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    for p in (latest, archived):
        p.write_text(text, encoding="utf-8")
    return str(latest), str(archived)
