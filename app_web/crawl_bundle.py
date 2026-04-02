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
    """只写 weibo_crawl_latest.json，避免每次抓取再堆一份时间戳副本。"""
    del job_id  # 保留签名兼容调用方
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    latest = Path(output_dir) / "weibo_crawl_latest.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(latest), ""


def cleanup_post_crawl_artifacts(output_dir: str) -> dict[str, int]:
    """抓取合并完成后删除：unified_*.jsonl、user_aggregate_*.json、以及非 latest 的 weibo_crawl_*.json。"""
    out = Path(output_dir)
    counts = {"unified": 0, "user_aggregate": 0, "weibo_crawl_extra": 0}
    if not out.is_dir():
        return counts
    for p in list(out.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        try:
            if name.startswith("unified_") and name.endswith(".jsonl"):
                p.unlink()
                counts["unified"] += 1
            elif name.startswith("user_aggregate_") and name.endswith(".json"):
                p.unlink()
                counts["user_aggregate"] += 1
            elif name.startswith("weibo_crawl_") and name.endswith(".json") and name != "weibo_crawl_latest.json":
                p.unlink()
                counts["weibo_crawl_extra"] += 1
        except OSError:
            pass
    return counts
