"""
将一条或多条 unified_*.jsonl 合并为 weibo_crawl_latest 风格的 bundle JSON。
同 (user_id, item_id) 去重；按 user_id 排序输出。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def merge_paths(paths: List[str]) -> Tuple[Dict[str, Any], int, int]:
    by_user: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seen: Set[Tuple[str, str]] = set()
    n_in = 0
    n_dup = 0
    for p in paths:
        fp = Path(p)
        if not fp.is_file():
            raise FileNotFoundError(str(fp))
        with fp.open("rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                n_in += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = str(row.get("user_id", "")).strip()
                if not uid:
                    continue
                iid = str(row.get("item_id", "") or "").strip()
                if not iid:
                    iid = hashlib.md5(line.encode("utf-8", errors="replace")).hexdigest()
                key = (uid, iid)
                if key in seen:
                    n_dup += 1
                    continue
                seen.add(key)
                by_user[uid].append(row)

    uids = sorted(by_user.keys())
    users = [{"user_id": uid, "records": by_user[uid]} for uid in uids]
    bundle = {
        "schema": "weibo_crawl_bundle_v1",
        "job_id": "merge_unified_jsonl",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "user_ids_order": uids,
        "users": users,
    }
    return bundle, n_in, n_dup


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge unified jsonl files into crawl bundle JSON.")
    ap.add_argument("jsonl", nargs="+", help="Paths to unified_*.jsonl")
    ap.add_argument("-o", "--output", default="output/weibo_crawl_latest.json", help="Output bundle path")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    bundle, n_in, n_dup = merge_paths([str(Path(p).resolve()) for p in args.jsonl])
    out.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    n_users = len(bundle["user_ids_order"])
    n_records = sum(len(u["records"]) for u in bundle["users"])
    print(f"[OK] lines_read={n_in} duplicates_skipped={n_dup} users={n_users} records={n_records} -> {out}")


if __name__ == "__main__":
    main()
