"""从本地 unified_*.jsonl 聚合某用户的原始微博（正文 + 外链），供「来源」详情页使用。"""

from __future__ import annotations

import glob
import json
import os
from typing import Any


def _unified_jsonl_paths(output_dir: str) -> list[str]:
    paths = glob.glob(os.path.join(output_dir, "unified_*.jsonl"))
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths


def load_user_weibo_posts(
    user_id: str,
    output_dir: str,
    *,
    limit: int = 800,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    扫描 output 下全部 unified_*.jsonl，按 item_id 去重，按时间倒序。
    返回 (posts, files_that_contained_rows)；posts 每项含 text, url, created_at, source_type, item_id。
    """
    uid = str(user_id).strip()
    if not uid:
        return [], []

    paths = _unified_jsonl_paths(output_dir)
    if not paths:
        return [], []

    seen_ids: set[str] = set()
    posts: list[dict[str, Any]] = []
    files_hit: list[str] = []

    for path in paths:
        local_found = 0
        with open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_uid = str(row.get("user_id", "")).strip()
                if row_uid != uid:
                    continue
                raw = row.get("raw") or {}
                item_id = str(
                    raw.get("_id") or raw.get("id") or row.get("item_id") or ""
                ).strip()
                if item_id and item_id in seen_ids:
                    continue
                if item_id:
                    seen_ids.add(item_id)

                text = str(raw.get("content") or row.get("text") or "").strip()
                url = str(raw.get("url") or "").strip()
                if not text and not url:
                    continue

                created_at = str(
                    raw.get("created_at") or row.get("created_at") or ""
                ).strip()
                source_type = str(row.get("source_type") or "").strip()
                ip_loc = str(raw.get("ip_location") or "").strip()
                src_client = str(raw.get("source") or "").strip()

                posts.append(
                    {
                        "text": text,
                        "url": url,
                        "created_at": created_at,
                        "source_type": source_type,
                        "item_id": item_id,
                        "ip_location": ip_loc,
                        "weibo_source": src_client,
                    }
                )
                local_found += 1

        if local_found:
            files_hit.append(os.path.basename(path))

    posts.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    if limit and len(posts) > limit:
        posts = posts[:limit]

    return posts, files_hit
