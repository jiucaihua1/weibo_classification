"""从本地 unified_*.jsonl 与/合并后的 weibo_crawl_latest.json 聚合用户原始微博（正文 + 外链）。"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Optional

from app_pipeline.data_io import iter_crawl_bundle_users


def normalize_uid(raw: Any) -> str:
    """
    画像 / URL / JSON 里 user_id 可能是 str、int 或 "123.0"；统一成可比对的数字字符串。
    """
    if raw is None or raw is True or raw is False:
        return ""
    if isinstance(raw, int):
        return str(raw)
    if isinstance(raw, float):
        if raw != raw:  # NaN
            return ""
        ir = int(round(raw))
        if abs(float(ir) - raw) < 1e-9:
            return str(ir)
        return str(raw).strip()
    s = str(raw).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def _unified_jsonl_paths(output_dir: str) -> list[str]:
    paths = glob.glob(os.path.join(output_dir, "unified_*.jsonl"))
    paths.sort(key=os.path.getmtime, reverse=True)
    return paths


def _post_from_row(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """unified 单行或 bundle 内单条 record → 展示用 dict；无正文且无链接则跳过。"""
    raw = row.get("raw") if isinstance(row.get("raw"), dict) else {}
    item_id = str(
        raw.get("_id") or raw.get("id") or row.get("item_id") or ""
    ).strip()
    text = str(raw.get("content") or row.get("text") or "").strip()
    url = str(raw.get("url") or "").strip()
    if not text and not url:
        return None
    created_at = str(raw.get("created_at") or row.get("created_at") or "").strip()
    source_type = str(row.get("source_type") or "").strip()
    ip_loc = str(raw.get("ip_location") or "").strip()
    src_client = str(raw.get("source") or "").strip()
    return {
        "text": text,
        "url": url,
        "created_at": created_at,
        "source_type": source_type,
        "item_id": item_id,
        "ip_location": ip_loc,
        "weibo_source": src_client,
    }


def _merge_bundle_posts_for_user(
    bundle_path: str,
    uid: str,
    *,
    seen_ids: set[str],
    posts: list[dict[str, Any]],
) -> int:
    if not os.path.isfile(bundle_path):
        return 0
    local_found = 0
    for user_block in iter_crawl_bundle_users(bundle_path):
        if not isinstance(user_block, dict):
            continue
        if normalize_uid(user_block.get("user_id")) != uid:
            continue
        records = user_block.get("records") or []
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, dict):
                continue
            p = _post_from_row(rec)
            if p is None:
                continue
            iid = str(p.get("item_id") or "").strip()
            if iid and iid in seen_ids:
                continue
            if iid:
                seen_ids.add(iid)
            posts.append(p)
            local_found += 1
    return local_found


def load_user_weibo_posts(
    user_id: str,
    output_dir: str,
    *,
    limit: int = 800,
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    先扫 unified_*.jsonl，再补扫 weibo_crawl_latest.json（合并后中间件常已删除，正文在 bundle）。
    按 item_id 去重，按时间倒序。
    """
    uid = normalize_uid(user_id)
    if not uid:
        return [], []

    out_abs = os.path.abspath(output_dir)
    seen_ids: set[str] = set()
    posts: list[dict[str, Any]] = []
    files_hit: list[str] = []

    for path in _unified_jsonl_paths(out_abs):
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
                if normalize_uid(row.get("user_id")) != uid:
                    continue
                p = _post_from_row(row)
                if p is None:
                    continue
                iid = str(p.get("item_id") or "").strip()
                if iid and iid in seen_ids:
                    continue
                if iid:
                    seen_ids.add(iid)
                posts.append(p)
                local_found += 1
        if local_found:
            files_hit.append(os.path.basename(path))

    bundle_path = os.path.join(out_abs, "weibo_crawl_latest.json")
    n_bundle = _merge_bundle_posts_for_user(bundle_path, uid, seen_ids=seen_ids, posts=posts)
    if n_bundle:
        files_hit.append(os.path.basename(bundle_path))

    posts.sort(key=lambda x: x.get("created_at") or "", reverse=True)
    if limit and len(posts) > limit:
        posts = posts[:limit]

    return posts, files_hit
