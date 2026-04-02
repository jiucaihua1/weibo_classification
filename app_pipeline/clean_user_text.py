"""
clean_user_text.py

只做“数据清洗”这一步，把 `weibo_crawl_latest.json`（bundle）里每个 user 的 tweets
过滤并清洗后拼成长文本，写出到 output，供后续 TextRank / BERT / KMeans / 决策树复用。

输出：
  output/cleaned_user_texts.jsonl
每行一个 JSON：
  {"user_id": "...", "cleaned_text": "...", "n_texts": 123}
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app_pipeline.data_io import iter_crawl_bundle_users
from app_pipeline.preprocess import clean_text


_RETWEET_RE = re.compile(r"^\s*转发微博\s*$")


def _text_is_retweet(text: str) -> bool:
    if not text:
        return False
    return bool(_RETWEET_RE.match(str(text).strip()))


def _is_verified_v(raw: Any) -> bool:
    """
    大 V 判断：raw.user.verified == True
    兼容不同 raw 结构：raw.user.verified 或 raw.verified
    """
    if not isinstance(raw, dict):
        return False
    user = raw.get("user")
    if isinstance(user, dict):
        return bool(user.get("verified") is True)
    return bool(raw.get("verified") is True)


@dataclass
class CleanConfig:
    min_chars: int = 200


def clean_bundle_users_to_jsonl(
    bundle_path: str,
    output_jsonl_path: str,
    *,
    cfg: CleanConfig,
    max_users: Optional[int] = None,
    expected_users: Optional[int] = None,
    progress_cb: Optional[Callable[[int, Optional[int]], None]] = None,
) -> Dict[str, Any]:
    """
    清洗 bundle 并持久化到 jsonl。
    """
    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    processed_users = 0
    kept_users = 0
    filtered_verified = 0
    filtered_short = 0
    filtered_empty_text = 0
    filtered_non_tweet = 0
    filtered_retweet = 0

    with open(output_jsonl_path, "wt", encoding="utf-8") as out_f:
        for user_block in iter_crawl_bundle_users(bundle_path):
            if not isinstance(user_block, dict):
                continue
            uid = str(user_block.get("user_id", "")).strip()
            if not uid:
                continue
            processed_users += 1
            if max_users is not None and kept_users >= max_users:
                break

            # 进度回调（节流：每 2% 或至少每 50 个用户更新一次）
            if progress_cb is not None and expected_users is not None and expected_users > 0:
                step = max(50, int(expected_users * 0.02))
                if (processed_users % step) == 0:
                    progress_cb(processed_users, expected_users)

            records = user_block.get("records") or []
            if not isinstance(records, list) or not records:
                filtered_empty_text += 1
                continue

            verified = False
            parts: List[str] = []
            n_texts = 0

            for rec in records:
                if not isinstance(rec, dict):
                    continue
                raw = rec.get("raw") or {}
                if _is_verified_v(raw):
                    verified = True
                    break

                source_type = rec.get("source_type")
                if source_type != "tweet":
                    filtered_non_tweet += 1
                    continue

                text = rec.get("text") or ""
                if _text_is_retweet(str(text)):
                    filtered_retweet += 1
                    continue

                cleaned = clean_text(str(text))
                if cleaned:
                    parts.append(cleaned)
                    n_texts += 1

            if verified:
                filtered_verified += 1
                continue

            if not parts or n_texts == 0:
                filtered_empty_text += 1
                continue

            long_text = " ".join(parts).strip()
            if len(long_text) < cfg.min_chars:
                filtered_short += 1
                continue

            out_f.write(json.dumps({"user_id": uid, "cleaned_text": long_text, "n_texts": n_texts}, ensure_ascii=False) + "\n")
            kept_users += 1

    return {
        "bundle_path": bundle_path,
        "output_jsonl": output_jsonl_path,
        "total_users": processed_users,
        "kept_users": kept_users,
        "filtered_verified": filtered_verified,
        "filtered_short": filtered_short,
        "filtered_empty_text": filtered_empty_text,
        "filtered_non_tweet_rows": filtered_non_tweet,
        "filtered_retweet_rows": filtered_retweet,
        "min_chars": cfg.min_chars,
    }

