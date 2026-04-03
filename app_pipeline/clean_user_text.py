"""
clean_user_text.py

把 `weibo_crawl_latest.json`（bundle）里每个 user 的 tweets 按与单条微博聚类
（train_tweet_topic / feature_tweet_embeddings）相同的规则过滤、清洗后拼成长文本，
写出到 output，与 KMeans 工作链语义一致。

输出：
  output/cleaned_user_texts.jsonl
每行一个 JSON：
  {"user_id": "...", "cleaned_text": "...", "n_texts": 123}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from app_pipeline.data_io import iter_crawl_bundle_users
from app_pipeline.feature_bert_textrank import _text_is_retweet
from app_pipeline.tweet_text_normalize import clean_tweet_for_encode


@dataclass
class CleanConfig:
    """min_tweet_chars 与 TweetEmbeddingConfig 默认一致；min_chars 为拼接后用户级总长度下限。"""
    min_chars: int = 200
    min_tweet_chars: int = 5


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
    filtered_short = 0
    filtered_empty_text = 0
    filtered_non_tweet = 0
    filtered_retweet = 0
    filtered_tweet_below_min = 0

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

            parts: List[str] = []
            n_texts = 0

            for rec in records:
                if not isinstance(rec, dict):
                    continue
                source_type = rec.get("source_type")
                if source_type != "tweet":
                    filtered_non_tweet += 1
                    continue

                text = rec.get("text") or ""
                if _text_is_retweet(str(text)):
                    filtered_retweet += 1
                    continue

                cleaned = clean_tweet_for_encode(str(text))
                if len(cleaned) < cfg.min_tweet_chars:
                    filtered_tweet_below_min += 1
                    continue
                parts.append(cleaned)
                n_texts += 1

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
        "filtered_short": filtered_short,
        "filtered_empty_text": filtered_empty_text,
        "filtered_non_tweet_rows": filtered_non_tweet,
        "filtered_retweet_rows": filtered_retweet,
        "filtered_tweet_below_min_chars": filtered_tweet_below_min,
        "min_chars": cfg.min_chars,
        "min_tweet_chars": cfg.min_tweet_chars,
    }

