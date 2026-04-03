"""
从抓取 bundle 或 unified jsonl 导出 KMeans 专用输入：每行一条微博（已清洗），
供 train_tweet_topic / infer 直接作为 --input 使用，与嵌入流水线过滤规则一致。
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Optional

from app_pipeline.data_io import iter_crawl_bundle_users, iter_unified_jsonl_records
from app_pipeline.feature_bert_textrank import _text_is_retweet
from app_pipeline.tweet_text_normalize import clean_tweet_for_encode


def _emit_tweet_line(
    uid: str,
    rec: Dict[str, Any],
    *,
    min_tweet_chars: int,
) -> Optional[Dict[str, Any]]:
    if not isinstance(rec, dict):
        return None
    if rec.get("source_type") != "tweet":
        return None
    text = rec.get("text") or ""
    if _text_is_retweet(str(text)):
        return None
    cleaned = clean_tweet_for_encode(str(text))
    if len(cleaned) < min_tweet_chars:
        return None
    item_id = str(rec.get("item_id", "") or "").strip()
    row: Dict[str, Any] = {
        "user_id": uid,
        "text": cleaned,
        "source_type": "tweet",
        "item_id": item_id,
        "kmeans_prep": True,
    }
    for k in ("created_at", "crawl_time", "spider"):
        if rec.get(k) is not None:
            row[k] = rec[k]
    return row


def export_kmeans_tweets_jsonl(
    source_path: str,
    output_jsonl_path: str,
    *,
    min_tweet_chars: int = 5,
    progress_every: int = 5000,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """
    写出 unified 风格 jsonl；仅含通过过滤的微博，text 已为清洗后正文。
    """
    os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)

    n_written = 0
    src = os.path.abspath(source_path)

    with open(output_jsonl_path, "wt", encoding="utf-8") as out_f:
        if src.lower().endswith(".jsonl"):
            for rec in iter_unified_jsonl_records(src):
                uid = str(rec.get("user_id", "")).strip()
                if not uid:
                    continue
                row = _emit_tweet_line(uid, rec, min_tweet_chars=min_tweet_chars)
                if row is None:
                    continue
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_written += 1
                if progress_every and n_written % progress_every == 0:
                    if progress_cb:
                        progress_cb(n_written)
        else:
            for user_block in iter_crawl_bundle_users(src):
                if not isinstance(user_block, dict):
                    continue
                uid = str(user_block.get("user_id", "")).strip()
                if not uid:
                    continue
                records = user_block.get("records") or []
                if not isinstance(records, list):
                    continue
                for rec in records:
                    if not isinstance(rec, dict):
                        continue
                    row = _emit_tweet_line(uid, rec, min_tweet_chars=min_tweet_chars)
                    if row is None:
                        continue
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n_written += 1
                    if progress_every and n_written % progress_every == 0:
                        if progress_cb:
                            progress_cb(n_written)

    if progress_cb:
        progress_cb(n_written)

    return {
        "source_path": source_path,
        "output_jsonl": output_jsonl_path,
        "n_tweets": n_written,
        "min_tweet_chars": min_tweet_chars,
    }
