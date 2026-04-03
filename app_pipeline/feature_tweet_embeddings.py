"""
Per-tweet sentence embeddings (text2vec / BERT-style) for topic-level K-Means.

Reuses the same user/tweet filters as feature_bert_textrank (drop 大 V、转发微博等)，
但不再做 TextRank：对每条有效微博的清洗全文直接 encode，一条微博一个向量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from app_pipeline.data_io import iter_crawl_bundle_users, iter_unified_jsonl_records
from app_pipeline.feature_bert_textrank import _text_is_retweet
from app_pipeline.tweet_text_normalize import clean_tweet_for_encode


@dataclass
class TweetEmbeddingConfig:
    min_tweet_chars: int = 5
    encode_batch_size: int = 64
    device: str = "cuda"
    model_name: str = "shibing624/text2vec-base-chinese"


def _append_tweet_from_record(
    uid: str,
    rec: Dict[str, Any],
    *,
    min_tweet_chars: int,
    user_ids: List[str],
    texts: List[str],
    keys: List[str],
) -> bool:
    if not isinstance(rec, dict):
        return False
    if rec.get("source_type") != "tweet":
        return False
    text = rec.get("text") or ""
    if _text_is_retweet(text):
        return False
    cleaned = clean_tweet_for_encode(text)
    if len(cleaned) < min_tweet_chars:
        return False
    item_id = str(rec.get("item_id", "") or "").strip()
    key = f"{uid}|{item_id}" if item_id else f"{uid}|{len(user_ids)}"
    user_ids.append(uid)
    texts.append(cleaned)
    keys.append(key)
    return True


def collect_tweets_flat(
    bundle_path: str,
    *,
    min_tweet_chars: int = 5,
    max_tweets: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Flatten crawl bundle JSON 或 unified_*.jsonl → parallel lists: user_id, cleaned text, row key.
    二者语义一致：unified 经 merge 即得到 weibo_crawl_latest 风格 bundle。
    """
    user_ids: List[str] = []
    texts: List[str] = []
    keys: List[str] = []

    if bundle_path.lower().endswith(".jsonl"):
        for rec in iter_unified_jsonl_records(bundle_path):
            uid = str(rec.get("user_id", "")).strip()
            if not uid:
                continue
            _append_tweet_from_record(
                uid,
                rec,
                min_tweet_chars=min_tweet_chars,
                user_ids=user_ids,
                texts=texts,
                keys=keys,
            )
            if max_tweets is not None and len(texts) >= max_tweets:
                return user_ids, texts, keys
        return user_ids, texts, keys

    for user_block in iter_crawl_bundle_users(bundle_path):
        if not isinstance(user_block, dict):
            continue
        uid = str(user_block.get("user_id", "")).strip()
        if not uid:
            continue
        records = user_block.get("records") or []
        if not isinstance(records, list) or not records:
            continue

        for rec in records:
            _append_tweet_from_record(
                uid,
                rec,
                min_tweet_chars=min_tweet_chars,
                user_ids=user_ids,
                texts=texts,
                keys=keys,
            )
            if max_tweets is not None and len(texts) >= max_tweets:
                return user_ids, texts, keys

    return user_ids, texts, keys


def extract_tweet_embeddings(
    bundle_path: str,
    *,
    cfg: TweetEmbeddingConfig,
    max_tweets: Optional[int] = None,
    progress_every: int = 2000,
) -> Tuple[List[str], np.ndarray, List[str]]:
    """
    Returns:
      tweet_user_ids: length N (parallel to rows of X)
      X: (N, dim) float32 — one row per tweet
      tweet_texts: cleaned text per row（与 infer 证据对齐）
    """
    from sentence_transformers import SentenceTransformer

    uids, texts, _keys = collect_tweets_flat(
        bundle_path, min_tweet_chars=cfg.min_tweet_chars, max_tweets=max_tweets
    )
    if not texts:
        raise RuntimeError("No tweets passed filters; check bundle content and min_tweet_chars.")

    model = SentenceTransformer(cfg.model_name, device=cfg.device)
    dim = None
    chunks: List[np.ndarray] = []
    n = len(texts)
    bs = max(1, int(cfg.encode_batch_size))
    done = 0
    while done < n:
        batch = texts[done : done + bs]
        vecs = model.encode(
            batch,
            batch_size=bs,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32, copy=False)
        if dim is None:
            dim = vecs.shape[1]
        chunks.append(vecs)
        done += len(batch)
        if progress_every and done % progress_every < bs or done == n:
            print(f"[EMBED] tweets={done}/{n}", flush=True)

    x = np.concatenate(chunks, axis=0)
    if len(uids) != x.shape[0]:
        raise RuntimeError("tweet user id list length mismatch vs embeddings")
    return uids, x, texts


def mean_embedding_by_user(
    tweet_user_ids: Sequence[str], tweet_vectors: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """Average tweet vectors per user (order of users is sorted by uid)."""
    from collections import defaultdict

    by_u: Dict[str, List[int]] = defaultdict(list)
    for i, u in enumerate(tweet_user_ids):
        by_u[str(u).strip()].append(i)
    users = sorted(by_u.keys())
    out: List[np.ndarray] = []
    for u in users:
        idxs = by_u[u]
        out.append(tweet_vectors[idxs].mean(axis=0))
    return users, np.stack(out, axis=0).astype(np.float32, copy=False)


def static_features_for_users(
    bundle_path: str, user_ids: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    从 bundle 或 unified jsonl 里为每个 user_id 取一条 raw.user 统计量（log1p），找不到则 0。
    返回 shape (n_users, 3)，列顺序: followers, friends, statuses。
    """
    want = {str(u).strip() for u in user_ids if str(u).strip()}
    found: Dict[str, Dict[str, Any]] = {}

    if bundle_path.lower().endswith(".jsonl"):
        for rec in iter_unified_jsonl_records(bundle_path):
            uid = str(rec.get("user_id", "")).strip()
            if not uid or uid not in want or uid in found:
                continue
            raw = rec.get("raw") if isinstance(rec.get("raw"), dict) else {}
            user = raw.get("user") if isinstance(raw.get("user"), dict) else {}
            if user:
                found[uid] = user
        names = ["followers_count", "friends_count", "statuses_count"]
        mat = np.zeros((len(user_ids), len(names)), dtype=np.float32)
        for i, uid in enumerate(user_ids):
            u = found.get(str(uid).strip())
            if not u:
                continue
            for j, k in enumerate(names):
                try:
                    mat[i, j] = float(np.log1p(max(0, int(u.get(k) or 0))))
                except Exception:
                    pass
        return mat, names

    for user_block in iter_crawl_bundle_users(bundle_path):
        if not isinstance(user_block, dict):
            continue
        uid = str(user_block.get("user_id", "")).strip()
        if not uid or uid not in want or uid in found:
            continue
        records = user_block.get("records") or []
        if not isinstance(records, list):
            continue
        for rec in records:
            raw = rec.get("raw") if isinstance(rec.get("raw"), dict) else {}
            user = raw.get("user") if isinstance(raw.get("user"), dict) else {}
            if not user:
                continue
            found[uid] = user
            break

    names = ["followers_count", "friends_count", "statuses_count"]
    mat = np.zeros((len(user_ids), len(names)), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        u = found.get(str(uid).strip())
        if not u:
            continue
        for j, k in enumerate(names):
            try:
                mat[i, j] = float(np.log1p(max(0, int(u.get(k) or 0))))
            except Exception:
                pass
    return mat, names
