"""
按簇聚合用户清洗文本，用 jieba 分词 + TF-IDF 提取关键词（与 test_sandbox/cluster_viz.py 逻辑一致）。
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# 常见中文停用词（与 cluster_viz 对齐，可按需扩充）
STOP_WORDS_CN = frozenset(
    [
        "的",
        "了",
        "是",
        "我",
        "在",
        "和",
        "就",
        "都",
        "也",
        "有",
        "与",
        "及",
        "等",
        "为",
        "对",
        "被",
        "自己",
        "就是",
    ]
)


def load_user_texts_from_cleaned_jsonl(path: str) -> dict[str, str]:
    by_uid: dict[str, str] = {}
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(obj.get("user_id", "")).strip()
            text = str(obj.get("cleaned_text", "") or "").strip()
            if uid and text:
                by_uid[uid] = text
    return by_uid


def get_top_keywords(
    text_list: list[str],
    top_n: int = 10,
    *,
    stop_words: Iterable[str] | None = None,
) -> list[str]:
    stop = frozenset(stop_words) if stop_words is not None else STOP_WORDS_CN
    words_list: list[str] = []
    for text in text_list:
        words = [w for w in jieba.cut(str(text)) if w not in stop and len(w) > 1]
        words_list.append(" ".join(words))
    if not words_list or not any(words_list):
        return []
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(words_list)
    feature_names = vectorizer.get_feature_names_out()
    sum_tfidf = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    words_freq = [(feature_names[i], float(sum_tfidf[i])) for i in range(len(feature_names))]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return [word for word, _ in words_freq[: int(top_n)]]


def keywords_per_primary_cluster(
    user_ids: list,
    primary_per_user: list[int],
    uid_to_text: dict[str, str],
    *,
    n_clusters: int,
    top_n: int = 10,
) -> dict[int, list[str]]:
    """按「主簇」聚合该簇内用户文本，再 TF-IDF 取词。"""
    by_c: dict[int, list[str]] = {c: [] for c in range(int(n_clusters))}
    for uid, pc in zip(user_ids, primary_per_user):
        uid_s = str(uid).strip()
        t = uid_to_text.get(uid_s, "")
        if t and 0 <= int(pc) < int(n_clusters):
            by_c[int(pc)].append(t)
    out: dict[int, list[str]] = {}
    for c in range(int(n_clusters)):
        texts = by_c.get(c, [])
        out[c] = get_top_keywords(texts, top_n=top_n) if texts else []
    return out


def pick_default_cleaned_jsonl(output_dir: str = "output") -> str:
    """优先合并清洗结果，否则单文件。"""
    candidates = [
        os.path.join(output_dir, "cleaned_user_texts_merged.jsonl"),
        os.path.join(output_dir, "cleaned_user_texts.jsonl"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return os.path.normpath(p)
    return ""
