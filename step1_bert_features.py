"""
step1_bert_features.py

目的：
1) 读取 output/weibo_crawl_latest.json（bundle：top-level 含 users 列表，每个 user 含 records 列表）
2) 数据解析与清洗：
   - 剔除大 V：raw.user.verified == True
   - 剔除非原创推文：source_type != 'tweet' 或 text == '转发微博'
   - 使用正则去掉话题、@、链接和表情符号
   - 将每个普通网民洗净后的微博拼成一个长字符串；过短则丢弃该用户
3) 关键词提取（TextRank）：每个用户提取 top10 核心关键词
4) BERT 向量化与池化：
   - 使用 sentence-transformers 加载轻量中文模型：shibing624/text2vec-base-chinese
   - 指定 device='cuda'，使用 GPU 加速
   - 将每个用户的 10 个关键词分别编码为向量，然后对 10 个向量做 Mean Pooling 得到 768 维用户特征
5) 持久化保存：
   - user_features.npy：形状 (n_users, 768)，float32
   - user_ids.pkl：list[str]

用法示例：
  python step1_bert_features.py --input output/weibo_crawl_latest.json --outdir output/step1_bert
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
from typing import Any, Dict, Iterable, Iterator, List, Tuple

import numpy as np
import jieba
import jieba.analyse


# -----------------------------
# 正则：清洗微博文本
# -----------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
# #话题# 或 # 话题 # 形式，尽量覆盖常见写法
_HASHTAG_RE = re.compile(r"#([^#]+)#")
# 粗略表情符号范围：捕获常见 emoji codepoint（不会覆盖所有情况，但足够用于微博文本清洗）
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F6FF"  # emoticons & symbols
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-a
    "\U00002600-\U000027BF"  # miscellaneous symbols
    "\uFE0F"  # variation selector
    "]+"
)
_WS_RE = re.compile(r"\s+")


def clean_text_basic(text: str) -> str:
    """移除话题、@、链接、表情，并做空白规范化。"""
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    text = _EMOJI_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def is_verified_v(raw: Any) -> bool:
    """
    判断是否大 V：
    - 需求：raw.user.verified == True
    - 这里做更宽容的兼容：raw 里可能有 verified 字段直接给出
    """
    if not isinstance(raw, dict):
        return False
    user = raw.get("user")
    if isinstance(user, dict):
        return bool(user.get("verified") is True)
    return bool(raw.get("verified") is True)


def iter_users_from_bundle(path: str) -> Iterator[Dict[str, Any]]:
    """
    从 bundle JSON 流式读取 users.item，减少内存压力。
    如果没有安装 ijson，则退化为 json.load（可能内存占用较高）。
    """
    try:
        import ijson  # type: ignore
    except Exception:
        with open(path, "rt", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        users = data.get("users") if isinstance(data, dict) else data
        if not isinstance(users, list):
            raise ValueError("crawl bundle must contain a 'users' list/array")
        for u in users:
            yield u
        return

    with open(path, "rb") as f:
        # top-level: { ..., "users": [ ... ] }
        for item in ijson.items(f, "users.item"):
            if isinstance(item, dict):
                yield item


def text_is_retweet(text: str) -> bool:
    """需求：文本为“转发微博”。做宽容匹配。"""
    if not text:
        return False
    t = str(text).strip()
    return t == "转发微博" or t.startswith("转发微博")


def build_user_long_text(user_block: Dict[str, Any]) -> str:
    """
    按规则过滤 records，拼接为“洗净后微博长字符串”。
    过滤逻辑：
    - raw.user.verified == True：丢弃
    - source_type != 'tweet'：丢弃非原创
    - text == '转发微博'：丢弃
    """
    raw_user_verified = False
    # 有些记录的 verified 信息可能在不同 item 中出现；这里尽量从记录 raw 里判断
    # 如果 user_block 本身含 raw/user 字段，也兼容一下
    ub_raw = user_block.get("raw") if isinstance(user_block, dict) else None
    if is_verified_v(ub_raw):
        return ""

    records = user_block.get("records") or []
    parts: List[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        raw = rec.get("raw") or {}
        if is_verified_v(raw):
            raw_user_verified = True
            break
        source_type = rec.get("source_type")
        if source_type != "tweet":
            continue
        text = rec.get("text") or ""
        if text_is_retweet(text):
            continue
        cleaned = clean_text_basic(text)
        if cleaned:
            parts.append(cleaned)

    if raw_user_verified:
        return ""
    return " ".join(parts).strip()


def extract_keywords_textrank(long_text: str, top_k: int = 10) -> List[str]:
    """使用 jieba.analyse.textrank 抽取核心关键词。"""
    if not long_text:
        return []
    # jieba 的 textrank 默认会做词频统计 + 图算法；withWeight=False 返回纯词
    keywords = jieba.analyse.textrank(long_text, topK=top_k, withWeight=False)
    # textrank 可能返回少于 topK 的关键词，这里后续会按“必须 >= top_k 才保留用户”的规则处理
    return [str(k).strip() for k in keywords if str(k).strip()]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default="output/weibo_crawl_latest.json", help="path to weibo_crawl_latest.json")
    ap.add_argument("--outdir", type=str, default="output/step1_bert", help="output directory for artifacts")
    ap.add_argument("--min_chars", type=int, default=200, help="if concatenated cleaned text length < min_chars, drop user")
    ap.add_argument("--top_k", type=int, default=10, help="topK keywords to extract by TextRank")
    ap.add_argument("--batch_size", type=int, default=64, help="sentence-transformers encode batch_size (tune for RTX3050 memory)")
    ap.add_argument("--device", type=str, default="cuda", help="device for sentence-transformers; requirement says use 'cuda'")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.device != "cuda":
        print("[WARN] requirement asks device='cuda'. You passed:", args.device)

    # GPU 前置检查
    try:
        import torch

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot set device='cuda'.")
    except Exception as e:
        raise RuntimeError("Please ensure torch+CUDA environment is installed correctly.") from e

    # 加载模型（轻量中文句向量模型）
    from sentence_transformers import SentenceTransformer

    model_name = "shibing624/text2vec-base-chinese"
    model = SentenceTransformer(model_name, device=args.device)

    # 单词向量维度（text2vec-base-chinese 通常 768；但仍以模型输出为准）
    emb_dim: int | None = None

    user_ids: List[str] = []
    user_keywords: List[List[str]] = []

    # 解析 users，先做清洗 + TextRank 得到每用户 10 个关键词
    for user_block in iter_users_from_bundle(args.input):
        if not isinstance(user_block, dict):
            continue
        uid = user_block.get("user_id")
        if uid is None:
            continue
        uid = str(uid).strip()
        if not uid:
            continue

        long_text = build_user_long_text(user_block)
        if not long_text or len(long_text) < args.min_chars:
            continue

        keywords = extract_keywords_textrank(long_text, top_k=args.top_k)
        if len(keywords) < args.top_k:
            # 为了严格满足“每用户 10 个关键词分别编码”的要求，这里丢弃关键词不足的用户
            continue

        user_ids.append(uid)
        user_keywords.append(keywords[: args.top_k])

    if not user_ids:
        raise RuntimeError("No user passed the filters. Try lowering --min_chars or relaxing filters.")

    # 扁平化 keywords：总输入条数 = n_users * top_k
    flat_keywords: List[str] = []
    for kws in user_keywords:
        flat_keywords.extend(kws)

    # 编码：输出 shape = (n_users*top_k, emb_dim)
    # 注意：sentence-transformers 的 encode 已做了 pooling，符合“词向量”的语义
    all_embs = model.encode(
        flat_keywords,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    if all_embs.ndim != 2:
        raise RuntimeError("Unexpected embedding shape from sentence-transformers.")

    emb_dim = int(all_embs.shape[1])
    n_users = len(user_ids)
    top_k = args.top_k
    expected = n_users * top_k
    if int(all_embs.shape[0]) != expected:
        raise RuntimeError(f"Embedding count mismatch: got {all_embs.shape[0]}, expected {expected}")

    # Mean pooling over the 10 keyword vectors per user
    all_embs = all_embs.astype(np.float32, copy=False)
    all_embs = all_embs.reshape(n_users, top_k, emb_dim)
    user_features = all_embs.mean(axis=1)  # (n_users, emb_dim)

    # 持久化
    npy_path = os.path.join(args.outdir, "user_features.npy")
    ids_path = os.path.join(args.outdir, "user_ids.pkl")
    np.save(npy_path, user_features)
    with open(ids_path, "wb") as f:
        pickle.dump(user_ids, f)

    print(f"[OK] saved user_features.npy: {npy_path}, shape={user_features.shape}")
    print(f"[OK] saved user_ids.pkl: {ids_path}, n_users={len(user_ids)}")


if __name__ == "__main__":
    main()

