"""
feature_bert_textrank.py

给定我们已爬取的 weibo_crawl_latest.json（bundle：users[].records[]），为每个用户提取一个 768 维向量：

1) 数据解析与清洗 + 过滤：
   - 内容级过滤：
       - source_type 必须为 'tweet'
       - text 必须不是 '转发微博'
   - 对保留文本进行正则清洗：去话题(#...#)、@、链接、表情符号，并做空白归一
   - 拼接为用户长文本；拼接后文本过短则丢弃该用户

2) 关键词提取（TextRank）：
   - jieba.analyse.textrank(long_text, topK=10) 取核心关键词
   - 若关键词不足 topK，则丢弃该用户（保证输入与 pooling 维度对齐）

3) 语义向量化（轻量中文 text2vec）：
   - 使用 sentence-transformers 加载 shibing624/text2vec-base-chinese
   - device='cuda'（按需；代码里会用参数控制）
   - 对每个用户的 10 个关键词分别 encode 得到词向量，再对 10 个向量做 Mean Pooling 得到用户向量

4) 输出：
   - 返回 user_ids（list[str]）
   - 返回 features（numpy.ndarray float32, shape=(n_users, 768)）

注意：这是“建模前置步骤”，也可以被 train/infer 复用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import jieba.analyse
import numpy as np

from app_pipeline.data_io import iter_crawl_bundle_users
from app_pipeline.preprocess import clean_text


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#([^#]+)#?")
# emoji codepoints (较通用的范围，避免引入额外依赖)
_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F6FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\uFE0F"
    "]+"
)
_WS_RE = re.compile(r"\s+")


def _text_is_retweet(text: str) -> bool:
    if not text:
        return False
    t = str(text).strip()
    return t == "转发微博" or t.startswith("转发微博")


def _clean_text_for_keywords(text: str) -> str:
    """
    清洗文本用于关键词提取：
    - 使用 preprocess.clean_text 为主（它已经去 URL/@/话题/非文本）
    - 再补一层 emoji codepoint 清理 + 空白归一
    """
    if not text:
        return ""
    t = clean_text(text)
    t = _URL_RE.sub(" ", t)
    t = _MENTION_RE.sub(" ", t)
    t = _HASHTAG_RE.sub(" ", t)
    t = _EMOJI_RE.sub(" ", t)
    t = _WS_RE.sub(" ", t).strip()
    return t


def extract_textrank_keywords(long_text: str, top_k: int) -> List[str]:
    if not long_text:
        return []
    kws = jieba.analyse.textrank(long_text, topK=top_k, withWeight=False)
    out: List[str] = []
    for k in kws:
        kk = str(k).strip()
        if kk:
            out.append(kk)
    return out


@dataclass
class FeatureExtractionConfig:
    top_k: int = 10
    min_chars: int = 200
    # 编码关键词列表时的 batch_size（不是用户数）
    encode_batch_size: int = 64
    device: str = "cuda"
    model_name: str = "shibing624/text2vec-base-chinese"
    # 控制每个向量化 batch 中包含多少用户（避免一次性堆太多关键词）
    user_batch_size: int = 64
    # 是否强制关键词数量必须==top_k（不够则丢弃用户）
    require_full_topk: bool = True


def _build_user_long_text(records: Sequence[Dict[str, Any]]) -> str:
    """
    对一个用户的 records 做：
    - 非原创过滤：source_type != 'tweet' 或 text == '转发微博' 丢弃
    - 文本清洗 + 拼接
    （大 V 不再整用户丢弃，与数据采集口径一致。）
    """
    parts: List[str] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get("source_type") != "tweet":
            continue
        text = rec.get("text") or ""
        if _text_is_retweet(text):
            continue
        cleaned = _clean_text_for_keywords(str(text))
        if cleaned:
            parts.append(cleaned)
    return " ".join(parts).strip()


def _mean_pool_keywords(keyword_vecs: np.ndarray) -> np.ndarray:
    """
    keyword_vecs: shape=(top_k, dim)
    return: shape=(dim,)
    """
    return keyword_vecs.mean(axis=0)


def extract_user_features(
    bundle_path: str,
    *,
    cfg: FeatureExtractionConfig,
    max_users: Optional[int] = None,
) -> Tuple[List[str], np.ndarray]:
    """
    主函数：从 bundle_path 抽取全部用户特征。
    - 返回 user_ids 和 features (float32, shape=(n_users, dim))
    """
    # 延迟导入，避免未安装依赖时影响其它模块
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.model_name, device=cfg.device)

    user_ids: List[str] = []
    features: List[np.ndarray] = []

    pending_users: List[Tuple[str, List[str]]] = []  # (uid, keywords)
    flat_keywords: List[str] = []

    def flush_encode() -> None:
        nonlocal pending_users, flat_keywords, features
        if not pending_users:
            return
        # sentence-transformers encode 返回 (n, dim)
        # 注意：flat_keywords 长度应为 len(pending_users)*top_k
        vecs = model.encode(
            flat_keywords,
            batch_size=cfg.encode_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype(np.float32, copy=False)

        dim = vecs.shape[1]
        top_k = cfg.top_k
        n_users_batch = len(pending_users)
        if vecs.shape[0] != n_users_batch * top_k:
            raise RuntimeError(f"encode shape mismatch: got {vecs.shape[0]}, expect {n_users_batch*top_k}")

        vecs = vecs.reshape(n_users_batch, top_k, dim)
        for i in range(n_users_batch):
            features.append(_mean_pool_keywords(vecs[i]))

        pending_users = []
        flat_keywords = []

    for user_block in iter_crawl_bundle_users(bundle_path):
        if not isinstance(user_block, dict):
            continue
        uid = str(user_block.get("user_id", "")).strip()
        if not uid:
            continue
        if max_users is not None and len(user_ids) >= max_users:
            break

        records = user_block.get("records") or []
        if not isinstance(records, list) or not records:
            continue

        long_text = _build_user_long_text(records)
        if not long_text or len(long_text) < cfg.min_chars:
            continue

        keywords = extract_textrank_keywords(long_text, top_k=cfg.top_k)
        if cfg.require_full_topk:
            if len(keywords) < cfg.top_k:
                continue
        else:
            keywords = keywords[: cfg.top_k]
            if len(keywords) < cfg.top_k:
                continue

        keywords = keywords[: cfg.top_k]
        user_ids.append(uid)
        pending_users.append((uid, keywords))
        # flatten keywords for encoding
        for kw in keywords:
            flat_keywords.append(kw)

        # 达到用户批大小，开始编码并 mean pooling
        if len(pending_users) >= cfg.user_batch_size:
            flush_encode()

    # flush remaining
    flush_encode()

    if not user_ids or not features:
        raise RuntimeError("No users passed filters; try lowering min_chars or relaxing constraints.")

    x = np.stack(features, axis=0).astype(np.float32, copy=False)
    return user_ids, x

