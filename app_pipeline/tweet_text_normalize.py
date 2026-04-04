"""
与单条微博聚类（feature_tweet_embeddings / train_tweet_topic）一致的文本规范化。

供嵌入与聚类流水线（feature_tweet_embeddings / kmeans_prep 等）共用，避免规则漂移。
"""

from __future__ import annotations

import re


def text_is_retweet(text: str) -> bool:
    """原创过滤：微博正文仅为「转发微博」或以其开头时视为转发占位，不参与嵌入/聚类。"""
    if not text:
        return False
    t = str(text).strip()
    return t == "转发微博" or t.startswith("转发微博")


def clean_tweet_for_encode(text: str) -> str:
    """
    White-list based cleaning (aligned with train_tweet_topic pipeline):
    - remove url / mention / #topic#
    - remove zero-width chars + emoji ranges (best-effort)
    - keep only Chinese / letters / digits + basic punctuation
    - normalize whitespace
    """
    t = str(text or "")
    if not t.strip():
        return ""

    t = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", t)

    t = re.sub(r"http[s]?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\S+", " ", t)
    t = re.sub(r"#.*?#", " ", t)

    t = re.sub(
        "["
        "\U0001F300-\U0001F6FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000027BF"
        "\uFE0F"
        "]+",
        " ",
        t,
    )

    t = re.sub(
        r"[^\u4e00-\u9fffA-Za-z0-9，。！？、；：“”‘’（）()【】\[\]{} ,.!?;:\s]+",
        " ",
        t,
    )
    t = re.sub(r"\s+", " ", t).strip()
    return t
