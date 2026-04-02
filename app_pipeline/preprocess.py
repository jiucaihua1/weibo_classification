import re
from typing import List

import jieba


_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#([^#]+)#?")
_EMOJI_PLACEHOLDER_RE = re.compile(r"\[[^\[\]]{1,8}\]")
# emoji codepoints (用于微博正文中常见的实际 emoji 字符)
_EMOJI_CODEPOINT_RE = re.compile(
    "["
    "\U0001F300-\U0001F6FF"  # emoticons & symbols
    "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
    "\U0001FA70-\U0001FAFF"  # symbols & pictographs extended-a
    "\U00002600-\U000027BF"  # misc symbols
    "\uFE0F"  # variation selector
    "]+"
)
_WHITESPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9 ]+")


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(r" \1 ", text)
    text = _EMOJI_PLACEHOLDER_RE.sub(" ", text)
    text = _EMOJI_CODEPOINT_RE.sub(" ", text)
    text = _NON_TEXT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def tokenize_zh(text: str) -> List[str]:
    if not text:
        return []
    return [token.strip() for token in jieba.cut(text) if token.strip() and len(token.strip()) > 1]
