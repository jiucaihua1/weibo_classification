"""
step2_textrank_keywords.py

对 `output/cleaned_user_texts.jsonl` 做 TextRank 关键词提取（jieba.analyse.textrank）。

输入（jsonl，每行一个用户）：
  {"user_id": "...", "cleaned_text": "...", "n_texts": 123}

输出（jsonl，每行一个用户）：
  {"user_id": "...", "keywords": ["kw1", ..., "kw10"]}

说明：
- 为了匹配你后续“每个用户都要向量化 10 个关键词”的要求，
  默认：提取到的关键词数量 < 10 时，直接跳过该用户。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import jieba.analyse


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default=os.path.join(OUTPUT_DIR, "cleaned_user_texts.jsonl"),
        help="input jsonl path",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=os.path.join(OUTPUT_DIR, "user_keywords_top10.jsonl"),
        help="output jsonl path",
    )
    ap.add_argument("--top-k", type=int, default=10, help="TextRank topK keywords per user")
    ap.add_argument(
        "--min-keywords",
        type=int,
        default=10,
        help="if extracted keywords < min_keywords, skip this user",
    )
    ap.add_argument("--stats-out", type=str, default=os.path.join(OUTPUT_DIR, "textrank_stats.json"))
    return ap.parse_args()


def extract_textrank(text: str, top_k: int) -> List[str]:
    kws = jieba.analyse.textrank(text, topK=top_k, withWeight=False)
    out: List[str] = []
    for k in kws:
        kk = str(k).strip()
        if kk:
            out.append(kk)
    return out


def main() -> None:
    args = parse_args()

    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"input not found: {args.input}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total = 0
    kept = 0
    skipped_empty = 0
    skipped_short = 0

    with open(args.input, "rt", encoding="utf-8", errors="replace") as f_in, open(
        args.output, "wt", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            total += 1

            obj: Dict[str, Any] = json.loads(line)
            user_id = str(obj.get("user_id", "")).strip()
            text = str(obj.get("cleaned_text", "") or "").strip()

            if not user_id:
                continue
            if not text:
                skipped_empty += 1
                continue

            keywords = extract_textrank(text, top_k=args.top_k)
            if len(keywords) < args.min_keywords:
                skipped_short += 1
                continue

            keywords = keywords[: args.top_k]
            f_out.write(json.dumps({"user_id": user_id, "keywords": keywords}, ensure_ascii=False) + "\n")
            kept += 1

    stats = {
        "input": args.input,
        "output": args.output,
        "top_k": int(args.top_k),
        "min_keywords": int(args.min_keywords),
        "total_users_lines": total,
        "kept_users": kept,
        "skipped_empty_text": skipped_empty,
        "skipped_too_few_keywords": skipped_short,
    }
    with open(args.stats_out, "wt", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("[OK] textrank done")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

