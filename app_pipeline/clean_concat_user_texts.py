"""
clean_concat_user_texts.py

把 `output/user_text_concat_*.jsonl`（每行一个用户的拼接原文）进一步清洗，产出与
`output/cleaned_user_texts.jsonl` 同结构的文件，供后续 TextRank/text2vec/聚类直接使用。

输入行格式（由 concat_user_texts.py 生成）：
  {"user_id": "...", "n_texts": 46, "text": "..."}

输出行格式：
  {"user_id": "...", "cleaned_text": "...", "n_texts": 46}
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from app_pipeline.preprocess import clean_text


_RETWEET_RE = re.compile(r"^\s*转发微博\s*$")


def _text_is_retweet(text: str) -> bool:
    if not text:
        return False
    return bool(_RETWEET_RE.match(str(text).strip()))


@dataclass
class CleanConfig:
    min_chars: int = 200


def clean_concat_jsonl_to_jsonl(
    input_jsonl_path: str,
    output_jsonl_path: str,
    *,
    cfg: CleanConfig,
    max_users: Optional[int] = None,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_jsonl_path) or ".", exist_ok=True)

    processed = 0
    kept = 0
    filtered_short = 0
    filtered_empty = 0

    with open(input_jsonl_path, "rt", encoding="utf-8", errors="replace") as in_f, open(
        output_jsonl_path, "wt", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            uid = str(obj.get("user_id", "")).strip()
            if not uid:
                continue
            processed += 1
            if max_users is not None and processed > max_users:
                break

            raw_text = str(obj.get("text", "") or "").strip()
            n_texts = int(obj.get("n_texts", 0) or 0)
            if not raw_text:
                filtered_empty += 1
                continue

            # 清洗：链接/话题/提及/表情占位符/emoji/非文本字符等
            cleaned = clean_text(raw_text)
            if not cleaned or _text_is_retweet(cleaned):
                filtered_empty += 1
                continue
            if len(cleaned) < int(cfg.min_chars):
                filtered_short += 1
                continue

            out_f.write(
                json.dumps(
                    {"user_id": uid, "cleaned_text": cleaned, "n_texts": n_texts},
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1

    return {
        "input_jsonl": input_jsonl_path,
        "output_jsonl": output_jsonl_path,
        "processed_users": processed,
        "kept_users": kept,
        "filtered_short": filtered_short,
        "filtered_empty": filtered_empty,
        "min_chars": cfg.min_chars,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean concatenated per-user text jsonl.")
    parser.add_argument(
        "--input",
        default=os.path.join("output", "user_text_concat_20260402224047.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("output", "cleaned_user_texts_20260402224047.jsonl"),
    )
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--max-users", type=int, default=0)
    args = parser.parse_args()

    input_path = str(args.input)
    output_path = str(args.output)
    if not os.path.isfile(input_path):
        print(f"[ERROR] input not found: {input_path}")
        return 2

    cfg = CleanConfig(min_chars=int(args.min_chars))
    meta = clean_concat_jsonl_to_jsonl(
        input_path,
        output_path,
        cfg=cfg,
        max_users=(int(args.max_users) if int(args.max_users) > 0 else None),
    )
    print("[DONE] cleaned concat-user dataset")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

