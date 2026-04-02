import argparse
import json
from collections import defaultdict
from pathlib import Path


def concat_from_unified(input_path: Path) -> dict[str, list[str]]:
    by_user: dict[str, list[str]] = defaultdict(list)
    with input_path.open("rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            uid = str(row.get("user_id", "")).strip()
            if not uid:
                continue
            raw = row.get("raw") or {}
            # Prefer raw.content when available; fallback to unified text.
            text = str(raw.get("content") or row.get("text") or "").strip()
            if text:
                by_user[uid].append(text)
    return dict(by_user)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate all texts per user_id from a unified_*.jsonl file."
    )
    parser.add_argument(
        "--input",
        default=str(Path("output") / "unified_20260402224047.jsonl"),
        help="Input unified jsonl (default: output/unified_20260402224047.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=str(Path("output") / "user_text_concat_20260402224047.jsonl"),
        help="Output jsonl (one user per line)",
    )
    parser.add_argument(
        "--user-id",
        default="",
        help="Optional: only export one user_id (for debugging)",
    )
    parser.add_argument(
        "--sep",
        default=" ",
        help="Separator used to join texts (default: single space)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[ERROR] input not found: {input_path}")
        return 2

    by_user = concat_from_unified(input_path)
    if not by_user:
        print("[ERROR] no texts found in unified file.")
        return 3

    target_uid = (args.user_id or "").strip()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    with output_path.open("wt", encoding="utf-8") as f:
        for uid in sorted(by_user.keys()):
            if target_uid and uid != target_uid:
                continue
            texts = by_user[uid]
            payload = {
                "user_id": uid,
                "n_texts": len(texts),
                "text": str(args.sep).join(texts),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[DONE] wrote {n_written} users to {output_path}")
    if target_uid:
        print(f"       (filtered by user_id={target_uid})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

