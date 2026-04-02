import argparse
import json
from collections import defaultdict
from pathlib import Path


def discover_unified_jsonl(output_dir: Path, *, sort_by: str = "name") -> list[Path]:
    """
    自动收集 output_dir 下所有 unified_*.jsonl。
    sort_by: 'name'（按文件名，时间戳文件名即时间顺序）或 'mtime'（磁盘修改时间从早到晚）。
    """
    if not output_dir.is_dir():
        return []
    paths = sorted(output_dir.glob("unified_*.jsonl"))
    paths = [p for p in paths if p.is_file()]
    if sort_by == "mtime":
        paths = sorted(paths, key=lambda p: p.stat().st_mtime)
    return paths


def _tweet_key(uid: str, row: dict, text: str) -> tuple[str, str]:
    raw = row.get("raw") or {}
    item_id = str(row.get("item_id") or raw.get("_id") or "").strip()
    if item_id:
        return (uid, f"id:{item_id}")
    return (uid, f"t:{hash(text)}")


def concat_from_unified(input_path: Path, *, dedupe_keys: set[tuple[str, str]] | None) -> dict[str, list[str]]:
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
            text = str(raw.get("content") or row.get("text") or "").strip()
            if not text:
                continue
            if dedupe_keys is not None:
                key = _tweet_key(uid, row, text)
                if key in dedupe_keys:
                    continue
                dedupe_keys.add(key)
            by_user[uid].append(text)
    return dict(by_user)


def concat_from_unified_files(paths: list[Path], *, dedupe: bool = True) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = defaultdict(list)
    keys: set[tuple[str, str]] | None = set() if dedupe else None
    for p in paths:
        part = concat_from_unified(p, dedupe_keys=keys)
        for uid, texts in part.items():
            merged[uid].extend(texts)
    return dict(merged)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Concatenate all texts per user_id from one or more unified_*.jsonl files."
    )
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        default=None,
        metavar="PATH",
        help="Input unified jsonl（可重复）；不传则自动使用 output 下全部 unified_*.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="未指定 --input 时，在此目录下扫描 unified_*.jsonl（默认: output）",
    )
    parser.add_argument(
        "--sort-unified",
        choices=("name", "mtime"),
        default="name",
        help="自动发现时的文件顺序：name=按文件名；mtime=按修改时间（默认 name）",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not dedupe tweets across files (by item_id / _id).",
    )
    parser.add_argument(
        "--output",
        default=str(Path("output") / "user_text_concat_merged.jsonl"),
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

    if args.inputs:
        input_paths = [Path(p) for p in args.inputs]
    else:
        scan_dir = Path(args.output_dir)
        input_paths = discover_unified_jsonl(scan_dir, sort_by=str(args.sort_unified))
        if not input_paths:
            print(f"[ERROR] 在 {scan_dir.resolve()} 下未找到 unified_*.jsonl，请先生成抓取数据或改用 --input 指定文件。")
            return 2
        print(f"[INFO] 自动发现 {len(input_paths)} 个 unified 文件（--sort-unified={args.sort_unified}）")

    for input_path in input_paths:
        if not input_path.is_file():
            print(f"[ERROR] input not found: {input_path}")
            return 2

    by_user = concat_from_unified_files(input_paths, dedupe=not bool(args.no_dedupe))
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
    print(f"       sources ({len(input_paths)}): " + ", ".join(str(p) for p in input_paths))
    if target_uid:
        print(f"       (filtered by user_id={target_uid})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

