import argparse
import datetime
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WEIBO_DIR = PROJECT_ROOT / "weibospider"
OUTPUT_DIR = PROJECT_ROOT / "output"
INPUT_USER_IDS = PROJECT_ROOT / "input" / "user_ids.txt"


def _run_spider(mode: str, env: dict[str, str]) -> None:
    cmd = [sys.executable, "run_spider.py", mode]
    print(f"[RUN] {' '.join(cmd)} (cwd={WEIBO_DIR})")
    try:
        subprocess.run(cmd, cwd=str(WEIBO_DIR), env=env, check=True)
    except KeyboardInterrupt as exc:
        raise RuntimeError(f"Interrupted while running spider: {mode}") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Spider failed: {mode} (exit_code={exc.returncode})") from exc


def _list_unified_files() -> set[Path]:
    if not OUTPUT_DIR.is_dir():
        return set()
    return set(p for p in OUTPUT_DIR.glob("unified_*.jsonl") if p.is_file())


def _extract_user_candidates(unified_paths: list[Path]) -> dict[str, set[str]]:
    """
    Return mapping: keyword -> candidate user_id set
    """
    by_keyword: dict[str, set[str]] = defaultdict(set)
    for path in unified_paths:
        with path.open("rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                spider_name = str(row.get("spider", "")).strip().lower()
                if "keyword" not in spider_name:
                    continue
                uid = str(row.get("user_id", "")).strip()
                raw = row.get("raw") or {}
                if not uid:
                    # Compatibility fallback when unified user_id is empty but raw has nested user info.
                    user_obj = raw.get("user") or {}
                    if isinstance(user_obj, dict):
                        uid = str(user_obj.get("_id", "")).strip()
                if not uid:
                    uid = str(raw.get("user_id", "")).strip()
                keyword = str(raw.get("keyword", "")).strip()
                if uid and keyword:
                    by_keyword[keyword].add(uid)
    return dict(by_keyword)


def _balanced_user_pool(
    by_keyword: dict[str, set[str]],
    per_keyword_limit: int,
    total_limit: int,
    seed: int,
) -> list[str]:
    rnd = random.Random(seed)
    picked: set[str] = set()

    keywords = sorted(by_keyword.keys())
    rnd.shuffle(keywords)

    # First pass: sample up to per_keyword_limit from each keyword.
    for keyword in keywords:
        uids = list(by_keyword.get(keyword, set()))
        rnd.shuffle(uids)
        for uid in uids[: max(0, per_keyword_limit)]:
            if uid in picked:
                continue
            picked.add(uid)
            if len(picked) >= total_limit:
                return sorted(picked)

    # Second pass: fill remainder from leftover users globally.
    leftovers: list[str] = []
    for keyword in keywords:
        for uid in by_keyword.get(keyword, set()):
            if uid not in picked:
                leftovers.append(uid)
    rnd.shuffle(leftovers)
    for uid in leftovers:
        picked.add(uid)
        if len(picked) >= total_limit:
            break
    return sorted(picked)


def _write_user_ids(user_ids: list[str], append: bool) -> tuple[int, int]:
    INPUT_USER_IDS.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    if append and INPUT_USER_IDS.exists():
        with INPUT_USER_IDS.open("rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                uid = line.strip()
                if uid:
                    existing.add(uid)
    before = len(existing)
    merged = sorted(existing.union(user_ids))
    with INPUT_USER_IDS.open("wt", encoding="utf-8") as f:
        f.write("\n".join(merged))
    return before, len(merged)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Three-step multi-dimensional crawl: keyword recall -> user profile -> user timeline."
    )
    parser.add_argument(
        "--keywords",
        required=True,
        help="Comma-separated keywords, e.g. '手机评测,骁龙,AIGC,ETF,原神'",
    )
    parser.add_argument("--max-pages", type=int, default=5, help="WEIBO_MAX_PAGES for keyword crawl")
    parser.add_argument(
        "--split-by-hour",
        action="store_true",
        help="Enable WEIBO_SPLIT_BY_HOUR=true for keyword crawl (default off to reduce 418 risk)",
    )
    parser.add_argument(
        "--start-time",
        default="",
        help='Optional WEIBO_START_TIME, e.g. "2026-03-01 00:00:00" or "2026-03-01"',
    )
    parser.add_argument(
        "--end-time",
        default="",
        help='Optional WEIBO_END_TIME, e.g. "2026-04-01 23:59:59" or "2026-04-01"',
    )
    parser.add_argument("--per-keyword-limit", type=int, default=100, help="Max sampled users per keyword")
    parser.add_argument("--total-limit", type=int, default=1000, help="Total user_ids written to input/user_ids.txt")
    parser.add_argument("--download-delay", type=float, default=0.2, help="WEIBO_DOWNLOAD_DELAY")
    parser.add_argument("--concurrent-requests", type=int, default=8, help="WEIBO_CONCURRENT_REQUESTS")
    parser.add_argument("--retry-times", type=int, default=2, help="WEIBO_RETRY_TIMES")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balanced sampling")
    parser.add_argument(
        "--append-user-ids",
        action="store_true",
        help="Append sampled IDs to existing input/user_ids.txt instead of overwrite",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["WEIBO_KEYWORDS"] = args.keywords
    env["WEIBO_MAX_PAGES"] = str(max(1, int(args.max_pages)))
    env["WEIBO_SPLIT_BY_HOUR"] = "true" if bool(args.split_by_hour) else "false"
    env["WEIBO_DOWNLOAD_DELAY"] = str(max(0.0, float(args.download_delay)))
    env["WEIBO_CONCURRENT_REQUESTS"] = str(max(1, int(args.concurrent_requests)))
    env["WEIBO_RETRY_TIMES"] = str(max(0, int(args.retry_times)))
    env["WEIBO_CRAWL_TIME_SPAN"] = "false"
    start_time = (args.start_time or "").strip()
    end_time = (args.end_time or "").strip()
    if not start_time or not end_time:
        now = datetime.datetime.now().replace(microsecond=0)
        default_start = (now - datetime.timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
        default_end = now.strftime("%Y-%m-%d %H:%M:%S")
        if not start_time:
            start_time = default_start
        if not end_time:
            end_time = default_end
    env["WEIBO_START_TIME"] = start_time
    env["WEIBO_END_TIME"] = end_time
    print(f"[INFO] keyword crawl time range: {start_time} -> {end_time}")

    try:
        before = _list_unified_files()
        print("[STEP 1/3] Keyword recall crawl (tweet_by_keyword)")
        _run_spider("tweet_by_keyword", env)
        after = _list_unified_files()
        new_unified = sorted(list(after - before), key=lambda p: p.name)
        if not new_unified:
            print("[ERROR] No new unified_*.jsonl produced by keyword crawl.")
            print("[HINT] Usually caused by anti-crawl (418), stale cookie, or no results in current time range.")
            print("[HINT] Try smaller keyword set, newer cookie, and lower speed:")
            print("       --max-pages 1 --download-delay 1.2 --concurrent-requests 1")
            return 2
        print(f"[INFO] New unified files: {len(new_unified)}")

        by_keyword = _extract_user_candidates(new_unified)
        if not by_keyword:
            print("[ERROR] No candidate user_id found from keyword crawl output.")
            return 3
        for keyword in sorted(by_keyword.keys()):
            print(f"[INFO] keyword={keyword} candidates={len(by_keyword[keyword])}")

        picked = _balanced_user_pool(
            by_keyword=by_keyword,
            per_keyword_limit=max(1, int(args.per_keyword_limit)),
            total_limit=max(1, int(args.total_limit)),
            seed=int(args.seed),
        )
        if not picked:
            print("[ERROR] Balanced sampler produced empty user list.")
            return 4
        before_cnt, after_cnt = _write_user_ids(picked, append=bool(args.append_user_ids))
        print(f"[INFO] input/user_ids.txt updated: before={before_cnt} after={after_cnt} sampled_now={len(picked)}")

        env_followup = os.environ.copy()
        env_followup["WEIBO_MAX_PAGES"] = str(max(1, int(args.max_pages)))
        env_followup["WEIBO_DOWNLOAD_DELAY"] = str(max(0.0, float(args.download_delay)))
        env_followup["WEIBO_CONCURRENT_REQUESTS"] = str(max(1, int(args.concurrent_requests)))
        env_followup["WEIBO_RETRY_TIMES"] = str(max(0, int(args.retry_times)))
        # Rely on runtime_config fallback: input/user_ids.txt
        env_followup.pop("WEIBO_USER_IDS", None)

        print("[STEP 2/3] Crawl static profiles (user)")
        _run_spider("user", env_followup)

        print("[STEP 3/3] Crawl user timelines (tweet_by_user_id)")
        _run_spider("tweet_by_user_id", env_followup)

        print("[DONE] Multi-dimensional crawl finished.")
        print("       Next: run your existing clean/train/infer pipeline.")
        return 0
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 10


if __name__ == "__main__":
    raise SystemExit(main())
