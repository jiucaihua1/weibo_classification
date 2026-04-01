import glob
import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request as UrlRequest, urlopen


def get_hot_search_speed_profile(speed_mode: str) -> dict:
    mode = (speed_mode or "").strip().lower()
    if mode == "fast":
        return {
            "keyword_max_pages": 5,
            "sleep_between_keywords_sec": 0.0,
            "tweets_per_keyword": 50,
            "download_delay": 0.1,
            "concurrent_requests": 10,
            "retry_times": 1,
            "speed_mode": "fast",
        }
    return {
        "keyword_max_pages": 5,
        "sleep_between_keywords_sec": 1.0,
        "tweets_per_keyword": 50,
        "download_delay": 0.5,
        "concurrent_requests": 6,
        "retry_times": 3,
        "speed_mode": "steady",
    }


def latest_hot_search_debug_file(output_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(output_dir, "hot_search_debug_*.json")), key=os.path.getmtime, reverse=True)
    return files[0] if files else ""


def _list_unified_files(output_dir: str) -> set[str]:
    if not os.path.exists(output_dir):
        return set()
    return {os.path.join(output_dir, n) for n in os.listdir(output_dir) if n.startswith("unified_") and n.endswith(".jsonl")}


def _list_user_aggregate_files(output_dir: str) -> set[str]:
    if not os.path.exists(output_dir):
        return set()
    return {
        os.path.join(output_dir, n)
        for n in os.listdir(output_dir)
        if n.startswith("user_aggregate_") and n.endswith(".json")
    }


def cleanup_hot_search_new_crawl_artifacts(
    output_dir: str, before_unified: set[str], before_aggregate: set[str]
) -> tuple[int, int]:
    """
    Delete every unified_*.jsonl / user_aggregate_*.json created during this job.
    Covers orphan aggregates when spider wrote no unified (previously leaked).
    """
    after_u = _list_unified_files(output_dir)
    after_a = _list_user_aggregate_files(output_dir)
    new_u = after_u - before_unified
    new_a = after_a - before_aggregate
    ru, ra = 0, 0
    for p in new_u:
        if os.path.isfile(p):
            try:
                os.remove(p)
                ru += 1
            except OSError:
                pass
        agg = _paired_user_aggregate_path(p)
        if agg and os.path.isfile(agg):
            try:
                os.remove(agg)
                ra += 1
            except OSError:
                pass
    for p in new_a:
        if os.path.isfile(p):
            try:
                os.remove(p)
                ra += 1
            except OSError:
                pass
    return ru, ra


def _pick_new_unified(before: set[str], after: set[str]) -> str:
    candidates = list(after - before)
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _extract_tweet_ids_from_unified(unified_path: str, limit: int = 200) -> list[str]:
    tweet_ids, seen = [], set()
    with open(unified_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("source_type") != "tweet":
                continue
            raw = row.get("raw") or {}
            mblogid = raw.get("mblogid") if isinstance(raw, dict) else None
            if mblogid:
                mblogid = str(mblogid)
                if mblogid not in seen:
                    seen.add(mblogid)
                    tweet_ids.append(mblogid)
            if len(tweet_ids) >= limit:
                break
    return tweet_ids


def _extract_user_ids_by_source_with_stats(unified_path: str, source_type: str, limit: int = 500) -> tuple[list[str], dict]:
    uids, seen = [], set()
    stats = {
        "unified_path": unified_path,
        "source_type": source_type,
        "total_lines": 0,
        "json_error_lines": 0,
        "matched_rows": 0,
        "non_matched_rows": 0,
        "empty_uid_rows": 0,
        "duplicate_uid_rows": 0,
        "accepted_unique_uids": 0,
        "limit_reached": False,
    }
    with open(unified_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            stats["total_lines"] += 1
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                stats["json_error_lines"] += 1
                continue
            if row.get("source_type") != source_type:
                stats["non_matched_rows"] += 1
                continue
            stats["matched_rows"] += 1
            uid = str(row.get("user_id", "")).strip()
            if not uid:
                stats["empty_uid_rows"] += 1
                continue
            if uid in seen:
                stats["duplicate_uid_rows"] += 1
                continue
            seen.add(uid)
            uids.append(uid)
            if len(uids) >= limit:
                stats["limit_reached"] = True
                break
    stats["accepted_unique_uids"] = len(uids)
    return uids, stats


def _extract_commenter_user_ids_with_stats(unified_path: str, limit: int = 500) -> tuple[list[str], dict]:
    uids, seen = [], set()
    stats = {
        "unified_path": unified_path,
        "total_lines": 0,
        "json_error_lines": 0,
        "comment_rows": 0,
        "non_comment_rows": 0,
        "empty_uid_rows": 0,
        "duplicate_uid_rows": 0,
        "accepted_unique_uids": 0,
        "limit_reached": False,
    }
    with open(unified_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            stats["total_lines"] += 1
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                stats["json_error_lines"] += 1
                continue
            if row.get("source_type") != "comment":
                stats["non_comment_rows"] += 1
                continue
            stats["comment_rows"] += 1
            uid = str(row.get("user_id", "")).strip()
            if not uid:
                stats["empty_uid_rows"] += 1
                continue
            if uid in seen:
                stats["duplicate_uid_rows"] += 1
                continue
            seen.add(uid)
            uids.append(uid)
            if len(uids) >= limit:
                stats["limit_reached"] = True
                break
    stats["accepted_unique_uids"] = len(uids)
    return uids, stats


def _fetch_hot_search_keywords(cookie: str, limit: int = 10) -> list[str]:
    req = UrlRequest(
        "https://weibo.com/ajax/side/hotSearch",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://weibo.com/",
            "Cookie": cookie.strip(),
        },
    )
    with urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read().decode("utf-8", errors="replace"))
    realtime = (data.get("data") or {}).get("realtime") or []
    words = []
    for item in realtime:
        word = (item.get("word") or "").strip()
        if word:
            words.append(word)
        if len(words) >= int(limit):
            break
    return words


def _write_hot_search_debug_file(output_dir: str, job_id: str, debug_payload: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    debug_path = Path(output_dir) / f"hot_search_debug_{ts}_{job_id[:8]}.json"
    with debug_path.open("wt", encoding="utf-8") as f:
        json.dump(debug_payload, f, ensure_ascii=False, indent=2)
    return str(debug_path)


def _write_hot_search_summary_file(output_dir: str, summary_latest_path: str, job_id: str, summary_payload: dict) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    summary_path = Path(output_dir) / f"hot_search_keyword_user_summary_{ts}_{job_id[:8]}.json"
    with summary_path.open("wt", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    with open(summary_latest_path, "wt", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)
    return str(summary_path)


def _paired_user_aggregate_path(unified_path: str) -> str:
    """pipelines.JsonWriterPipeline: unified_{tag}.jsonl -> user_aggregate_{tag}.json"""
    basename = os.path.basename(unified_path)
    if not basename.startswith("unified_") or not basename.endswith(".jsonl"):
        return ""
    tag = basename[len("unified_") : -len(".jsonl")]
    return os.path.join(os.path.dirname(unified_path), f"user_aggregate_{tag}.json")


def _cleanup_hot_search_intermediate_outputs(unified_paths: list[str]) -> tuple[int, int]:
    """Remove unified jsonl and matching user_aggregate from this hot-search crawl only."""
    removed_unified = 0
    removed_aggregate = 0
    seen: set[str] = set()
    for p in {x for x in unified_paths if x}:
        if p in seen:
            continue
        seen.add(p)
        if os.path.isfile(p):
            try:
                os.remove(p)
                removed_unified += 1
            except OSError:
                pass
        agg = _paired_user_aggregate_path(p)
        if agg and os.path.isfile(agg):
            try:
                os.remove(agg)
                removed_aggregate += 1
            except OSError:
                pass
    return removed_unified, removed_aggregate


def cleanup_paired_unified_and_aggregate(unified_paths: list[str]) -> tuple[int, int]:
    """Public alias: remove unified + paired user_aggregate (e.g. after pipeline train/infer)."""
    return _cleanup_hot_search_intermediate_outputs(unified_paths)


def run_hot_search_sample_job(
    *,
    job_id: str,
    cookie: str,
    hotword_limit: int,
    users_per_keyword: int,
    total_uid_limit: int,
    speed_profile: dict,
    output_dir: str,
    cookie_path: str,
    summary_latest_path: str,
    weibo_dir: str,
    python_bin: str,
    run_cmd,
    append_job_log,
    update_job,
):
    try:
        update_job(job_id, status="running")
        with open(cookie_path, "wt", encoding="utf-8") as f:
            f.write(cookie.strip())
        append_job_log(job_id, "Cookie updated.")
        hotwords = _fetch_hot_search_keywords(cookie, limit=int(hotword_limit))
        if not hotwords:
            raise RuntimeError("hot search returned empty")

        before_job_unified = _list_unified_files(output_dir)
        before_job_aggregate = _list_user_aggregate_files(output_dir)

        env_base = os.environ.copy()
        env_base["WEIBO_SPLIT_BY_HOUR"] = "false"
        env_base["WEIBO_MAX_PAGES"] = str(speed_profile["keyword_max_pages"])
        env_base["WEIBO_DOWNLOAD_DELAY"] = str(speed_profile["download_delay"])
        env_base["WEIBO_CONCURRENT_REQUESTS"] = str(speed_profile["concurrent_requests"])
        env_base["WEIBO_RETRY_TIMES"] = str(speed_profile["retry_times"])
        end = datetime.now()
        start = end - timedelta(hours=2)
        env_base["WEIBO_START_TIME"] = start.strftime("%Y-%m-%d %H:%M:%S")
        env_base["WEIBO_END_TIME"] = end.strftime("%Y-%m-%d %H:%M:%S")

        all_uids, seen, stats = [], set(), []
        debug_items = []
        for idx, kw in enumerate(hotwords, start=1):
            env = dict(env_base)
            env["WEIBO_KEYWORDS"] = kw
            before = _list_unified_files(output_dir)
            run_cmd(job_id, [python_bin, "run_spider.py", "tweet_by_keyword"], weibo_dir, env)
            after = _list_unified_files(output_dir)
            unified_kw = _pick_new_unified(before, after)
            tweet_ids = _extract_tweet_ids_from_unified(unified_kw, limit=int(speed_profile["tweets_per_keyword"])) if unified_kw else []
            tweet_authors = []
            tweet_author_stats = {}
            if unified_kw:
                tweet_authors, tweet_author_stats = _extract_user_ids_by_source_with_stats(unified_kw, source_type="tweet", limit=500)
            commenters = []
            comment_extract_stats = {}
            if tweet_ids:
                env2 = dict(env_base)
                env2["WEIBO_TWEET_IDS"] = ",".join(tweet_ids)
                before_c = _list_unified_files(output_dir)
                run_cmd(job_id, [python_bin, "run_spider.py", "comment"], weibo_dir, env2)
                after_c = _list_unified_files(output_dir)
                unified_c = _pick_new_unified(before_c, after_c)
                if unified_c:
                    commenters, comment_extract_stats = _extract_commenter_user_ids_with_stats(unified_c, limit=1000)
            candidate_pool, pool_seen = [], set()
            for uid in tweet_authors + commenters:
                if uid not in pool_seen:
                    pool_seen.add(uid)
                    candidate_pool.append(uid)
            pick_n = min(max(0, int(users_per_keyword)), len(candidate_pool))
            picked = random.sample(candidate_pool, k=pick_n) if pick_n > 0 else []
            added_count = 0
            dup_in_global_seen = 0
            for uid in picked:
                if uid not in seen:
                    seen.add(uid)
                    all_uids.append(uid)
                    added_count += 1
                else:
                    dup_in_global_seen += 1
            stats.append({
                "keyword": kw,
                "tweet_ids": len(tweet_ids),
                "commenter_candidates": len(commenters),
                "tweet_author_candidates": len(tweet_authors),
                "interaction_candidates": len(candidate_pool),
                "picked": len(picked),
                "newly_added": added_count,
                "picked_but_duplicate_globally": dup_in_global_seen,
            })
            debug_items.append({
                "keyword_index": idx,
                "keyword": kw,
                "tweet_unified_file": unified_kw,
                "tweet_author_stats": tweet_author_stats,
                "comment_extract_stats": comment_extract_stats,
                "tweet_ids_collected": len(tweet_ids),
                "tweet_authors_unique": len(tweet_authors),
                "commenter_candidates_unique": len(commenters),
                "interaction_candidates_unique": len(candidate_pool),
                "picked_requested": int(users_per_keyword),
                "picked_actual": len(picked),
                "picked_user_ids": picked,
                "added_to_global_unique": added_count,
                "picked_but_duplicate_globally": dup_in_global_seen,
                "global_unique_after_keyword": len(all_uids),
            })
            append_job_log(
                job_id,
                f"[{idx}/{len(hotwords)}] {kw}: tweets={len(tweet_ids)} authors={len(tweet_authors)} commenters={len(commenters)} "
                f"pool={len(candidate_pool)} picked={len(picked)} new={added_count} dup_global={dup_in_global_seen} total={len(all_uids)}",
            )
            if total_uid_limit and len(all_uids) >= int(total_uid_limit):
                break
            sleep_sec = float(speed_profile["sleep_between_keywords_sec"])
            if sleep_sec and idx < len(hotwords):
                time.sleep(max(0.0, sleep_sec))

        if total_uid_limit:
            all_uids = all_uids[: int(total_uid_limit)]
        random.shuffle(all_uids)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "hot_search_user_ids_latest.txt")
        with open(output_file, "wt", encoding="utf-8") as f:
            for uid in all_uids:
                f.write(str(uid) + "\n")
        debug_payload = {
            "job_id": job_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "params": {
                "hotword_limit": int(hotword_limit),
                "users_per_keyword": int(users_per_keyword),
                "total_uid_limit": int(total_uid_limit),
                "keyword_max_pages": int(speed_profile["keyword_max_pages"]),
                "sleep_between_keywords_sec": float(speed_profile["sleep_between_keywords_sec"]),
                "download_delay": float(speed_profile["download_delay"]),
                "concurrent_requests": int(speed_profile["concurrent_requests"]),
                "retry_times": int(speed_profile["retry_times"]),
                "speed_mode": speed_profile.get("speed_mode", "steady"),
            },
            "hotwords": hotwords,
            "summary": {"total_unique_uids": len(all_uids), "output_file": output_file},
            "by_keyword": debug_items,
        }
        debug_file = _write_hot_search_debug_file(output_dir, job_id, debug_payload)
        summary_payload = {
            "job_id": job_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "hotword_count": len(hotwords),
            "total_unique_users": len(all_uids),
            "keywords": [
                {
                    "keyword_index": item["keyword_index"],
                    "keyword": item["keyword"],
                    "picked_actual": item["picked_actual"],
                    "added_to_global_unique": item["added_to_global_unique"],
                    "interaction_candidates_unique": item["interaction_candidates_unique"],
                    "picked_user_ids": item["picked_user_ids"],
                }
                for item in debug_items
            ],
        }
        summary_file = _write_hot_search_summary_file(output_dir, summary_latest_path, job_id, summary_payload)
        removed_unified, removed_aggregate = cleanup_hot_search_new_crawl_artifacts(
            output_dir, before_job_unified, before_job_aggregate
        )
        append_job_log(job_id, f"Debug log saved: {debug_file}")
        append_job_log(job_id, f"Keyword-user summary saved: {summary_file}")
        append_job_log(
            job_id,
            f"Removed intermediate crawl files (snapshot): unified={removed_unified}, user_aggregate={removed_aggregate}",
        )
        update_job(
            job_id,
            status="completed",
            completed_at=datetime.now().isoformat(timespec="seconds"),
            result={
                "user_ids": all_uids,
                "total_unique": len(all_uids),
                "stats": stats,
                "output_file": output_file,
                "debug_file": debug_file,
                "summary_file": summary_file,
                "removed_intermediate_unified": removed_unified,
                "removed_intermediate_user_aggregate": removed_aggregate,
            },
        )
    except Exception as exc:
        update_job(job_id, status="failed", error=str(exc), completed_at=datetime.now().isoformat(timespec="seconds"))
        append_job_log(job_id, f"Failed: {exc}")
