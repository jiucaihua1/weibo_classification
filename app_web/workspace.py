"""Discover local artifacts so the UI can default inputs after restart (UID list, crawl JSON, cookie file)."""
from __future__ import annotations

import os
from datetime import datetime


def _mtime_iso(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    return datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")


def rel_posix(project_dir: str, path: str) -> str:
    try:
        return os.path.relpath(path, project_dir).replace("\\", "/")
    except ValueError:
        return path.replace("\\", "/")


def load_hot_uids_csv(output_dir: str) -> tuple[str, int]:
    path = os.path.join(output_dir, "hot_search_user_ids_latest.txt")
    if not os.path.isfile(path):
        return "", 0
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        ids = [ln.strip() for ln in f if ln.strip()]
    return ",".join(ids), len(ids)


def load_cookie_text(cookie_path: str, max_bytes: int = 512_000) -> str:
    if not os.path.isfile(cookie_path):
        return ""
    with open(cookie_path, "rt", encoding="utf-8", errors="replace") as f:
        return f.read(max_bytes).strip()


def file_probe(project_dir: str, abs_path: str) -> dict:
    exists = os.path.isfile(abs_path)
    return {
        "path": rel_posix(project_dir, abs_path),
        "exists": bool(exists),
        "size_bytes": os.path.getsize(abs_path) if exists else 0,
        "mtime_iso": _mtime_iso(abs_path) if exists else "",
    }


def get_workspace_state(
    project_dir: str,
    output_dir: str,
    cookie_path: str,
    profile_path: str,
    crawl_json_path: str,
    *,
    fallback_demo_uid: str = "1087770692",
) -> dict:
    hot_csv, hot_n = load_hot_uids_csv(output_dir)
    cookie_full = load_cookie_text(cookie_path)
    hot_abs = os.path.join(output_dir, "hot_search_user_ids_latest.txt")
    train_rel = rel_posix(project_dir, crawl_json_path)

    cleaned_path = os.path.join(output_dir, "cleaned_user_texts.jsonl")
    user_features_path = os.path.join(output_dir, "user_features_text2vec.npy")
    user_ids_path = os.path.join(output_dir, "user_ids_text2vec.pkl")

    return {
        "default_user_id": hot_csv if hot_csv else fallback_demo_uid,
        "hot_uid_count": hot_n,
        "hot_uids_file": file_probe(project_dir, hot_abs),
        "cookie_saved": bool(cookie_full),
        "cookie_char_len": len(cookie_full),
        "default_cookie": cookie_full,
        "crawl_bundle": file_probe(project_dir, crawl_json_path),
        "cleaned_user_texts": file_probe(project_dir, cleaned_path),
        "user_features_text2vec": file_probe(project_dir, user_features_path),
        "user_ids_text2vec": file_probe(project_dir, user_ids_path),
        "profiles_file": file_probe(project_dir, profile_path),
        "train_input_default": train_rel,
    }
