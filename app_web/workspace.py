"""Discover local artifacts so the UI can default inputs after restart (UID list, crawl JSON, cookie file)."""
from __future__ import annotations

import glob
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
    kmeans_multilabel_path = os.path.join(output_dir, "kmeans_multilabel_users.jsonl")
    kmeans_multilabel_meta_path = os.path.join(output_dir, "kmeans_multilabel_meta.json")

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
        "kmeans_multilabel": file_probe(project_dir, kmeans_multilabel_path),
        "kmeans_multilabel_meta": file_probe(project_dir, kmeans_multilabel_meta_path),
    }


def _abs_under_project(project_dir: str, rel_or_abs: str) -> str:
    p = (rel_or_abs or "").strip().replace("\\", "/")
    if not p:
        return ""
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(project_dir, p))


def multilabel_data_source_rows(project_dir: str, output_dir: str, meta: dict) -> list[dict]:
    """
    多标签页「数据来源」列表：与 meta 中记录的路径对齐，便于对照画像页。
    每项含 label + file_probe 字段（path / exists / size_bytes / mtime_iso）。
    """
    rows: list[dict] = []
    p_jsonl = os.path.join(output_dir, "kmeans_multilabel_users.jsonl")
    rows.append({**file_probe(project_dir, p_jsonl), "label": "多标签用户表"})
    p_meta = os.path.join(output_dir, "kmeans_multilabel_meta.json")
    rows.append({**file_probe(project_dir, p_meta), "label": "多标签元数据"})

    feat = (meta.get("features") or "").strip()
    if feat:
        ap = _abs_under_project(project_dir, feat)
        if ap:
            rows.append({**file_probe(project_dir, ap), "label": "聚类用特征矩阵"})

    txt = (meta.get("texts_jsonl_for_keywords") or "").strip()
    if txt:
        ap = _abs_under_project(project_dir, txt)
        if ap:
            rows.append({**file_probe(project_dir, ap), "label": "簇关键词所用清洗文本"})

    km = (meta.get("kmeans_model_path") or "").strip()
    if km:
        ap = _abs_under_project(project_dir, km)
        if ap:
            rows.append({**file_probe(project_dir, ap), "label": "预训练 KMeans 模型"})

    unified_paths = glob.glob(os.path.join(output_dir, "unified_*.jsonl"))
    unified_paths.sort(key=os.path.getmtime, reverse=True)
    if unified_paths:
        rows.append(
            {
                **file_probe(project_dir, unified_paths[0]),
                "label": "原始微博抓取（当前最新 unified_*.jsonl；详情页会扫描全部）",
            }
        )

    return rows
