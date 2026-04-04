"""Delete subset11s of files under output/ by pipeline step."""
from __future__ import annotations

import os
import shutil
from typing import Any


def cleanup_output_full(output_dir: str, *, keep_ml_artifacts: bool = True) -> dict[str, Any]:
    """Remove everything under output_dir except optional ml_artifacts (legacy one-click)."""
    removed_files, removed_dirs = 0, 0
    if not os.path.isdir(output_dir):
        return {"status": "ok", "mode": "full", "removed_files": 0, "removed_dirs": 0, "breakdown": {}}
    for name in os.listdir(output_dir):
        path = os.path.join(output_dir, name)
        if keep_ml_artifacts and name == "ml_artifacts" and os.path.isdir(path):
            continue
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for fn in files:
                    try:
                        os.remove(os.path.join(root, fn))
                        removed_files += 1
                    except OSError:
                        pass
                for dn in dirs:
                    try:
                        os.rmdir(os.path.join(root, dn))
                        removed_dirs += 1
                    except OSError:
                        pass
            try:
                os.rmdir(path)
                removed_dirs += 1
            except OSError:
                pass
        else:
            try:
                os.remove(path)
                removed_files += 1
            except OSError:
                pass
    return {"status": "ok", "mode": "full", "removed_files": removed_files, "removed_dirs": removed_dirs, "breakdown": {}}


def cleanup_output_selective(
    output_dir: str,
    *,
    project_dir: str = "",
    del_crawl_intermediate: bool = False,
    del_crawl_bundle: bool = False,
    del_user_infos: bool = False,
    del_profiles: bool = False,
    del_ml_artifacts: bool = False,
    del_input_user_ids: bool = False,
) -> dict[str, Any]:
    breakdown: dict[str, int] = {}
    removed_files, removed_dirs = 0, 0

    if not os.path.isdir(output_dir):
        return {"status": "ok", "mode": "selective", "removed_files": 0, "removed_dirs": 0, "breakdown": {}}

    def rm_file(path: str) -> bool:
        nonlocal removed_files
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed_files += 1
                return True
            except OSError:
                return False
        return False

    if del_input_user_ids and project_dir:
        uid_path = os.path.join(project_dir, "input", "user_ids.txt")
        breakdown["input_user_ids"] = 1 if rm_file(uid_path) else 0

    if del_crawl_intermediate:
        n = 0
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if not os.path.isfile(path):
                continue
            if (name.startswith("unified_") and name.endswith(".jsonl")) or (
                name.startswith("user_aggregate_") and name.endswith(".json")
            ):
                if rm_file(path):
                    n += 1
        breakdown["crawl_intermediate"] = n

    if del_crawl_bundle:
        n = 0
        for name in os.listdir(output_dir):
            path = os.path.join(output_dir, name)
            if not os.path.isfile(path):
                continue
            if name.startswith("weibo_crawl_") and name.endswith(".json"):
                if rm_file(path):
                    n += 1
        breakdown["crawl_bundle"] = n

    if del_user_infos:
        path = os.path.join(output_dir, "user_infos.json")
        breakdown["user_infos"] = 1 if rm_file(path) else 0

    if del_profiles:
        path = os.path.join(output_dir, "user_interest_profiles.json")
        breakdown["profiles"] = 1 if rm_file(path) else 0

    if del_ml_artifacts:
        path = os.path.join(output_dir, "ml_artifacts")
        if os.path.isdir(path):
            try:
                shutil.rmtree(path, ignore_errors=False)
                removed_dirs += 1
                breakdown["ml_artifacts"] = 1
            except OSError:
                breakdown["ml_artifacts"] = 0
        else:
            breakdown["ml_artifacts"] = 0

    return {
        "status": "ok",
        "mode": "selective",
        "removed_files": removed_files,
        "removed_dirs": removed_dirs,
        "breakdown": breakdown,
    }
