import glob
import json
import os
import re
import subprocess
import threading
import uuid
import time
import tempfile
from collections import Counter
from datetime import datetime

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app_web.crawl_bundle import build_crawl_bundle, cleanup_post_crawl_artifacts, write_crawl_bundle
from app_web.output_cleanup import cleanup_output_full, cleanup_output_selective
from app_web.weibo_sources_service import load_user_weibo_posts, normalize_uid
from app_web.workspace import (
    get_workspace_state,
    rel_posix,
    resolve_prep_source_default,
    resolve_train_input_default,
)
from app_web.hot_search_service import (
    get_hot_search_speed_profile as service_get_hot_search_speed_profile,
    latest_hot_search_debug_file as service_latest_hot_search_debug_file,
    run_hot_search_sample_job as service_run_hot_search_sample_job,
)
from app_pipeline.cluster_llm_labels import (
    generate_cluster_llm_labels,
    generate_tweet_topic_cluster_llm,
    merge_llm_labels_into_meta,
    multilabel_jsonl_fingerprint,
    resolve_deepseek_api_key,
    try_load_cached_cluster_llm,
)
from app_pipeline.kmeans_prep import export_kmeans_tweets_jsonl
from app_pipeline.infer import infer_tweet_topic_multilabel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
PROFILE_PATH = os.path.join(PROJECT_DIR, "output", "user_interest_profiles.json")
USER_INFO_PATH = os.path.join(PROJECT_DIR, "output", "user_infos.json")
COOKIE_PATH = os.path.join(PROJECT_DIR, "weibospider", "cookie.txt")
WEIBO_DIR = os.path.join(PROJECT_DIR, "weibospider")
PYTHON_BIN = os.path.join(PROJECT_DIR, ".venv", "Scripts", "python.exe")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
HOT_SUMMARY_LATEST_PATH = os.path.join(OUTPUT_DIR, "hot_search_keyword_user_summary_latest.json")
WEIBO_CRAWL_LATEST_JSON = os.path.join(OUTPUT_DIR, "weibo_crawl_latest.json")
KMEANS_TWEETS_INPUT_JSONL = os.path.join(OUTPUT_DIR, "kmeans_tweets_input.jsonl")
KMEANS_MULTILABEL_JSONL = os.path.join(OUTPUT_DIR, "kmeans_multilabel_users.jsonl")
KMEANS_MULTILABEL_META = os.path.join(OUTPUT_DIR, "kmeans_multilabel_meta.json")

app = FastAPI(title="Weibo Interest Dashboard", version="0.2.0")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
jobs = {}
jobs_lock = threading.Lock()


def _page_context(request: Request, nav_active: str, **extra):
    ws = get_workspace_state(PROJECT_DIR, OUTPUT_DIR, COOKIE_PATH, PROFILE_PATH, WEIBO_CRAWL_LATEST_JSON)
    ws["profiles_count"] = len(load_profiles_effective())
    ctx = {"request": request, "nav_active": nav_active, "workspace": ws}
    ctx.update(extra)
    return ctx


def latest_unified_file() -> str:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "unified_*.jsonl")), key=os.path.getmtime, reverse=True)
    return files[0] if files else ""


def load_profiles():
    if not os.path.exists(PROFILE_PATH):
        return []
    with open(PROFILE_PATH, "rt", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def load_user_infos() -> dict:
    if not os.path.exists(USER_INFO_PATH):
        return {}
    with open(USER_INFO_PATH, "rt", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def load_kmeans_multilabel() -> list:
    if not os.path.isfile(KMEANS_MULTILABEL_JSONL):
        return []
    rows = []
    with open(KMEANS_MULTILABEL_JSONL, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_kmeans_multilabel_meta() -> dict:
    if not os.path.isfile(KMEANS_MULTILABEL_META):
        return {}
    with open(KMEANS_MULTILABEL_META, "rt", encoding="utf-8", errors="replace") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}


def _cluster_display_names(meta: dict) -> dict[str, str]:
    """优先 DeepSeek 生成的 cluster_llm_labels；否则词典 cluster_display_names；再否则由关键词现场匹配抽屉。"""
    llm = meta.get("cluster_llm_labels")
    if isinstance(llm, dict) and llm:
        return {str(k): str(v) for k, v in llm.items()}
    raw = meta.get("cluster_display_names")
    if isinstance(raw, dict) and raw:
        return {str(k): str(v) for k, v in raw.items()}
    kw = meta.get("cluster_keywords")
    if isinstance(kw, dict) and kw:
        from app_pipeline.cluster_category_names import map_cluster_keywords_to_names

        return map_cluster_keywords_to_names(kw)
    return {}


def _kmeans_multilabel_by_user() -> dict:
    out: dict = {}
    for row in load_kmeans_multilabel():
        uid = str(row.get("user_id", "")).strip()
        if uid:
            out[uid] = row
    return out


def _kmeans_ml_view(row: dict | None) -> dict | None:
    """主簇、按距离排序的次簇，供画像表与详情对照来源判断。"""
    if not row:
        return None
    primary = row.get("primary_cluster")
    all_ids = list(row.get("multilabel_cluster_ids") or [])
    dists = row.get("distances_to_centroids") or []

    def dist_for(cid):
        try:
            i = int(cid)
            if 0 <= i < len(dists):
                return float(dists[i])
        except (TypeError, ValueError):
            pass
        return float("inf")

    secondary = [x for x in all_ids if x != primary]
    secondary_sorted = sorted(secondary, key=dist_for)
    return {
        "primary": primary,
        "secondary": secondary_sorted,
        "multilabel_all": all_ids,
        "min_distance": row.get("min_distance"),
        "threshold": row.get("threshold"),
        "distances_to_centroids": dists,
    }


def load_cluster_llm_topic_by_label() -> dict:
    """cluster_llm_topic.json：核心标签 -> {persona, perception, summary}，供详情页展示。"""
    p = os.path.join(OUTPUT_DIR, "ml_artifacts", "cluster_llm_topic.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "rt", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    clusters = data.get("clusters") if isinstance(data, dict) else None
    if not isinstance(clusters, dict):
        return {}
    by_label: dict = {}
    for v in clusters.values():
        if not isinstance(v, dict):
            continue
        lab = str(v.get("label", "")).strip()
        if not lab:
            continue
        by_label[lab] = {
            "persona": str(v.get("persona", "") or ""),
            "perception": str(v.get("perception", "") or ""),
            "summary": str(v.get("summary", "") or ""),
        }
    return by_label


def load_active_labels() -> list[str]:
    """读取动态标签维度（由训练期写入 ml_artifacts/active_labels.json）。"""
    p = os.path.join(OUTPUT_DIR, "ml_artifacts", "active_labels.json")
    if not os.path.isfile(p):
        return []
    try:
        with open(p, "rt", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    raw = data.get("active_labels") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for x in raw:
        s = str(x or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _predict_personal_interest_from_local(uid: str) -> dict:
    """
    用当前已训练产物对单个 UID 做即时推断：
    1) 从 unified/bundle 收集该 UID 原始微博；
    2) 构造临时 jsonl；
    3) 调用 infer_tweet_topic_multilabel；
    4) 返回该用户画像。
    """
    uid_key = normalize_uid(uid)
    if not uid_key:
        raise ValueError("UID 为空或格式无效")

    model_dir = os.path.join(OUTPUT_DIR, "ml_artifacts")
    required = (
        "training_pipeline.json",
        "kmeans_tweet_model.pkl",
        "tweet_pca_64.pkl",
        "gbdt_multilabel.pkl",
        "cluster_label_map.json",
        "active_labels.json",
    )
    missing = [name for name in required if not os.path.isfile(os.path.join(model_dir, name))]
    if missing:
        raise RuntimeError(
            "模型产物不完整，请先在训练页点击「生成模型（全流程）」。缺失: " + ", ".join(missing)
        )

    posts, _files_hit = load_user_weibo_posts(uid_key, OUTPUT_DIR, limit=300)
    if not posts:
        raise RuntimeError("本地未找到该 UID 的原始微博，请先抓取并确认 unified/bundle 内存在该 UID。")

    rows = []
    for p in posts:
        txt = str(p.get("text") or "").strip()
        if not txt:
            continue
        row = {
            "user_id": uid_key,
            "item_id": str(p.get("item_id") or "").strip(),
            "source_type": "tweet",
            "text": txt,
            "created_at": str(p.get("created_at") or "").strip(),
            "raw": {
                "content": txt,
                "url": str(p.get("url") or "").strip(),
                "created_at": str(p.get("created_at") or "").strip(),
            },
        }
        rows.append(row)
    if not rows:
        raise RuntimeError("该 UID 原始微博均为空文本，无法推断。")

    tmp_dir = os.path.join(OUTPUT_DIR, "_tmp_personal_interest")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_in = tempfile.NamedTemporaryFile(
        mode="wt",
        encoding="utf-8",
        delete=False,
        suffix=".jsonl",
        dir=tmp_dir,
    )
    tmp_out = os.path.join(tmp_dir, f"personal_interest_{uid_key}_{uuid.uuid4().hex}.json")
    try:
        with tmp_in as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        profiles = infer_tweet_topic_multilabel(tmp_in.name, model_dir, tmp_out)
        target = None
        for p in profiles:
            if normalize_uid(p.get("user_id")) == uid_key:
                target = p
                break
        if not target and profiles:
            target = profiles[0]
        if not isinstance(target, dict):
            raise RuntimeError("推断未返回有效结果，请检查模型与输入数据。")
        target["source_post_count"] = len(rows)
        return target
    finally:
        for p in (tmp_in.name, tmp_out):
            if p and os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


def _synthetic_profile_from_multilabel_row(row: dict) -> dict:
    """多标签 jsonl 一行 → 与推断画像字段对齐的占位记录（无推断时用）。"""
    uid = str(row.get("user_id", "")).strip()
    ml = _kmeans_ml_view(row)
    prim = ml["primary"] if ml else row.get("primary_cluster")
    label = f"簇{prim}" if prim is not None else "—"
    return {
        "user_id": uid,
        "top_interest": label,
        "sample_size": "—",
        "source_stats": "仅 KMeans 多标签（未跑推断画像）",
        "cluster_id": prim,
        "interest_scores": "（未生成）",
        "evidence": [],
    }


def load_profiles_effective() -> list:
    """优先读取 user_interest_profiles.json；若文件不存在或列表为空，则用 kmeans_multilabel_users.jsonl 生成列表。"""
    raw = load_profiles()
    if raw:
        return raw
    out = []
    for row in load_kmeans_multilabel():
        uid = str(row.get("user_id", "")).strip()
        if not uid:
            continue
        out.append(_synthetic_profile_from_multilabel_row(row))
    return out


def _list_unified_files() -> set[str]:
    if not os.path.exists(OUTPUT_DIR):
        return set()
    out: set[str] = set()
    for n in os.listdir(OUTPUT_DIR):
        if n.startswith("unified_") and n.endswith(".jsonl"):
            p = os.path.join(OUTPUT_DIR, n)
            out.add(os.path.normcase(os.path.normpath(os.path.abspath(p))))
    return out


def _pick_new_unified(before: set[str], after: set[str]) -> str:
    candidates = list(after - before)
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _extract_user_infos_from_unified(unified_path: str, user_ids: list[str]) -> dict:
    targets = {str(x).strip() for x in user_ids if str(x).strip()}
    if not targets:
        return {}
    result = {}
    with open(unified_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = str(row.get("user_id", "")).strip()
            if not uid or uid not in targets or uid in result:
                continue
            raw = row.get("raw") or {}
            if "nick_name" in raw or "followers_count" in raw or "avatar_hd" in raw:
                result[uid] = raw
            if len(result) >= len(targets):
                break
    return result


def _parse_user_ids(user_ids_raw: str) -> list[str]:
    return [x.strip() for x in (user_ids_raw or "").split(",") if x.strip()]


def _ordered_unique_user_ids(user_ids_raw: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in (user_ids_raw or "").split(","):
        x = x.strip()
        if not x or x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _new_job_record(job_id: str, *, user_id: str = "") -> dict:
    ts = datetime.now().isoformat(timespec="seconds")
    return {
        "job_id": job_id,
        "status": "queued",
        "error": "",
        "created_at": ts,
        "completed_at": "",
        "user_id": user_id,
        "logs": [],
        "result": {},
        "progress": 0,
        "progress_label": "排队中…",
    }


def _append_job_log(job_id: str, message: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            return
        job["logs"].append(f"{datetime.now().strftime('%H:%M:%S')} {message}")
        if len(job["logs"]) > 200:
            job["logs"] = job["logs"][-200:]


def _update_job(job_id: str, **kwargs):
    with jobs_lock:
        job = jobs.get(job_id)
        if job:
            job.update(kwargs)


def _latest_hot_search_debug_file() -> str:
    return service_latest_hot_search_debug_file(OUTPUT_DIR)


def _run_cmd(job_id: str, cmd: list[str], cwd: str, env: dict):
    _append_job_log(job_id, f"[RUN] {' '.join(cmd)}")
    raw_timeout = (env or {}).get("WEB_CMD_TIMEOUT_SEC") or os.environ.get("WEB_CMD_TIMEOUT_SEC")
    timeout_sec = None
    if raw_timeout not in (None, ""):
        try:
            timeout_sec = float(raw_timeout)
            if timeout_sec <= 0:
                timeout_sec = None
        except (TypeError, ValueError):
            timeout_sec = None
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        _append_job_log(job_id, f"子进程超时（WEB_CMD_TIMEOUT_SEC={timeout_sec}），已终止")
        raise RuntimeError(f"命令超时（>{timeout_sec}s）: {' '.join(cmd)}") from None
    if completed.stdout:
        for line in completed.stdout.splitlines()[-20:]:
            _append_job_log(job_id, line)
    if completed.stderr:
        for line in completed.stderr.splitlines()[-20:]:
            _append_job_log(job_id, line)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _run_cmd_stream(
    job_id: str,
    cmd: list[str],
    cwd: str,
    env: dict,
    *,
    progress_updater=None,
):
    """
    Stream stdout/stderr line-by-line into job logs so the UI doesn't look stuck.
    progress_updater(line) -> dict|None (passed to _update_job)
    """
    _append_job_log(job_id, f"[RUN] {' '.join(cmd)}")
    raw_timeout = (env or {}).get("WEB_CMD_TIMEOUT_SEC") or os.environ.get("WEB_CMD_TIMEOUT_SEC")
    timeout_sec = None
    if raw_timeout not in (None, ""):
        try:
            timeout_sec = float(raw_timeout)
            if timeout_sec <= 0:
                timeout_sec = None
        except (TypeError, ValueError):
            timeout_sec = None

    env2 = dict(env or {})
    env2["PYTHONUNBUFFERED"] = "1"

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env2,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None

    try:
        while True:
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                proc.kill()
                _append_job_log(job_id, f"子进程超时（WEB_CMD_TIMEOUT_SEC={timeout_sec}），已终止")
                raise RuntimeError(f"命令超时（>{timeout_sec}s）: {' '.join(cmd)}")

            line = proc.stdout.readline()
            if line:
                line = line.rstrip("\r\n")
                if line:
                    _append_job_log(job_id, line)
                if progress_updater:
                    try:
                        upd = progress_updater(line)
                        if upd:
                            _update_job(job_id, **upd)
                    except Exception:
                        pass
                continue

            # no line available; check if finished
            if proc.poll() is not None:
                break
            time.sleep(0.1)

        # drain remaining output
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            if line:
                _append_job_log(job_id, line)
    finally:
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass

    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _get_hot_search_speed_profile(speed_mode: str) -> dict:
    return service_get_hot_search_speed_profile(speed_mode)


def _hot_search_sample_job(job_id: str, cookie: str, hotword_limit: int, keyword_max_pages: int, sleep_between_keywords_sec: float,
                           tweets_per_keyword: int, commenters_per_keyword: int, total_uid_limit: int,
                           download_delay: float, concurrent_requests: int, retry_times: int):
    speed_profile = {
        "keyword_max_pages": int(keyword_max_pages),
        "sleep_between_keywords_sec": float(sleep_between_keywords_sec),
        "tweets_per_keyword": int(tweets_per_keyword),
        "download_delay": float(download_delay),
        "concurrent_requests": int(concurrent_requests),
        "retry_times": int(retry_times),
        "speed_mode": "custom",
    }
    service_run_hot_search_sample_job(
        job_id=job_id,
        cookie=cookie,
        hotword_limit=int(hotword_limit),
        users_per_keyword=int(commenters_per_keyword),
        total_uid_limit=int(total_uid_limit),
        speed_profile=speed_profile,
        output_dir=OUTPUT_DIR,
        cookie_path=COOKIE_PATH,
        summary_latest_path=HOT_SUMMARY_LATEST_PATH,
        weibo_dir=WEIBO_DIR,
        python_bin=PYTHON_BIN,
        run_cmd=_run_cmd,
        append_job_log=_append_job_log,
        update_job=_update_job,
    )


def _crawl_weibo_job(job_id: str, user_id: str, cookie: str, max_pages: int,
                     download_delay: float, concurrent_requests: int, retry_times: int):
    try:
        _update_job(job_id, status="running", progress=5, progress_label="准备环境…")
        with open(COOKIE_PATH, "wt", encoding="utf-8") as f:
            f.write(cookie.strip())

        env = os.environ.copy()
        env["WEIBO_USER_IDS"] = user_id
        env["WEIBO_CRAWL_TIME_SPAN"] = "false"
        env["WEIBO_MAX_PAGES"] = str(max_pages)
        env["WEIBO_DOWNLOAD_DELAY"] = str(download_delay)
        env["WEIBO_CONCURRENT_REQUESTS"] = str(concurrent_requests)
        env["WEIBO_RETRY_TIMES"] = str(retry_times)

        before_pipeline = _list_unified_files()
        _update_job(job_id, progress=15, progress_label="① 抓取用户资料（user）…")
        _run_cmd(job_id, [PYTHON_BIN, "run_spider.py", "user"], WEIBO_DIR, env)
        after_user = _list_unified_files()
        user_unified = _pick_new_unified(before_pipeline, after_user)
        if user_unified:
            ids = _parse_user_ids(user_id)
            infos_new = _extract_user_infos_from_unified(user_unified, ids)
            if infos_new:
                infos = load_user_infos()
                infos.update({str(k): v for k, v in infos_new.items()})
                os.makedirs(os.path.dirname(USER_INFO_PATH), exist_ok=True)
                with open(USER_INFO_PATH, "wt", encoding="utf-8") as f:
                    json.dump(infos, f, ensure_ascii=False, indent=2)

        _update_job(job_id, progress=45, progress_label="② 抓取用户微博（tweet_by_user_id）…")
        _run_cmd(job_id, [PYTHON_BIN, "run_spider.py", "tweet_by_user_id"], WEIBO_DIR, env)
        # 统一在：抓取结束后把 output 目录下所有 unified_*.jsonl 合并进
        # output/weibo_crawl_latest.json，确保多次抓取会“累积”而不是覆盖/丢用户。
        after_pipeline = _list_unified_files()
        unified = latest_unified_file()
        if not unified:
            raise RuntimeError("No unified file produced after crawl.")
        all_unified_sorted = sorted(list(after_pipeline), key=lambda p: os.path.basename(p))
        ordered_ids = _ordered_unique_user_ids(user_id)
        _update_job(job_id, progress=78, progress_label="③ 合并为 weibo_crawl_latest.json …")
        from app_pipeline.merge_unified_to_bundle import merge_paths

        payload, n_in, n_dup = merge_paths([str(p) for p in all_unified_sorted])
        latest_path, _archive_placeholder = write_crawl_bundle(OUTPUT_DIR, job_id, payload)
        ordered_set = set(ordered_ids)
        n_with_data = sum(
            1
            for u in (payload.get("users") or [])
            if str(u.get("user_id", "")).strip() in ordered_set and u.get("records")
        )
        _append_job_log(job_id, f"Crawl bundle saved: {latest_path}")
        _append_job_log(job_id, f"Merge unified_*.jsonl: lines_read={n_in} duplicates_skipped={n_dup}")
        _append_job_log(job_id, f"Users with at least one record: {n_with_data} / {len(ordered_ids)}")
        _update_job(job_id, progress=92, progress_label="④ 清理中间文件…")
        swept = cleanup_post_crawl_artifacts(OUTPUT_DIR)
        _append_job_log(
            job_id,
            "保留 output/unified_*.jsonl（未删除）。"
            f" 其余清理: user_aggregate={swept['user_aggregate']} "
            f"extra_weibo_crawl_json={swept['weibo_crawl_extra']}",
        )
        _update_job(
            job_id,
            status="completed",
            progress=100,
            progress_label="完成",
            completed_at=datetime.now().isoformat(timespec="seconds"),
            result={
                "output_file": latest_path,
                "archive_file": "",
                "n_users_requested": len(ordered_ids),
                "n_users_with_records": n_with_data,
                "swept_unified": swept["unified"],
                "swept_user_aggregate": swept["user_aggregate"],
                "swept_extra_weibo_json": swept["weibo_crawl_extra"],
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            progress_label="失败",
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        _append_job_log(job_id, f"Failed: {exc}")


def _resolve_training_input(path: str) -> str:
    """空路径：优先 output/kmeans_tweets_input.jsonl（步骤① 已导出），否则 bundle / unified。"""
    p = (path or "").strip()
    if not p:
        rel = resolve_train_input_default(PROJECT_DIR, OUTPUT_DIR, WEIBO_CRAWL_LATEST_JSON)
        return os.path.normpath(os.path.join(PROJECT_DIR, rel.replace("/", os.sep)))
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(PROJECT_DIR, p))


def _resolve_prep_source(path: str) -> str:
    """步骤① 的原始抓取路径：空则 bundle → 最新 unified。"""
    p = (path or "").strip()
    if not p:
        rel = resolve_prep_source_default(PROJECT_DIR, OUTPUT_DIR, WEIBO_CRAWL_LATEST_JSON)
        return os.path.normpath(os.path.join(PROJECT_DIR, rel.replace("/", os.sep)))
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(PROJECT_DIR, p))


def _cluster_text2vec_job(job_id: str, input_path: str, k_min: int, k_max: int):
    try:
        _update_job(job_id, status="running", progress=8, progress_label="加载特征与搜索 k…")
        path = _resolve_training_input(input_path)
        if not os.path.isfile(path):
            raise RuntimeError(f"bundle not found: {path}")

        env = os.environ.copy()
        model_dir = os.path.join("output", "ml_artifacts")
        _update_job(job_id, progress=25, progress_label="① PCA(64)+K 搜索+KMeans…")

        def _progress_ksearch(line: str):
            if not line.startswith("[KSEARCH]"):
                return None
            m = re.search(r"k=(\d+)", line)
            if not m:
                return None
            k = int(m.group(1))
            span = max(1, int(k_max) - int(k_min) + 1)
            frac = (k - int(k_min) + 1) / float(span)
            progress = 25 + int(max(0.0, min(1.0, frac)) * 45)
            return {"progress": max(25, min(72, progress)), "progress_label": line.replace("[KSEARCH] ", "")}

        _run_cmd_stream(
            job_id,
            [
                PYTHON_BIN,
                "-m",
                "app_pipeline.train_tweet_topic",
                "--input",
                path,
                "--output-dir",
                model_dir,
                "--k-min",
                str(k_min),
                "--k-max",
                str(k_max),
                "--viz-only",
            ],
            PROJECT_DIR,
            env,
            progress_updater=_progress_ksearch,
        )

        # 读取聚类可视化数据（若存在）
        viz_data = None
        viz_path = os.path.join(model_dir, "cluster_viz_data.json")
        if os.path.isfile(viz_path):
            try:
                with open(viz_path, "rt", encoding="utf-8", errors="replace") as f:
                    viz_data = json.load(f)
            except Exception:
                viz_data = None

        _update_job(
            job_id,
            status="completed",
            progress=100,
            progress_label="聚类完成",
            completed_at=datetime.now().isoformat(timespec="seconds"),
            result={
                "model_dir": model_dir,
                "k_search_note": "tweet-topic: PCA(64)+silhouette(k)+KMeans（仅聚类+可视化）",
                "cluster_viz_data": viz_data,
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            progress_label="失败",
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        _append_job_log(job_id, f"Failed: {exc}")


ML_ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "ml_artifacts")


def _tweet_topic_cluster_llm_job(job_id: str, force: bool) -> None:
    def _done_ts() -> str:
        return datetime.now().isoformat(timespec="seconds")

    try:
        _update_job(job_id, status="running", progress=5, progress_label="准备 DeepSeek（推文簇深度命名）…")
        model_s = (os.environ.get("DEEPSEEK_MODEL", "") or "deepseek-chat").strip()
        base_url = (os.environ.get("DEEPSEEK_BASE_URL", "") or "https://api.deepseek.com").strip()

        viz_path = os.path.join(ML_ARTIFACTS_DIR, "cluster_viz_data.json")
        if not os.path.isfile(viz_path):
            _update_job(
                job_id,
                status="failed",
                error="未找到 output/ml_artifacts/cluster_viz_data.json，请先完成聚类",
                progress_label="失败",
                completed_at=_done_ts(),
            )
            return

        api_key = resolve_deepseek_api_key("")
        if not api_key:
            _update_job(
                job_id,
                status="failed",
                error="未配置 DeepSeek：请创建项目根目录 deepseek_api_key.txt 或设置 DEEPSEEK_API_KEY",
                progress_label="失败",
                completed_at=_done_ts(),
            )
            return

        def _log(msg: str) -> None:
            _append_job_log(job_id, msg)

        _update_job(job_id, progress=12, progress_label="DeepSeek：逐簇分析样本文…")
        clusters = generate_tweet_topic_cluster_llm(
            ML_ARTIFACTS_DIR,
            api_key=api_key,
            base_url=base_url,
            model=model_s,
            sleep_s=0.85,
            force=bool(force),
            log=_log,
        )
        _update_job(
            job_id,
            status="completed",
            progress=100,
            progress_label="深度命名已写入 cluster_label_map.json",
            completed_at=_done_ts(),
            result={"n_clusters": len(clusters), "cluster_llm_topic": clusters},
        )
    except Exception as exc:
        _append_job_log(job_id, f"失败: {exc}")
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            progress_label="失败",
            completed_at=_done_ts(),
        )


def _infer_text2vec_job(job_id: str, input_path: str):
    try:
        _update_job(job_id, status="running", progress=8, progress_label="读取聚类模型…")
        path = _resolve_training_input(input_path)
        if not os.path.isfile(path):
            raise RuntimeError(f"bundle not found: {path}")

        env = os.environ.copy()
        model_dir = os.path.join("output", "ml_artifacts")
        out_path = os.path.join("output", "user_interest_profiles.json")
        _update_job(job_id, progress=35, progress_label="② 生成兴趣画像（infer）…")

        def _progress_updater(line: str):
            if not line.startswith("[INFER]"):
                return None
            # [INFER] profiles=20/523
            m = re.search(r"profiles=(\d+)/(\d+)", line)
            if not m:
                return None
            done = int(m.group(1))
            total = int(m.group(2))
            if total <= 0:
                return None
            pct = done / float(total)
            progress = 35 + int(max(0.0, min(1.0, pct)) * 55)
            return {"progress": max(35, min(90, progress)), "progress_label": line.replace("[INFER] ", "")}

        _run_cmd_stream(
            job_id,
            [
                PYTHON_BIN,
                "-m",
                "app_pipeline.infer",
                "--input",
                path,
                "--model-dir",
                model_dir,
                "--output",
                out_path,
            ],
            PROJECT_DIR,
            env,
            progress_updater=_progress_updater,
        )
        _update_job(
            job_id,
            status="completed",
            progress=100,
            progress_label="画像生成完成",
            completed_at=datetime.now().isoformat(timespec="seconds"),
            result={"profiles": out_path},
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            progress_label="失败",
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        _append_job_log(job_id, f"Failed: {exc}")


def _cluster_and_infer_text2vec_job(job_id: str, input_path: str, k_min: int, k_max: int):
    """
    一键流程：① KMeans 自动选 k -> ② infer 生成兴趣画像。
    """
    try:
        _update_job(job_id, status="running", progress=8, progress_label="一键：加载特征…")
        path = _resolve_training_input(input_path)
        if not os.path.isfile(path):
            raise RuntimeError(f"bundle not found: {path}")

        env = os.environ.copy()
        model_dir = os.path.join("output", "ml_artifacts")
        out_path = os.path.join("output", "user_interest_profiles.json")

        _update_job(job_id, progress=25, progress_label="一键：PCA(64)+K 搜索+KMeans…")

        def _progress_ksearch2(line: str):
            if not line.startswith("[KSEARCH]"):
                return None
            m = re.search(r"k=(\d+)", line)
            if not m:
                return None
            k = int(m.group(1))
            span = max(1, int(k_max) - int(k_min) + 1)
            frac = (k - int(k_min) + 1) / float(span)
            progress = 25 + int(max(0.0, min(1.0, frac)) * 55)
            return {"progress": max(25, min(78, progress)), "progress_label": line.replace("[KSEARCH] ", "")}

        _run_cmd_stream(
            job_id,
            [
                PYTHON_BIN,
                "-m",
                "app_pipeline.train_tweet_topic",
                "--input",
                path,
                "--output-dir",
                model_dir,
                "--k-min",
                str(k_min),
                "--k-max",
                str(k_max),
            ],
            PROJECT_DIR,
            env,
            progress_updater=_progress_ksearch2,
        )

        # stage 2: infer
        _update_job(job_id, progress=85, progress_label="一键：生成兴趣画像（infer）…")

        def _progress_updater2(line: str):
            if not line.startswith("[INFER]"):
                return None
            m = re.search(r"profiles=(\d+)/(\d+)", line)
            if not m:
                return None
            done = int(m.group(1))
            total = int(m.group(2))
            if total <= 0:
                return None
            pct = done / float(total)
            progress = 85 + int(max(0.0, min(1.0, pct)) * 12)
            return {"progress": max(85, min(98, progress)), "progress_label": line.replace("[INFER] ", "")}

        _run_cmd_stream(
            job_id,
            [
                PYTHON_BIN,
                "-m",
                "app_pipeline.infer",
                "--input",
                path,
                "--model-dir",
                model_dir,
                "--output",
                out_path,
            ],
            PROJECT_DIR,
            env,
            progress_updater=_progress_updater2,
        )

        viz_data = None
        viz_path = os.path.join(model_dir, "cluster_viz_data.json")
        if os.path.isfile(viz_path):
            try:
                with open(viz_path, "rt", encoding="utf-8", errors="replace") as f:
                    viz_data = json.load(f)
            except Exception:
                viz_data = None

        _update_job(
            job_id,
            status="completed",
            progress=100,
            progress_label="一键完成",
            completed_at=datetime.now().isoformat(timespec="seconds"),
            result={
                "profiles": out_path,
                "model_dir": model_dir,
                "cluster_viz_data": viz_data,
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            progress_label="失败",
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        _append_job_log(job_id, f"Failed: {exc}")


def _build_model_all_job(
    job_id: str,
    source_file: str,
    min_tweet_chars: int,
    input_file: str,
    k_min: int,
    k_max: int,
    llm_force: bool = False,
):
    """
    训练页单按钮全流程：
    ① kmeans_prep -> ②+③ cluster_and_infer
    说明：DeepSeek 簇命名已并入 train_tweet_topic 训练阶段，以保证 GBDT 动态标签维度与簇标签一致。
    """
    try:
        _update_job(job_id, status="running", progress=2, progress_label="全流程：准备…")
        _kmeans_prep_job(job_id, source_path=source_file, min_tweet_chars=int(min_tweet_chars))
        if jobs.get(job_id, {}).get("status") != "completed":
            return
        _cluster_and_infer_text2vec_job(job_id, input_path=input_file, k_min=int(k_min), k_max=int(k_max))
        if jobs.get(job_id, {}).get("status") != "completed":
            return
        if bool(llm_force):
            _append_job_log(job_id, "提示：当前训练阶段已执行 DeepSeek 命名；无需额外强制重命名。")
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            progress_label="失败",
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        _append_job_log(job_id, f"Failed: {exc}")


def _cluster_llm_labels_status() -> dict:
    """多标签页展示：缓存是否命中、是否已配置 Key 等。"""
    meta = load_kmeans_multilabel_meta()
    model_s = (os.environ.get("DEEPSEEK_MODEL", "") or "deepseek-chat").strip()
    key_ok = bool(resolve_deepseek_api_key(""))
    if not os.path.isfile(KMEANS_MULTILABEL_JSONL):
        return {
            "has_jsonl": False,
            "cache_hit": False,
            "key_configured": key_ok,
            "model": model_s,
            "message": "尚无 kmeans_multilabel_users.jsonl",
        }
    fp = multilabel_jsonl_fingerprint(KMEANS_MULTILABEL_JSONL)
    labels = meta.get("cluster_llm_labels") if isinstance(meta.get("cluster_llm_labels"), dict) else {}
    fp_meta = meta.get("cluster_llm_source_fingerprint")
    model_meta = meta.get("cluster_llm_model") or ""
    cache_hit = bool(labels) and fp_meta == fp and (model_meta or "deepseek-chat") == model_s
    return {
        "has_jsonl": True,
        "cache_hit": cache_hit,
        "has_llm_labels": bool(labels),
        "key_configured": key_ok,
        "model": model_s,
        "labels": labels,
        "fingerprint_short": fp[:12] + "…",
    }


def _cluster_llm_labels_job(job_id: str, force: bool) -> None:
    def _done_ts() -> str:
        return datetime.now().isoformat(timespec="seconds")

    try:
        _update_job(job_id, progress=2, progress_label="准备…")
        model_s = (os.environ.get("DEEPSEEK_MODEL", "") or "deepseek-chat").strip()
        base_url = (os.environ.get("DEEPSEEK_BASE_URL", "") or "https://api.deepseek.com").strip()

        if not os.path.isfile(KMEANS_MULTILABEL_JSONL):
            _update_job(
                job_id,
                status="failed",
                error="未找到 kmeans_multilabel_users.jsonl",
                completed_at=_done_ts(),
                progress_label="失败",
            )
            return

        fp = multilabel_jsonl_fingerprint(KMEANS_MULTILABEL_JSONL)
        _update_job(job_id, progress=8, progress_label="检查聚类是否变化…")

        if not force:
            cached_frag = try_load_cached_cluster_llm(KMEANS_MULTILABEL_META, fp, model_s)
            if cached_frag is not None:
                _append_job_log(job_id, "多标签文件与 meta 指纹一致，沿用已保存标签，未调用 DeepSeek。")
                _update_job(
                    job_id,
                    status="completed",
                    progress=100,
                    progress_label="已缓存（未调用 API）",
                    completed_at=_done_ts(),
                    result={"cached": True, "cluster_llm_labels": cached_frag["cluster_llm_labels"]},
                )
                return

        api_key = resolve_deepseek_api_key("")
        if not api_key:
            _update_job(
                job_id,
                status="failed",
                error="未配置 DeepSeek：请创建项目根目录 deepseek_api_key.txt 或设置 DEEPSEEK_API_KEY",
                completed_at=_done_ts(),
                progress_label="失败",
            )
            return

        user_ids: list = []
        primaries: list[int] = []
        with open(KMEANS_MULTILABEL_JSONL, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                user_ids.append(row.get("user_id"))
                primaries.append(int(row.get("primary_cluster", 0)))

        n_clusters = max(primaries) + 1 if primaries else 0
        if n_clusters <= 0:
            _update_job(
                job_id,
                status="failed",
                error="无法解析簇数",
                completed_at=_done_ts(),
                progress_label="失败",
            )
            return

        _append_job_log(job_id, f"调用 DeepSeek，K={n_clusters}，模型={model_s}")
        _update_job(job_id, progress=15, progress_label="DeepSeek 生成各簇四字标签…")

        labels = generate_cluster_llm_labels(
            user_ids,
            primaries,
            output_dir=OUTPUT_DIR,
            n_clusters=n_clusters,
            api_key=api_key,
            seed=42,
            base_url=base_url,
            model=model_s,
        )
        merge_llm_labels_into_meta(
            KMEANS_MULTILABEL_META,
            labels,
            provider="deepseek",
            model=model_s,
            base_url=base_url,
            fingerprint=fp,
        )
        _append_job_log(job_id, "已写入 kmeans_multilabel_meta.json（含 cluster_llm_source_fingerprint）")
        _update_job(
            job_id,
            status="completed",
            progress=100,
            progress_label="完成",
            completed_at=_done_ts(),
            result={"cached": False, "cluster_llm_labels": labels},
        )
    except Exception as exc:
        _append_job_log(job_id, f"失败: {exc}")
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            completed_at=_done_ts(),
            progress_label="失败",
        )


def _train_pipeline_job(job_id: str, clusters: int, input_path: str):
    try:
        k_min = 8
        k_max = min(15, max(8, int(clusters)))
        _cluster_text2vec_job(job_id, input_path=input_path, k_min=k_min, k_max=k_max)
        if jobs.get(job_id, {}).get("status") != "completed":
            return
        _infer_text2vec_job(job_id, input_path=input_path)
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            progress_label="失败",
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        _append_job_log(job_id, f"Failed: {exc}")


def _kmeans_prep_job(job_id: str, source_path: str, min_tweet_chars: int):
    """步骤①：从 bundle/unified 导出 output/kmeans_tweets_input.jsonl，供后续聚类/推断。"""
    try:
        _update_job(job_id, status="running", progress=5, progress_label="准备源数据…")
        src = _resolve_prep_source(source_path)
        if not os.path.isfile(src):
            raise RuntimeError(f"源数据不存在: {src}")

        def _on_progress(n: int) -> None:
            pct = 10 + int(min(80.0, (n / 100000.0) * 80.0))
            _update_job(
                job_id,
                progress=min(92, max(10, int(pct))),
                progress_label=f"已写入 {n} 条微博…",
            )

        _update_job(job_id, progress=10, progress_label="写入 kmeans_tweets_input.jsonl…")
        result = export_kmeans_tweets_jsonl(
            src,
            KMEANS_TWEETS_INPUT_JSONL,
            min_tweet_chars=int(min_tweet_chars),
            progress_cb=_on_progress,
        )
        if int(result.get("n_tweets") or 0) <= 0:
            raise RuntimeError("无有效微博，请检查抓取数据或调低 min_tweet_chars")

        _update_job(
            job_id,
            status="completed",
            progress=100,
            progress_label="步骤① 完成",
            completed_at=datetime.now().isoformat(timespec="seconds"),
            result={
                **result,
                "output_jsonl": rel_posix(PROJECT_DIR, KMEANS_TWEETS_INPUT_JSONL),
            },
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            progress_label="失败",
            completed_at=datetime.now().isoformat(timespec="seconds"),
        )
        _append_job_log(job_id, f"Failed: {exc}")


@app.get("/api/profiles")
def api_profiles():
    return load_profiles_effective()


@app.get("/api/profiles/{user_id}")
def api_profile(user_id: str):
    uid = str(user_id).strip()
    for item in load_profiles_effective():
        if str(item.get("user_id", "")).strip() == uid:
            return item
    raise HTTPException(status_code=404, detail="user not found")


@app.get("/api/kmeans_multilabel")
def api_kmeans_multilabel():
    """KMeans 距离门槛多标签结果（由命令行 `python -m app_pipeline.kmeans_multilabel` 生成）。"""
    rows = load_kmeans_multilabel()
    meta = load_kmeans_multilabel_meta()
    return {"meta": meta, "users": rows, "count": len(rows)}


@app.get("/api/kmeans_multilabel/{user_id}")
def api_kmeans_multilabel_user(user_id: str):
    for row in load_kmeans_multilabel():
        if str(row.get("user_id")) == str(user_id):
            return row
    raise HTTPException(status_code=404, detail="user not found in kmeans multilabel")


@app.get("/api/cluster_llm_labels/status")
def api_cluster_llm_labels_status():
    """多标签页用：是否命中缓存、Key 是否就绪。"""
    return _cluster_llm_labels_status()


@app.post("/api/cluster_llm_labels/run")
def api_cluster_llm_labels_run(force: int = Form(0)):
    """
    网页一键生成簇四字标签。
    若 kmeans_multilabel_users.jsonl 内容与 meta 中 cluster_llm_source_fingerprint 一致，则跳过 API（除非 force=1）。
    """
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = _new_job_record(job_id)
    thread = threading.Thread(
        target=_cluster_llm_labels_job,
        args=(job_id, bool(int(force))),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/tweet_cluster_llm/run")
def api_tweet_cluster_llm_run(force: int = Form(0)):
    """
    基于 ml_artifacts/cluster_viz_data.json 的各簇样本文，调用 DeepSeek 生成中文标签 + 深度总结，
    并写入 cluster_label_map.json 与 cluster_llm_topic.json。
    """
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = _new_job_record(job_id)
    thread = threading.Thread(
        target=_tweet_topic_cluster_llm_job,
        args=(job_id, bool(int(force))),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/cluster_preview")
def api_cluster_preview():
    """
    聚类结果预览：用于 train 页面展示最佳 k / 分布等信息。
    如果尚未聚类，返回 404。
    """
    model_dir = os.path.join(OUTPUT_DIR, "ml_artifacts")
    metrics_path = os.path.join(model_dir, "metrics.json")
    k_path = os.path.join(model_dir, "kmeans_k.json")
    user_cluster_path = os.path.join(model_dir, "user_cluster.json")

    if not os.path.isfile(metrics_path) and not os.path.isfile(k_path):
        raise HTTPException(status_code=404, detail="cluster artifacts not found")

    metrics = {}
    if os.path.isfile(metrics_path):
        with open(metrics_path, "rt", encoding="utf-8", errors="replace") as f:
            metrics = json.load(f)

    kinfo = {}
    if os.path.isfile(k_path):
        with open(k_path, "rt", encoding="utf-8", errors="replace") as f:
            kinfo = json.load(f)

    user_cluster = {}
    cluster_sizes = {}
    if os.path.isfile(user_cluster_path):
        with open(user_cluster_path, "rt", encoding="utf-8", errors="replace") as f:
            user_cluster = json.load(f) or {}
        # user_cluster: {uid: cluster_id}
        for _, cid in user_cluster.items():
            cid_i = int(cid)
            cluster_sizes[cid_i] = cluster_sizes.get(cid_i, 0) + 1

    cluster_viz = None
    try:
        from app_pipeline.train_tweet_topic import ensure_cluster_viz_json

        cluster_viz = ensure_cluster_viz_json(model_dir)
    except Exception:
        cluster_viz = None

    cluster_llm_topic = None
    topic_path = os.path.join(model_dir, "cluster_llm_topic.json")
    if os.path.isfile(topic_path):
        try:
            with open(topic_path, "rt", encoding="utf-8", errors="replace") as f:
                cluster_llm_topic = json.load(f)
        except Exception:
            cluster_llm_topic = None

    return {
        "metrics": metrics,
        "kinfo": kinfo,
        "cluster_sizes": cluster_sizes,
        "cluster_viz": cluster_viz,
        "cluster_llm_topic": cluster_llm_topic,
    }


@app.get("/api/stats")
def api_stats():
    profiles = load_profiles_effective()
    counter = Counter([x.get("top_interest", "其他") for x in profiles])
    return {"total_users": len(profiles), "interest_distribution": dict(counter)}


@app.get("/api/workspace")
def api_workspace():
    ws = get_workspace_state(PROJECT_DIR, OUTPUT_DIR, COOKIE_PATH, PROFILE_PATH, WEIBO_CRAWL_LATEST_JSON)
    ws["profiles_count"] = len(load_profiles_effective())
    return {k: v for k, v in ws.items() if k != "default_cookie"}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job


@app.post("/api/cleanup_output")
def api_cleanup_output(
    mode: str = Form("selective"),
    keep_ml_artifacts: int = Form(1),
    del_crawl_intermediate: int = Form(0),
    del_crawl_bundle: int = Form(0),
    del_user_infos: int = Form(0),
    del_profiles: int = Form(0),
    del_ml_artifacts: int = Form(0),
    del_input_user_ids: int = Form(0),
):
    if mode == "full":
        return cleanup_output_full(OUTPUT_DIR, keep_ml_artifacts=bool(int(keep_ml_artifacts)))
    flags = [
        int(del_crawl_intermediate),
        int(del_crawl_bundle),
        int(del_user_infos),
        int(del_profiles),
        int(del_ml_artifacts),
        int(del_input_user_ids),
    ]
    if not any(flags):
        return JSONResponse(status_code=400, content={"error": "请至少勾选一类要删除的内容，或改用手动「清空 output」。"})
    return cleanup_output_selective(
        OUTPUT_DIR,
        project_dir=PROJECT_DIR,
        del_crawl_intermediate=bool(int(del_crawl_intermediate)),
        del_crawl_bundle=bool(int(del_crawl_bundle)),
        del_user_infos=bool(int(del_user_infos)),
        del_profiles=bool(int(del_profiles)),
        del_ml_artifacts=bool(int(del_ml_artifacts)),
        del_input_user_ids=bool(int(del_input_user_ids)),
    )


@app.post("/api/kmeans_prep")
def api_kmeans_prep(
    source_file: str = Form(""),
    min_tweet_chars: int = Form(5),
):
    """KMeans 步骤①：从抓取 bundle/unified 导出 output/kmeans_tweets_input.jsonl（单条已清洗，与嵌入一致）。"""
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = _new_job_record(job_id)

    thread = threading.Thread(
        target=_kmeans_prep_job,
        args=(job_id, (source_file or "").strip(), int(min_tweet_chars)),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/hot_search_sample")
def api_hot_search_sample(
    cookie: str = Form(...),
    hotword_limit: int = Form(10),
    commenters_per_keyword: int = Form(30),
    total_uid_limit: int = Form(200),
    speed_mode: str = Form("steady"),
):
    if not cookie.strip():
        return JSONResponse(status_code=400, content={"error": "cookie is required"})
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = _new_job_record(job_id)
    profile = _get_hot_search_speed_profile(speed_mode)
    thread = threading.Thread(
        target=_hot_search_sample_job,
        args=(
            job_id, cookie, hotword_limit,
            profile["keyword_max_pages"], profile["sleep_between_keywords_sec"], profile["tweets_per_keyword"],
            commenters_per_keyword, total_uid_limit, profile["download_delay"], profile["concurrent_requests"], profile["retry_times"],
        ),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/hot_search_debug_latest")
def api_hot_search_debug_latest():
    path = _latest_hot_search_debug_file()
    if not path or not os.path.exists(path):
        return JSONResponse(status_code=404, content={"error": "no debug log found"})
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    items = data.get("by_keyword") or []
    summary = {
        "keywords": len(items),
        "total_unique_uids": (data.get("summary") or {}).get("total_unique_uids", 0),
        "zero_tweet_keywords": 0,
        "zero_commenter_keywords": 0,
        "global_duplicate_picks": 0,
        "total_comment_rows": 0,
        "total_comment_unique_candidates": 0,
    }
    for item in items:
        if int(item.get("tweet_ids_collected", 0)) <= 0:
            summary["zero_tweet_keywords"] += 1
        if int(item.get("commenter_candidates_unique", 0)) <= 0:
            summary["zero_commenter_keywords"] += 1
        summary["global_duplicate_picks"] += int(item.get("picked_but_duplicate_globally", 0))
        cstats = item.get("comment_extract_stats") or {}
        summary["total_comment_rows"] += int(cstats.get("comment_rows", 0))
        summary["total_comment_unique_candidates"] += int(item.get("commenter_candidates_unique", 0))
    return {"path": path, "summary": summary, "by_keyword": items}


@app.post("/api/crawl_weibo")
def api_crawl_weibo(
    user_id: str = Form(...),
    cookie: str = Form(...),
    max_pages: int = Form(1),
    download_delay: float = Form(0.2),
    concurrent_requests: int = Form(8),
    retry_times: int = Form(2),
):
    if not cookie.strip():
        return JSONResponse(status_code=400, content={"error": "cookie is required"})
    if not (user_id or "").strip():
        return JSONResponse(status_code=400, content={"error": "user_id is required"})
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = _new_job_record(job_id, user_id=user_id)
    thread = threading.Thread(
        target=_crawl_weibo_job,
        args=(job_id, user_id, cookie, max_pages, download_delay, concurrent_requests, retry_times),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/train")
def api_train(
    clusters: int = Form(2),
    input_file: str = Form(""),
):
    job_id = str(uuid.uuid4())
    path = _resolve_training_input((input_file or "").strip())
    with jobs_lock:
        jobs[job_id] = _new_job_record(job_id)
    thread = threading.Thread(
        target=_train_pipeline_job,
        args=(job_id, clusters, path),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/cluster_text2vec")
def api_cluster_text2vec(
    input_file: str = Form(""),
    k_min: int = Form(8),
    k_max: int = Form(15),
):
    job_id = str(uuid.uuid4())
    path = _resolve_training_input((input_file or "").strip())
    with jobs_lock:
        jobs[job_id] = _new_job_record(job_id)
    thread = threading.Thread(
        target=_cluster_text2vec_job,
        args=(job_id, path, k_min, k_max),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/infer_text2vec")
def api_infer_text2vec(
    input_file: str = Form(""),
):
    job_id = str(uuid.uuid4())
    path = _resolve_training_input((input_file or "").strip())
    with jobs_lock:
        jobs[job_id] = _new_job_record(job_id)
    thread = threading.Thread(
        target=_infer_text2vec_job,
        args=(job_id, path),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/cluster_and_infer_text2vec")
def api_cluster_and_infer_text2vec(
    input_file: str = Form(""),
    k_min: int = Form(8),
    k_max: int = Form(15),
):
    job_id = str(uuid.uuid4())
    path = _resolve_training_input((input_file or "").strip())
    with jobs_lock:
        jobs[job_id] = _new_job_record(job_id)
    thread = threading.Thread(
        target=_cluster_and_infer_text2vec_job,
        args=(job_id, path, k_min, k_max),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/build_model_all")
def api_build_model_all(
    source_file: str = Form(""),
    min_tweet_chars: int = Form(5),
    input_file: str = Form(""),
    k_min: int = Form(8),
    k_max: int = Form(15),
    llm_force: int = Form(0),
):
    """训练页单按钮：导出输入 -> 聚类+画像 -> DeepSeek 命名。"""
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = _new_job_record(job_id)
    thread = threading.Thread(
        target=_build_model_all_job,
        args=(job_id, source_file, int(min_tweet_chars), input_file, int(k_min), int(k_max), bool(int(llm_force))),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    profiles = load_profiles_effective()
    top_counter = Counter([x.get("top_interest", "其他") for x in profiles])
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context=_page_context(
            request,
            "home",
            interest_distribution=dict(top_counter),
        ),
    )


@app.get("/crawl", response_class=HTMLResponse)
def page_crawl(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="crawl.html",
        context=_page_context(request, "crawl"),
    )


@app.get("/train", response_class=HTMLResponse)
def page_train(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="train.html",
        context=_page_context(request, "train"),
    )


@app.get("/keyword-crawl", response_class=HTMLResponse)
def page_keyword_crawl(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="keyword_crawl.html",
        context=_page_context(request, "keyword_crawl"),
    )


@app.get("/hot-search")
def redirect_hot_search_to_keyword_crawl():
    return RedirectResponse(url="/keyword-crawl", status_code=302)


@app.get("/clean")
def redirect_clean_to_train():
    """原「数据清洗」已并入训练页步骤①，保留旧链接。"""
    return RedirectResponse(url="/train", status_code=302)


@app.get("/profiles", response_class=HTMLResponse)
def page_profiles(request: Request):
    profiles = load_profiles_effective()
    user_infos = load_user_infos()
    meta = load_kmeans_multilabel_meta()
    by_uid = _kmeans_multilabel_by_user()
    cid2name = _cluster_display_names(meta)
    for p in profiles:
        uid = str(p.get("user_id", "")).strip()
        live_label = str(p.get("top_interest", "") or "")
        row = by_uid.get(uid)
        if isinstance(row, dict):
            cid = row.get("primary_cluster")
            key = str(cid) if cid is not None else ""
            if key and key in cid2name:
                live_label = str(cid2name.get(key) or live_label)
        p["top_interest_live"] = live_label
    return templates.TemplateResponse(
        request=request,
        name="profiles.html",
        context=_page_context(
            request,
            "profiles",
            profiles=profiles,
            user_infos=user_infos,
        ),
    )


@app.get("/personal-interest", response_class=HTMLResponse)
def page_personal_interest(request: Request, user_id: str = ""):
    uid = normalize_uid(user_id)
    result = None
    error = ""
    if uid:
        try:
            result = _predict_personal_interest_from_local(uid)
        except Exception as exc:
            error = str(exc)
    user_info = load_user_infos().get(uid, {}) if uid else {}
    active_labels = load_active_labels()
    return templates.TemplateResponse(
        request=request,
        name="personal_interest.html",
        context=_page_context(
            request,
            "personal_interest",
            query_uid=uid,
            predict_error=error,
            predict_profile=result,
            user_info=user_info,
            active_labels=active_labels,
        ),
    )


@app.get("/kmeans-multilabel")
def redirect_kmeans_multilabel_legacy():
    """旧「KMeans 多标签」页已下线，占位功能见「个人兴趣预测」。"""
    return RedirectResponse(url="/personal-interest", status_code=302)


@app.get("/users/{user_id}/weibo-sources", response_class=HTMLResponse)
def page_user_weibo_sources(request: Request, user_id: str):
    uid_key = normalize_uid(user_id)
    posts, files_hit = load_user_weibo_posts(uid_key, OUTPUT_DIR)
    user_info = load_user_infos().get(uid_key, {})
    latest_u = latest_unified_file()
    latest_unified_rel = rel_posix(PROJECT_DIR, latest_u) if latest_u else ""
    return templates.TemplateResponse(
        request=request,
        name="weibo_sources.html",
        context=_page_context(
            request,
            "profiles",
            user_id=uid_key,
            user_info=user_info,
            weibo_posts=posts,
            weibo_source_files=files_hit,
            latest_unified_rel=latest_unified_rel,
        ),
    )


@app.get("/users/{user_id}", response_class=HTMLResponse)
def user_detail(request: Request, user_id: str):
    uid_key = normalize_uid(user_id)
    target = None
    for item in load_profiles_effective():
        if normalize_uid(item.get("user_id")) == uid_key:
            target = item
            break
    if not target:
        raise HTTPException(status_code=404, detail="user not found")
    user_info = load_user_infos().get(uid_key, {})
    topic_llm_by_label = load_cluster_llm_topic_by_label()
    tm = target.get("topic_mix_from_kmeans") or {}
    active_labels = load_active_labels()
    mix_vals = {str(k): float(v) for k, v in tm.items() if float(v) > 1e-9}
    if active_labels:
        rank = {lab: i for i, lab in enumerate(active_labels)}
        topic_mix_sorted = sorted(
            mix_vals.items(),
            key=lambda x: (rank.get(x[0], 10**9), -x[1], x[0]),
        )
    else:
        topic_mix_sorted = sorted(mix_vals.items(), key=lambda x: -x[1])
    return templates.TemplateResponse(
        request=request,
        name="detail.html",
        context=_page_context(
            request,
            "profiles",
            profile=target,
            user_info=user_info,
            topic_llm_by_label=topic_llm_by_label,
            topic_mix_sorted=topic_mix_sorted,
        ),
    )
