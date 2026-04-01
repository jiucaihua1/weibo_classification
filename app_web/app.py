import glob
import json
import os
import subprocess
import threading
import uuid
from collections import Counter
from datetime import datetime

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app_web.crawl_bundle import build_crawl_bundle, write_crawl_bundle
from app_web.hot_search_service import (
    cleanup_paired_unified_and_aggregate,
    get_hot_search_speed_profile as service_get_hot_search_speed_profile,
    latest_hot_search_debug_file as service_latest_hot_search_debug_file,
    run_hot_search_sample_job as service_run_hot_search_sample_job,
)


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

app = FastAPI(title="Weibo Interest Dashboard", version="0.2.0")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
jobs = {}
jobs_lock = threading.Lock()


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


def _list_unified_files() -> set[str]:
    if not os.path.exists(OUTPUT_DIR):
        return set()
    return {os.path.join(OUTPUT_DIR, n) for n in os.listdir(OUTPUT_DIR) if n.startswith("unified_") and n.endswith(".jsonl")}


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


def _cleanup_output(keep_ml_artifacts: bool = True) -> dict:
    removed_files, removed_dirs = 0, 0
    if not os.path.exists(OUTPUT_DIR):
        return {"removed_files": 0, "removed_dirs": 0}
    for name in os.listdir(OUTPUT_DIR):
        path = os.path.join(OUTPUT_DIR, name)
        if keep_ml_artifacts and name == "ml_artifacts" and os.path.isdir(path):
            continue
        if os.path.isdir(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for fn in files:
                    try:
                        os.remove(os.path.join(root, fn)); removed_files += 1
                    except OSError:
                        pass
                for dn in dirs:
                    try:
                        os.rmdir(os.path.join(root, dn)); removed_dirs += 1
                    except OSError:
                        pass
            try:
                os.rmdir(path); removed_dirs += 1
            except OSError:
                pass
        else:
            try:
                os.remove(path); removed_files += 1
            except OSError:
                pass
    return {"removed_files": removed_files, "removed_dirs": removed_dirs}


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
    completed = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if completed.stdout:
        for line in completed.stdout.splitlines()[-20:]:
            _append_job_log(job_id, line)
    if completed.stderr:
        for line in completed.stderr.splitlines()[-20:]:
            _append_job_log(job_id, line)
    if completed.returncode != 0:
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
        _update_job(job_id, status="running")
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

        _run_cmd(job_id, [PYTHON_BIN, "run_spider.py", "tweet_by_user_id"], WEIBO_DIR, env)
        unified = latest_unified_file()
        if not unified:
            raise RuntimeError("No unified file produced after crawl.")
        after_pipeline = _list_unified_files()
        generated_unified = list(after_pipeline - before_pipeline)
        ordered_ids = _ordered_unique_user_ids(user_id)
        payload = build_crawl_bundle(job_id=job_id, user_ids_ordered=ordered_ids, unified_paths=generated_unified)
        latest_path, archived_path = write_crawl_bundle(OUTPUT_DIR, job_id, payload)
        n_with_data = sum(1 for u in payload["users"] if u.get("records"))
        _append_job_log(job_id, f"Crawl bundle saved: {latest_path} (archive: {archived_path})")
        _append_job_log(job_id, f"Users with at least one record: {n_with_data} / {len(ordered_ids)}")
        ru, ra = cleanup_paired_unified_and_aggregate(generated_unified)
        _append_job_log(job_id, f"Removed crawl artifacts: unified={ru}, user_aggregate={ra}")
        _update_job(
            job_id,
            status="completed",
            completed_at=datetime.now().isoformat(timespec="seconds"),
            result={
                "output_file": latest_path,
                "archive_file": archived_path,
                "n_users_requested": len(ordered_ids),
                "n_users_with_records": n_with_data,
                "removed_intermediate_unified": ru,
                "removed_intermediate_user_aggregate": ra,
            },
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), completed_at=datetime.now().isoformat(timespec="seconds"))
        _append_job_log(job_id, f"Failed: {exc}")


def _resolve_training_input(path: str) -> str:
    p = (path or "").strip()
    if not p:
        return WEIBO_CRAWL_LATEST_JSON
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(PROJECT_DIR, p))


def _train_pipeline_job(job_id: str, clusters: int, input_path: str):
    try:
        _update_job(job_id, status="running")
        path = _resolve_training_input(input_path)
        if not os.path.isfile(path):
            raise RuntimeError(f"训练输入文件不存在: {path}，请先执行「抓取微博」。")
        env = os.environ.copy()
        _run_cmd(
            job_id,
            [PYTHON_BIN, "-m", "app_pipeline.train", "--input", path, "--output-dir", os.path.join("output", "ml_artifacts"), "--clusters", str(clusters)],
            PROJECT_DIR,
            env,
        )
        _run_cmd(
            job_id,
            [PYTHON_BIN, "-m", "app_pipeline.infer", "--input", path, "--model-dir", os.path.join("output", "ml_artifacts"), "--output", os.path.join("output", "user_interest_profiles.json")],
            PROJECT_DIR,
            env,
        )
        _append_job_log(job_id, f"Train/infer done. Input: {path}")
        _update_job(
            job_id,
            status="completed",
            completed_at=datetime.now().isoformat(timespec="seconds"),
            result={"input_file": path, "profiles": os.path.join("output", "user_interest_profiles.json")},
        )
    except Exception as exc:
        _update_job(job_id, status="failed", error=str(exc), completed_at=datetime.now().isoformat(timespec="seconds"))
        _append_job_log(job_id, f"Failed: {exc}")


@app.get("/api/profiles")
def api_profiles():
    return load_profiles()


@app.get("/api/profiles/{user_id}")
def api_profile(user_id: str):
    for item in load_profiles():
        if item.get("user_id") == user_id:
            return item
    raise HTTPException(status_code=404, detail="user not found")


@app.get("/api/stats")
def api_stats():
    profiles = load_profiles()
    counter = Counter([x.get("top_interest", "其他") for x in profiles])
    return {"total_users": len(profiles), "interest_distribution": dict(counter)}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return job


@app.post("/api/cleanup_output")
def api_cleanup_output(keep_ml_artifacts: int = Form(1)):
    summary = _cleanup_output(keep_ml_artifacts=bool(int(keep_ml_artifacts)))
    return {"status": "ok", **summary}


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
        jobs[job_id] = {"job_id": job_id, "status": "queued", "error": "", "created_at": datetime.now().isoformat(timespec="seconds"),
                        "completed_at": "", "user_id": "", "logs": [], "result": {}}
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
        jobs[job_id] = {"job_id": job_id, "status": "queued", "error": "", "created_at": datetime.now().isoformat(timespec="seconds"),
                        "completed_at": "", "user_id": user_id, "logs": [], "result": {}}
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
    path = (input_file or "").strip() or WEIBO_CRAWL_LATEST_JSON
    with jobs_lock:
        jobs[job_id] = {"job_id": job_id, "status": "queued", "error": "", "created_at": datetime.now().isoformat(timespec="seconds"),
                        "completed_at": "", "user_id": "", "logs": [], "result": {}}
    thread = threading.Thread(
        target=_train_pipeline_job,
        args=(job_id, clusters, path),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id, "status": "queued"}


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    profiles = load_profiles()
    user_infos = load_user_infos()
    top_counter = Counter([x.get("top_interest", "其他") for x in profiles])
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request, "profiles": profiles, "user_infos": user_infos,
                 "interest_distribution": dict(top_counter), "default_user_id": "1087770692"},
    )


@app.get("/users/{user_id}", response_class=HTMLResponse)
def user_detail(request: Request, user_id: str):
    target = None
    for item in load_profiles():
        if item.get("user_id") == user_id:
            target = item
            break
    if not target:
        raise HTTPException(status_code=404, detail="user not found")
    user_info = load_user_infos().get(str(user_id), {})
    return templates.TemplateResponse(request=request, name="detail.html", context={"request": request, "profile": target, "user_info": user_info})
