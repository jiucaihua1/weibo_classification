"""
使用 DeepSeek Chat API，按主簇从原始微博中抽样，为每个簇生成 4 字中文兴趣标签。
与 OpenAI 兼容接口：POST https://api.deepseek.com/chat/completions

密钥优先级：显式参数 > 环境变量 DEEPSEEK_API_KEY > 项目根目录 deepseek_api_key.txt（首行非 # 注释）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from typing import Any

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_PIPELINE_DIR)
DEFAULT_DEEPSEEK_KEY_FILE = os.path.join(_PROJECT_ROOT, "deepseek_api_key.txt")


def resolve_deepseek_api_key(explicit: str = "", *, key_file: str = "") -> str:
    """与 cookie.txt 类似：本地单行存 Key；支持 # 开头注释行。"""
    if (explicit or "").strip():
        return explicit.strip()
    env = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env:
        return env
    path = (key_file or "").strip() or DEFAULT_DEEPSEEK_KEY_FILE
    if not path or not os.path.isfile(path):
        return ""
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            return line.strip()
    return ""


def multilabel_jsonl_fingerprint(jsonl_path: str) -> str:
    """当前多标签结果文件内容的 sha256；聚类不变则哈希不变。"""
    h = hashlib.sha256()
    with open(jsonl_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def try_load_cached_cluster_llm(
    meta_path: str,
    jsonl_fingerprint: str,
    model: str,
) -> dict[str, Any] | None:
    """
    若 meta 中已有与当前 jsonl 指纹一致、且模型一致的 LLM 标签，返回可合并进 meta 的字段 dict；否则 None。
    """
    if not meta_path or not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "rt", encoding="utf-8", errors="replace") as f:
            old = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if old.get("cluster_llm_source_fingerprint") != jsonl_fingerprint:
        return None
    if (old.get("cluster_llm_model") or "") != (model or ""):
        return None
    labels = old.get("cluster_llm_labels")
    if not isinstance(labels, dict) or not labels:
        return None
    return {
        "cluster_llm_labels": labels,
        "cluster_llm_provider": old.get("cluster_llm_provider", "deepseek"),
        "cluster_llm_model": old.get("cluster_llm_model", model),
        "cluster_llm_base_url": old.get("cluster_llm_base_url", ""),
        "cluster_llm_source_fingerprint": jsonl_fingerprint,
    }


# ---------------------------------------------------------------------------
# 从 unified 聚合每个用户的正文列表（与 weibo_sources 逻辑一致）
# ---------------------------------------------------------------------------


def _load_uid_texts_from_unified_files(output_dir: str) -> dict[str, list[str]]:
    import glob

    paths = glob.glob(os.path.join(output_dir, "unified_*.jsonl"))
    paths.sort(key=os.path.getmtime, reverse=True)
    by_uid: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)

    for path in paths:
        with open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = str(row.get("user_id", "")).strip()
                if not uid:
                    continue
                raw = row.get("raw") or {}
                text = str(raw.get("content") or row.get("text") or "").strip()
                if not text:
                    continue
                item_id = str(raw.get("_id") or row.get("item_id") or "").strip()
                dedupe_key = item_id or text[:80]
                if dedupe_key in seen[uid]:
                    continue
                seen[uid].add(dedupe_key)
                by_uid[uid].append(text)
    return dict(by_uid)


def _sample_texts_for_cluster(all_texts: list[str], rng: random.Random) -> list[str]:
    if not all_texts:
        return []
    n = len(all_texts)
    if n >= 50:
        k = 50
    elif n >= 30:
        k = rng.randint(30, min(50, n))
    else:
        k = n
    if k >= n:
        return list(all_texts)
    return rng.sample(all_texts, k)


def build_prompt(texts: list[str]) -> str:
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    return (
        f"这里有 {len(texts)} 条来自同一个兴趣群体（同一聚类簇）的用户微博原始文本。"
        "请你阅读后，分析这些用户的共同兴趣爱好，"
        "用恰好 4 个汉字为该群体生成一个精准、简洁的中文兴趣标签。"
        "不要标点、不要英文、不要序号或解释，只输出这 4 个字。\n\n"
        f"文本如下：\n{lines}"
    )


def parse_four_char_label(response: str) -> str:
    first = (response or "").strip().splitlines()[0].strip()
    first = re.sub(r"[「」『』\s\"'“”《》]", "", first)
    chars = re.findall(r"[\u4e00-\u9fff]", first)
    if len(chars) >= 4:
        return "".join(chars[:4])
    if len(chars) > 0:
        s = "".join(chars)
        return (s + "兴趣群体")[:4]
    return "兴趣群体"


def deepseek_chat(
    user_prompt: str,
    *,
    api_key: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    timeout_s: float = 120.0,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": 0.35,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"DeepSeek HTTP {e.code}: {err}") from e
    try:
        return str(body["choices"][0]["message"]["content"] or "")
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected API response: {body!r}") from e


def generate_cluster_llm_labels(
    user_ids: list,
    primaries: list[int],
    *,
    output_dir: str,
    n_clusters: int,
    api_key: str,
    seed: int = 42,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    sleep_s: float = 0.8,
) -> dict[str, str]:
    """
    返回 { \"0\": \"四字标签\", ... }。某一簇无文本时跳过 API，标签为「暂无文本」。
    """
    rng = random.Random(int(seed))
    uid_texts = _load_uid_texts_from_unified_files(output_dir)
    by_cluster: dict[int, list[str]] = defaultdict(list)
    for uid, pc in zip(user_ids, primaries):
        pc_i = int(pc)
        if 0 <= pc_i < int(n_clusters):
            for t in uid_texts.get(str(uid).strip(), []):
                by_cluster[pc_i].append(t)

    out: dict[str, str] = {}
    for c in range(int(n_clusters)):
        pool = by_cluster.get(c, [])
        sampled = _sample_texts_for_cluster(pool, rng)
        if not sampled:
            print(f"[LLM] 簇 {c}: 无可用微博文本，跳过 API。")
            out[str(c)] = "暂无文本"
            continue
        prompt = build_prompt(sampled)
        print(f"[LLM] 簇 {c}: 调用 DeepSeek，样本 {len(sampled)} 条…")
        raw = deepseek_chat(prompt, api_key=api_key, base_url=base_url, model=model)
        label = parse_four_char_label(raw)
        print(f"[LLM] 簇 {c}: 模型原文 → {raw.strip()[:80]}…")
        print(f"[LLM] 簇 {c}: 采用标签 → 【{label}】")
        out[str(c)] = label
        if sleep_s > 0 and c < int(n_clusters) - 1:
            time.sleep(sleep_s)

    return out


def merge_llm_labels_into_meta(meta_path: str, labels: dict[str, str], **extra: Any) -> None:
    meta: dict[str, Any] = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "rt", encoding="utf-8", errors="replace") as f:
            try:
                meta = json.load(f)
            except json.JSONDecodeError:
                meta = {}
    meta["cluster_llm_labels"] = labels
    meta["cluster_llm_provider"] = extra.get("provider", "deepseek")
    meta["cluster_llm_model"] = extra.get("model", "deepseek-chat")
    if extra.get("base_url"):
        meta["cluster_llm_base_url"] = extra["base_url"]
    fp = extra.get("fingerprint") or extra.get("cluster_llm_source_fingerprint")
    if fp:
        meta["cluster_llm_source_fingerprint"] = fp
    meta.pop("cluster_display_names", None)
    with open(meta_path, "wt", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[OK] 已写入 {meta_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="用 DeepSeek 为各主簇生成 4 字兴趣标签（基于 unified 微博抽样）。")
    ap.add_argument("--output-dir", default="output", help="含 unified_*.jsonl 与 kmeans_multilabel_users.jsonl")
    ap.add_argument(
        "--multilabel-jsonl",
        default="",
        help="多标签结果 jsonl，默认 output/kmeans_multilabel_users.jsonl",
    )
    ap.add_argument(
        "--meta-out",
        default="",
        help="写入 cluster_llm_labels 的 meta 路径，默认 output/kmeans_multilabel_meta.json",
    )
    ap.add_argument("--api-key", default="", help="若空则读环境变量或项目根目录 deepseek_api_key.txt")
    ap.add_argument(
        "--key-file",
        default="",
        help=f"Key 文件路径，默认 {DEFAULT_DEEPSEEK_KEY_FILE}",
    )
    ap.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip())
    ap.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.8, help="簇与簇之间休眠秒数，降频")
    ap.add_argument(
        "--force",
        action="store_true",
        help="忽略缓存，即使聚类文件指纹未变也重新调用 API",
    )
    args = ap.parse_args()

    out_dir = os.path.normpath(args.output_dir)
    jsonl_path = (args.multilabel_jsonl or "").strip() or os.path.join(out_dir, "kmeans_multilabel_users.jsonl")
    meta_path = (args.meta_out or "").strip() or os.path.join(out_dir, "kmeans_multilabel_meta.json")

    if not os.path.isfile(jsonl_path):
        print(f"[ERROR] 未找到 {jsonl_path}")
        return 1

    fp = multilabel_jsonl_fingerprint(jsonl_path)
    if not args.force:
        hit = try_load_cached_cluster_llm(meta_path, fp, str(args.model))
        if hit is not None:
            print("[INFO] 多标签文件指纹未变，沿用 meta 中已有标签，跳过 API。需要重跑请加 --force。")
            return 0

    api_key = resolve_deepseek_api_key(args.api_key, key_file=args.key_file)
    if not api_key:
        print(
            f"[ERROR] 未找到 DeepSeek Key：请设置环境变量 DEEPSEEK_API_KEY，"
            f"或在项目根创建 {os.path.basename(DEFAULT_DEEPSEEK_KEY_FILE)}（参考 deepseek_api_key.txt.example）"
        )
        return 1

    user_ids: list = []
    primaries: list[int] = []
    with open(jsonl_path, "rt", encoding="utf-8", errors="replace") as f:
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
        print("[ERROR] 无法解析簇数")
        return 1

    labels = generate_cluster_llm_labels(
        user_ids,
        primaries,
        output_dir=out_dir,
        n_clusters=n_clusters,
        api_key=api_key,
        seed=int(args.seed),
        base_url=args.base_url,
        model=args.model,
        sleep_s=float(args.sleep),
    )
    merge_llm_labels_into_meta(
        meta_path,
        labels,
        provider="deepseek",
        model=args.model,
        base_url=args.base_url,
        fingerprint=fp,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
