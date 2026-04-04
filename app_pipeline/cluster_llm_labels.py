"""
DeepSeek Chat API（OpenAI 兼容）：为推文话题聚类簇生成深度命名与总结。

密钥优先级：显式参数 > 环境变量 DEEPSEEK_API_KEY > 项目根目录 deepseek_api_key.txt（首行非 # 注释）
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from typing import Any, Callable

DEFAULT_DEEPSEEK_KEY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deepseek_api_key.txt"
)


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


def deepseek_chat(
    user_prompt: str,
    *,
    api_key: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    timeout_s: float = 120.0,
    temperature: float = 0.35,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": float(temperature),
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


# ---------------------------------------------------------------------------
# 推文话题 KMeans 簇：基于 cluster_viz_data.json 的深度命名（标签 + 总结）
# ---------------------------------------------------------------------------


def cluster_viz_fingerprint(viz_path: str) -> str:
    h = hashlib.sha256()
    with open(viz_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def tweet_topic_artifacts_fingerprint(model_dir: str) -> str:
    """
    聚类 KMeans + PCA + 可视化 JSON 联合指纹。任一文件随重训变化则整体变化，
    避免仅比较 cluster_viz 字节导致「重训后仍沿用旧 LLM 命名」。
    """
    h = hashlib.sha256()
    for name in ("kmeans_tweet_model.pkl", "tweet_pca_64.pkl", "cluster_viz_data.json"):
        h.update(name.encode("utf-8"))
        path = os.path.join(model_dir, name)
        if not os.path.isfile(path):
            h.update(b"\x00missing")
            continue
        with open(path, "rb") as fp:
            for chunk in iter(lambda: fp.read(65536), b""):
                h.update(chunk)
    return h.hexdigest()


def try_load_cached_tweet_topic_llm(
    model_dir: str, viz_fp: str, model: str, artifacts_fp: str
) -> dict[str, dict[str, str]] | None:
    p = os.path.join(model_dir, "cluster_llm_topic.json")
    if not os.path.isfile(p):
        return None
    try:
        with open(p, "rt", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    meta = data.get("meta") or {}
    if meta.get("viz_fingerprint") != viz_fp:
        return None
    if meta.get("artifacts_fingerprint") != artifacts_fp:
        return None
    if (meta.get("model") or "") != (model or ""):
        return None
    clusters = data.get("clusters")
    if not isinstance(clusters, dict) or not clusters:
        return None
    out: dict[str, dict[str, str]] = {}
    for k, v in clusters.items():
        if isinstance(v, dict) and v.get("label"):
            out[str(k)] = {
                "label": str(v.get("label", "")),
                "persona": str(v.get("persona", "")),
                "perception": str(v.get("perception", "")),
                "summary": str(v.get("summary", "")),
            }
    return out if out else None


def build_prompt_topic_depth(texts: list[str]) -> str:
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    n = len(texts)
    return (
        f"下面是聚类算法归到同一簇的微博文本（约 {n} 条，已做基础清洗），条与条之间话题可能不完全一致，"
        "请你不要偷懒：必须结合**话题（Topic）+ 语境/文体（Context）**归纳出稳定特征。\n\n"
        "【禁止】不要使用「日常杂谈」「杂谈」「其它」等笼统兜底词作为核心标签；不要指令模型在「内容分散」时敷衍了事。"
        "即使表面主题零散，也要从**情绪共性、叙事口吻、信息形态（如短吐槽/快讯体/应援口号）**里找出可归因的一类。\n\n"
        "请按「双层定义法」输出：\n"
        "1）【核心标签】：恰好 2～4 个汉字。要同时反映「在聊什么」与「怎么在聊」（语境）。\n"
        "2）【人群画像】：单独一句话（勿与标签混写），说明这群用户**是谁、在干什么**（角色 + 行为）。\n"
        "3）【共性感知】：1～2 句中文，点明这约十余条推文里隐藏的**情绪共性**或**文体/语体共性**"
        "（例如：集体焦虑、讽刺口吻、快讯拼接感、粉圈仪式感语言等）。\n\n"
        "【引导示例】（仅说明风格，请按真实文本归纳，勿照抄）：\n"
        "- 话题杂乱但句式都像简讯/转引、信息密度高 → 核心标签可类似【资讯播报】；人群画像如「关注时事与公开信息、习惯转发与短评的采集型用户」。\n"
        "- 多为短句、抱怨、琐事、碎碎念 → 核心标签可类似【琐碎生活】；人群画像如「高频记录生活情绪与琐事的碎碎念用户」。\n"
        "- 加油、应援、爱豆、控评语气 → 核心标签可类似【饭圈应援】；人群画像如「围绕偶像活动进行情感应援与站队表达的粉丝群体」。\n\n"
        "输出：请**严格只输出一个 JSON 对象**，不要 Markdown，不要其它说明。键名固定为：\n"
        '{"label":"2至4字核心标签","persona":"一句话人群画像","perception":"1至2句共性感知"}\n\n'
        f"微博列表：\n{lines}"
    )


def parse_topic_depth_response(response: str) -> tuple[str, str, str, str]:
    """
    解析双层定义 + 共性感知；返回 (label, persona, perception, summary)。
    summary 供下游兼容：拼接 persona + perception。
    """
    text = (response or "").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        stub = (text[:240] + "…") if len(text) > 240 else text or "（模型无输出）"
        return ("解析失败", stub, "", stub)
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        stub = (text[:240] + "…") if len(text) > 240 else text
        return ("解析失败", stub, "", stub)

    label_raw = str(obj.get("label", "")).strip()
    persona = str(obj.get("persona", "") or obj.get("crowd", "") or obj.get("人群画像", "")).strip()
    perception = str(obj.get("perception", "") or obj.get("共性感知", "") or "").strip()
    legacy_summary = str(obj.get("summary", "")).strip()

    zh = "".join(re.findall(r"[\u4e00-\u9fff]", label_raw))
    if len(zh) >= 2:
        label = zh[:4]
    elif len(zh) == 1:
        label = zh
    else:
        zh2 = "".join(re.findall(r"[\u4e00-\u9fff]", persona))
        label = (zh2[:4] if len(zh2) >= 2 else "多元议题") if zh2 else "多元议题"

    if not persona and legacy_summary:
        persona = legacy_summary
    if not perception and legacy_summary and not persona:
        perception = legacy_summary

    parts: list[str] = []
    if persona:
        parts.append(f"【人群画像】{persona}")
    if perception:
        parts.append(f"【共性感知】{perception}")
    summary = "\n".join(parts) if parts else "（模型未完整返回 persona/perception，请结合样本文。）"
    return (label, persona, perception, summary)


def write_tweet_topic_llm_artifacts(
    model_dir: str,
    clusters: dict[str, dict[str, str]],
    *,
    viz_fingerprint: str,
    artifacts_fingerprint: str,
    model: str,
) -> None:
    topic_path = os.path.join(model_dir, "cluster_llm_topic.json")
    label_map: dict[str, str] = {}
    for k, v in sorted(clusters.items(), key=lambda x: int(x[0])):
        lab = str(v.get("label", "")).strip() or f"簇{k}"
        label_map[str(k)] = lab
    cluster_payload: dict[str, Any] = {}
    for k, v in clusters.items():
        cluster_payload[str(k)] = {
            "label": str(v.get("label", "")),
            "persona": str(v.get("persona", "")),
            "perception": str(v.get("perception", "")),
            "summary": str(v.get("summary", "")),
        }
    payload: dict[str, Any] = {
        "meta": {
            "viz_fingerprint": viz_fingerprint,
            "artifacts_fingerprint": artifacts_fingerprint,
            "model": model,
            "provider": "deepseek",
        },
        "clusters": cluster_payload,
    }
    os.makedirs(model_dir, exist_ok=True)
    with open(topic_path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    cmap_path = os.path.join(model_dir, "cluster_label_map.json")
    with open(cmap_path, "wt", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"[OK] 已写入 {cmap_path} 与 {topic_path}", flush=True)


def sync_cluster_label_map_from_cluster_llm_topic(model_dir: str) -> dict[str, str]:
    """
    将 cluster_llm_topic.json 中各簇的「核心标签」写回 cluster_label_map.json，
    与 DeepSeek 深度命名保持一致（不调用 API）。
    """
    model_dir = os.path.abspath(model_dir)
    topic_path = os.path.join(model_dir, "cluster_llm_topic.json")
    if not os.path.isfile(topic_path):
        raise FileNotFoundError(f"未找到 {topic_path}")
    with open(topic_path, "rt", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    clusters = data.get("clusters") if isinstance(data, dict) else None
    if not isinstance(clusters, dict):
        raise ValueError("cluster_llm_topic.json 缺少 clusters")
    label_map: dict[str, str] = {}
    for k, v in sorted(clusters.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else 0):
        if not isinstance(v, dict):
            continue
        lab = str(v.get("label", "")).strip() or f"簇{k}"
        label_map[str(k)] = lab
    cmap_path = os.path.join(model_dir, "cluster_label_map.json")
    os.makedirs(model_dir, exist_ok=True)
    with open(cmap_path, "wt", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"[OK] 已从 cluster_llm_topic 同步 {len(label_map)} 条标签 -> {cmap_path}", flush=True)
    return label_map


def generate_tweet_topic_cluster_llm(
    model_dir: str,
    *,
    api_key: str,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    sleep_s: float = 0.85,
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict[str, dict[str, str]]:
    """
    读取 cluster_viz_data.json 的 samples，逐簇调用 DeepSeek，写入 cluster_llm_topic.json 与 cluster_label_map.json。
    返回 clusters: id 字符串 -> {"label","persona","perception","summary"}。
    """
    def _log(msg: str) -> None:
        print(msg, flush=True)
        if log:
            log(msg)

    model_dir = os.path.abspath(model_dir)
    viz_path = os.path.join(model_dir, "cluster_viz_data.json")
    if not os.path.isfile(viz_path):
        raise FileNotFoundError(f"未找到 {viz_path}，请先完成聚类可视化或重新训练。")

    viz_fp = cluster_viz_fingerprint(viz_path)
    art_fp = tweet_topic_artifacts_fingerprint(model_dir)
    if not force:
        hit = try_load_cached_tweet_topic_llm(model_dir, viz_fp, model, art_fp)
        if hit is not None:
            _log(
                "[LLM-TOPIC] 聚类模型与可视化联合指纹未变，沿用已缓存的深度命名。"
                "若需强制重新调用 DeepSeek，请点「强制重命名」或命令行 --force。"
            )
            write_tweet_topic_llm_artifacts(
                model_dir, hit, viz_fingerprint=viz_fp, artifacts_fingerprint=art_fp, model=model
            )
            return hit

    with open(viz_path, "rt", encoding="utf-8", errors="replace") as f:
        viz = json.load(f)
    samples = viz.get("samples") or {}
    keys = [k for k in samples if isinstance(k, str) and k.startswith("cluster_")]
    cids: list[int] = []
    for k in keys:
        try:
            cids.append(int(k.replace("cluster_", "")))
        except ValueError:
            continue
    cids.sort()

    if not cids:
        raise RuntimeError("cluster_viz_data.json 中无 samples 簇数据")

    rng = random.Random(42)
    out: dict[str, dict[str, str]] = {}

    for i, cid in enumerate(cids):
        key = f"cluster_{cid}"
        texts = [str(t).strip() for t in (samples.get(key) or []) if str(t).strip()]
        if not texts:
            out[str(cid)] = {
                "label": "暂无文本",
                "persona": "",
                "perception": "",
                "summary": "该簇在可视化样本中无可用正文，未调用 API。",
            }
            _log(f"[LLM-TOPIC] 簇 {cid}: 无样本文本，跳过。")
            continue
        if len(texts) > 24:
            texts = rng.sample(texts, 24)
        prompt = build_prompt_topic_depth(texts)
        _log(f"[LLM-TOPIC] 簇 {cid}/{max(cids)}: 调用 DeepSeek，样例 {len(texts)} 条…")
        raw = deepseek_chat(
            prompt,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_s=180.0,
            temperature=0.62,
        )
        label, persona, perception, summary = parse_topic_depth_response(raw)
        p_short = (persona[:48] + "…") if len(persona) > 48 else persona
        c_short = (perception[:48] + "…") if len(perception) > 48 else perception
        _log(f"[LLM-TOPIC] 簇 {cid}: 【{label}】 | 画像：{p_short} | 感知：{c_short}")
        out[str(cid)] = {
            "label": label,
            "persona": persona,
            "perception": perception,
            "summary": summary,
        }
        if sleep_s > 0 and i < len(cids) - 1:
            time.sleep(sleep_s)

    write_tweet_topic_llm_artifacts(
        model_dir, out, viz_fingerprint=viz_fp, artifacts_fingerprint=art_fp, model=model
    )
    return out
