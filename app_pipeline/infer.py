import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from app_pipeline.data_io import iter_crawl_bundle_users
from app_pipeline.feature_tweet_embeddings import (
    TweetEmbeddingConfig,
    extract_tweet_embeddings,
    mean_embedding_by_user,
    static_features_for_users,
)
from app_pipeline.train_tweet_topic import TweetMultilabelGBDT


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _extract_evidence_rows(records: List[Dict], user_id: str, limit: int = 20) -> List[Dict]:
    rows = []
    for row in records:
        if str(row.get("user_id", "")) != str(user_id):
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        raw = row.get("raw") or {}
        rows.append(
            {
                "created_at": row.get("created_at"),
                "source_type": row.get("source_type"),
                "text": text,
                "item_id": row.get("item_id"),
                "url": raw.get("url"),
            }
        )
    rows.sort(key=lambda r: _safe_int((r.get("item_id") or 0), 0), reverse=True)
    return rows[:limit]


def _is_tweet_topic_pipeline(model_dir: str) -> bool:
    p = os.path.join(model_dir, "training_pipeline.json")
    if not os.path.isfile(p):
        return False
    try:
        with open(p, "rt", encoding="utf-8", errors="replace") as f:
            m = json.load(f)
        return m.get("pipeline") == "tweet_topic_multilabel"
    except Exception:
        return False


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _topic_interest_narratives(topic_mix: Dict[str, float]) -> List[str]:
    """按话题占比生成可读叙事句，供前端展示。"""
    items = [(lab, float(p)) for lab, p in topic_mix.items() if float(p) > 1e-9]
    items.sort(key=lambda x: -x[1])
    out: List[str] = []
    for lab, p in items:
        pct = int(round(p * 100))
        if pct <= 0:
            continue
        if p >= 0.35:
            level = "极高"
        elif p >= 0.22:
            level = "较高"
        elif p >= 0.12:
            level = "一定"
        else:
            level = "轻度"
        out.append(f"该用户对 [{lab}] 表现出{level}关注（占比 {pct}%）")
    return out


def _primary_topic_from_mix(topic_mix: Dict[str, float]) -> str:
    items = [(lab, float(p)) for lab, p in topic_mix.items() if float(p) > 1e-9]
    if not items:
        return ""
    return max(items, key=lambda x: x[1])[0]


def _load_cluster_llm_topic(model_dir: str) -> Dict[int, Dict[str, str]]:
    """cluster_llm_topic.json -> {cluster_id: {label, persona, perception, summary}}"""
    p = os.path.join(model_dir, "cluster_llm_topic.json")
    if not os.path.isfile(p):
        return {}
    try:
        with open(p, "rt", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except Exception:
        return {}
    clusters = data.get("clusters") if isinstance(data, dict) else None
    if not isinstance(clusters, dict):
        return {}
    out: Dict[int, Dict[str, str]] = {}
    for k, v in clusters.items():
        try:
            ki = int(k)
        except (TypeError, ValueError):
            continue
        if isinstance(v, dict):
            out[ki] = {
                "label": str(v.get("label", "") or ""),
                "persona": str(v.get("persona", "") or ""),
                "perception": str(v.get("perception", "") or ""),
                "summary": str(v.get("summary", "") or ""),
            }
    return out


def _load_active_labels(model_dir: str, fallback: List[str]) -> List[str]:
    p = os.path.join(model_dir, "active_labels.json")
    if not os.path.isfile(p):
        return list(fallback)
    try:
        with open(p, "rt", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
    except Exception:
        return list(fallback)
    raw = data.get("active_labels") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        return list(fallback)
    out: List[str] = []
    seen: set[str] = set()
    for x in raw:
        s = str(x or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out or list(fallback)


def infer_tweet_topic_multilabel(
    unified_path: str,
    model_dir: str,
    output_file: str,
) -> List[Dict]:
    """
    单条微博向量流水线：加载 tweet KMeans + GBDT，按用户均值向量（+静态特征）预测多标签；
    同时给出基于微博条目的「话题占比」作解释。
    """
    with open(os.path.join(model_dir, "training_pipeline.json"), "rt", encoding="utf-8", errors="replace") as f:
        meta = json.load(f)
    with open(os.path.join(model_dir, "cluster_label_map.json"), "rt", encoding="utf-8", errors="replace") as f:
        cluster_map = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(model_dir, "kmeans_tweet_model.pkl"), "rb") as f:
        km = pickle.load(f)
    tweet_pca_path = os.path.join(model_dir, "tweet_pca_64.pkl")
    tweet_pca = None
    if os.path.isfile(tweet_pca_path):
        try:
            with open(tweet_pca_path, "rb") as f:
                tweet_pca = pickle.load(f)
        except Exception:
            tweet_pca = None
    with open(os.path.join(model_dir, "gbdt_multilabel.pkl"), "rb") as f:
        gbdt_raw = pickle.load(f)
        if isinstance(gbdt_raw, dict) and gbdt_raw.get("_gbdt_artifact_version") == 1:
            clf = TweetMultilabelGBDT(gbdt_raw["estimators"], gbdt_raw["constants"])
        else:
            clf = gbdt_raw

    interest_labels = _load_active_labels(model_dir, list(meta.get("interest_labels") or []))
    noise_cids = set()
    for x in meta.get("noise_cluster_ids") or []:
        try:
            noise_cids.add(int(x))
        except (TypeError, ValueError):
            pass
    with_static = bool(meta.get("with_static_features", True))
    model_name = str(meta.get("model_name", "shibing624/text2vec-base-chinese"))
    device = str(meta.get("device", "cuda"))
    min_tweet_chars = int(meta.get("min_tweet_chars", 5))

    cfg = TweetEmbeddingConfig(
        min_tweet_chars=min_tweet_chars,
        encode_batch_size=64,
        device=device,
        model_name=model_name,
    )
    print("[INFER] tweet-topic: encoding per-tweet vectors …", flush=True)
    tweet_user_ids, x_tweets, tweet_texts = extract_tweet_embeddings(unified_path, cfg=cfg)
    users_order, x_mean = mean_embedding_by_user(tweet_user_ids, x_tweets)
    if with_static:
        static_mat, _ = static_features_for_users(unified_path, users_order)
        x_in = np.concatenate([x_mean, static_mat], axis=1).astype(np.float32, copy=False)
    else:
        x_in = x_mean

    prob_list = (
        clf.predict_proba_per_label(x_in)
        if hasattr(clf, "predict_proba_per_label")
        else clf.predict_proba(x_in)
    )
    y_hat = clf.predict(x_in)

    x768 = x_tweets.astype(np.float32, copy=False)
    if tweet_pca is not None:
        # Strict order must match training:
        #   768 -> PCA(64) -> L2 normalize -> KMeans
        x64 = tweet_pca.transform(x768).astype(np.float32, copy=False)
        x_kmeans_input = _normalize_rows(x64)
    else:
        # Backward compatibility for old artifacts.
        x_kmeans_input = _normalize_rows(x768)

    tweet_cids = km.predict(x_kmeans_input)

    topic_llm_by_c = _load_cluster_llm_topic(model_dir)

    _skip_label = {"Noise", "日常杂谈"}
    topic_label_universe = sorted({v for v in cluster_map.values() if v not in _skip_label})
    profiles: List[Dict] = []
    n_users = len(users_order)
    print(f"[INFER] total_feature_users={n_users}", flush=True)

    uid_to_indices: Dict[str, List[int]] = defaultdict(list)
    for i, u in enumerate(tweet_user_ids):
        uid_to_indices[str(u).strip()].append(i)

    for ui, uid in enumerate(users_order):
        idxs = uid_to_indices[uid]
        n_t = max(1, len(idxs))
        label_counts = Counter()
        n_eff = 0
        for j in idxs:
            cid = int(tweet_cids[j])
            if cid in noise_cids:
                continue
            lab = cluster_map.get(cid, "其他")
            if lab in _skip_label:
                continue
            n_eff += 1
            label_counts[lab] += 1
        if n_eff <= 0:
            topic_mix = {lab: 0.0 for lab in topic_label_universe}
        else:
            topic_mix = {lab: label_counts.get(lab, 0) / float(n_eff) for lab in topic_label_universe}
        narratives = _topic_interest_narratives(topic_mix)
        primary_topic = _primary_topic_from_mix(topic_mix)

        scores_gbdt: Dict[str, float] = {}
        for j, lab in enumerate(interest_labels):
            arr = prob_list[j]
            # positive class column 1
            scores_gbdt[lab] = float(arr[ui, 1]) if arr.shape[1] > 1 else float(arr[ui, 0])

        active = [interest_labels[j] for j in range(len(interest_labels)) if int(y_hat[ui, j]) == 1]
        top_s = sorted(scores_gbdt.items(), key=lambda x: x[1], reverse=True)
        gbdt_top = "、".join(active) if active else (top_s[0][0] if top_s else "—")
        top_interest = primary_topic if primary_topic else gbdt_top

        evidence: List[Dict[str, Any]] = []
        for j in idxs[:20]:
            cj = int(tweet_cids[j])
            tl = cluster_map.get(cj, "其他")
            tmeta = topic_llm_by_c.get(cj) or {}
            raw = tweet_texts[j] if j < len(tweet_texts) else ""
            snippet = raw if len(raw) <= 160 else raw[:160] + "…"
            evidence.append(
                {
                    "topic_hint": tl,
                    "topic_summary": tmeta.get("summary", ""),
                    "topic_persona": tmeta.get("persona", ""),
                    "topic_perception": tmeta.get("perception", ""),
                    "text": snippet,
                }
            )

        profiles.append(
            {
                "user_id": uid,
                "top_interest": top_interest,
                "cluster_id": None,
                "interest_scores": scores_gbdt,
                "topic_mix_from_kmeans": topic_mix,
                "topic_interest_narratives": narratives,
                "gbdt_top_interest": gbdt_top,
                "gbdt_positive_labels": active,
                "source_stats": {"tweet": n_t},
                "sample_size": n_t,
                "evidence": evidence,
            }
        )
        if (ui + 1) % 20 == 0:
            print(f"[INFER] profiles={ui + 1}/{n_users}", flush=True)

    profiles.sort(key=lambda x: x["sample_size"], reverse=True)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "wt", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    return profiles


def infer(
    unified_path: str,
    model_dir: str,
    output_file: str,
) -> List[Dict]:
    """仅支持训练页产出的 tweet_topic_multilabel 模型（由 Web 子进程调用）。"""
    if not _is_tweet_topic_pipeline(model_dir):
        raise RuntimeError(
            "当前仅支持 tweet_topic_multilabel 流水线：请使用网页「训练/全流程」生成 output/ml_artifacts，"
            "并确保 training_pipeline.json 中 pipeline 为 tweet_topic_multilabel。"
        )
    return infer_tweet_topic_multilabel(unified_path, model_dir, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer user interest profiles.")
    parser.add_argument("--input", required=True, help="Path to unified jsonl or weibo_crawl_*.json bundle.")
    parser.add_argument("--model-dir", default="output/ml_artifacts", help="Model artifact directory.")
    parser.add_argument("--output", default="output/user_interest_profiles.json", help="Output profile json file.")
    args = parser.parse_args()
    result = infer(args.input, args.model_dir, args.output)
    print(f"inferred_users={len(result)}")
