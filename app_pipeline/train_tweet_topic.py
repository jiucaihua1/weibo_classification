"""
Train: 单条微博向量 → K-Means（话题簇）→ 用户多标签比例（门槛）→ GBDT 多输出。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from app_pipeline.cluster_llm_labels import (
    generate_tweet_topic_cluster_llm,
    resolve_deepseek_api_key,
)
from app_pipeline.feature_tweet_embeddings import (
    TweetEmbeddingConfig,
    extract_tweet_embeddings,
    mean_embedding_by_user,
    static_features_for_users,
)


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _subsample_for_score(n: int, max_n: int = 8000, seed: int = 42) -> np.ndarray:
    if n <= max_n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_n, replace=False)


NOISE_LABEL = "Noise"


def _is_noise_label_name(label: str) -> bool:
    s = str(label or "").strip().lower()
    if not s:
        return True
    if s == "noise":
        return True
    for token in ("噪声", "杂谈", "闲聊", "无意义", "其他", "misc", "unknown", "暂无文本"):
        if token in s:
            return True
    return False


def _build_dynamic_labels(cluster_label_map: Dict[int, str]) -> Tuple[List[str], List[int]]:
    labels: List[str] = []
    seen: set[str] = set()
    noise_ids: List[int] = []
    for cid in sorted(cluster_label_map.keys()):
        lab = str(cluster_label_map.get(cid, "") or "").strip()
        if _is_noise_label_name(lab):
            noise_ids.append(int(cid))
            continue
        if lab not in seen:
            seen.add(lab)
            labels.append(lab)
    return labels, noise_ids


def _build_cluster_viz_data(
    x64: np.ndarray,
    tweet_cluster_ids: np.ndarray,
    tweet_texts: Sequence[str],
    *,
    per_cluster_samples: int = 10,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    构建前端可视化用的散点与示例文本：
    - 在 PCA(64) 空间随机抽取 min(2000, 10%·N) 条（上限 2000），用 t-SNE 映射到 2D（非线性，利于簇分离）；
    - 每个簇随机抽取若干条清洗后微博文本（与散点独立）。
    """
    n = x64.shape[0]
    if tweet_cluster_ids.shape[0] != n:
        raise ValueError("tweet_cluster_ids length mismatch vs x64")
    if len(tweet_texts) != n:
        m = min(len(tweet_texts), n)
    else:
        m = n

    rng = np.random.default_rng(random_state)
    # 展示子样本：min(2000, 10%·N)，且不超过 m
    ten_pct = max(1, int(round(m * 0.1)))
    n_sub = min(2000, ten_pct, m)
    if n_sub < m:
        sel = rng.choice(m, size=n_sub, replace=False)
    else:
        sel = np.arange(m, dtype=np.int64)

    x_sub = x64[sel].astype(np.float64, copy=False)
    n_s = int(x_sub.shape[0])
    if n_s >= 2:
        # sklearn：perplexity 须严格小于样本数；目标 30，不足时自动下调
        perp = float(min(30, max(1, n_s - 1)))
        print(f"[VIZ] t-SNE 2D: n_sub={n_s} perplexity={perp} (PCA64→t-SNE)", flush=True)
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=int(random_state),
            init="pca",
            learning_rate="auto",
        )
        x2 = tsne.fit_transform(x_sub)
    else:
        x2 = np.zeros((n_s, 2), dtype=np.float64)
        perp = 0.0

    scatter: List[Dict[str, Any]] = []
    for i, idx in enumerate(sel):
        cid = int(tweet_cluster_ids[idx])
        txt = tweet_texts[idx] if idx < len(tweet_texts) else ""
        snippet = txt[:20]
        scatter.append(
            {
                "x": float(x2[i, 0]),
                "y": float(x2[i, 1]),
                "cluster": cid,
                "text": snippet,
            }
        )

    samples: Dict[str, List[str]] = {}
    unique_clusters = sorted({int(c) for c in tweet_cluster_ids[:m]})
    for cid in unique_clusters:
        idxs = [i for i in range(m) if int(tweet_cluster_ids[i]) == cid]
        if not idxs:
            continue
        if len(idxs) > per_cluster_samples:
            idxs = list(rng.choice(idxs, size=per_cluster_samples, replace=False))
        texts = []
        for i in idxs:
            t = tweet_texts[i] if i < len(tweet_texts) else ""
            if t:
                texts.append(t)
        samples[f"cluster_{cid}"] = texts

    return {
        "scatter": scatter,
        "samples": samples,
        "viz_method": "tsne",
        "viz_n_subsample": n_s,
        "viz_perplexity": perp,
    }


def ensure_cluster_viz_json(model_dir: str) -> Optional[Dict[str, Any]]:
    """
    读取或补写 output/ml_artifacts/cluster_viz_data.json。
    若文件缺失但已有 kmeans_tweet_model.pkl、tweet_pca_64.pkl、source_bundle_path.txt、
    training_pipeline.json，则重新对 bundle 做 encode 并生成可视化（首次可能较慢）。
    """
    path = os.path.join(model_dir, "cluster_viz_data.json")
    if os.path.isfile(path):
        try:
            with open(path, "rt", encoding="utf-8", errors="replace") as f:
                cached = json.load(f)
            if cached.get("viz_method") == "tsne":
                return cached
        except Exception:
            pass

    km_path = os.path.join(model_dir, "kmeans_tweet_model.pkl")
    pca_path = os.path.join(model_dir, "tweet_pca_64.pkl")
    src_path = os.path.join(model_dir, "source_bundle_path.txt")
    meta_path = os.path.join(model_dir, "training_pipeline.json")
    if not all(os.path.isfile(p) for p in (km_path, pca_path, src_path, meta_path)):
        return None

    with open(meta_path, "rt", encoding="utf-8", errors="replace") as f:
        meta = json.load(f)
    with open(src_path, "rt", encoding="utf-8", errors="replace") as f:
        bundle_path = f.read().strip()
    if not bundle_path or not os.path.isfile(bundle_path):
        return None

    with open(km_path, "rb") as f:
        km = pickle.load(f)
    with open(pca_path, "rb") as f:
        tweet_pca = pickle.load(f)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    device = str(meta.get("device", "cuda"))
    model_name = _resolve_local_model_dir(str(meta.get("model_name", "shibing624/text2vec-base-chinese")))
    min_tweet_chars = int(meta.get("min_tweet_chars", 5))
    cfg = TweetEmbeddingConfig(
        min_tweet_chars=min_tweet_chars,
        encode_batch_size=64,
        device=device,
        model_name=model_name,
    )
    print("[VIZ-REBUILD] cluster_viz_data.json missing — re-encoding bundle …", flush=True)
    _tweet_uids, x_tweets, tweet_texts = extract_tweet_embeddings(bundle_path, cfg=cfg)
    x768 = x_tweets.astype(np.float32, copy=False)
    x64 = tweet_pca.transform(x768).astype(np.float32, copy=False)
    x64_norm = _normalize_rows(x64)
    tweet_cluster_ids = km.predict(x64_norm).astype(np.int32)
    n_tweets = int(x64.shape[0])
    best_k = int(meta.get("n_clusters", int(np.max(tweet_cluster_ids)) + 1))
    k_lo = int(meta.get("k_min", 8))
    k_hi = int(meta.get("k_max", 15))
    viz_data = _build_cluster_viz_data(
        x64,
        tweet_cluster_ids,
        tweet_texts,
        per_cluster_samples=10,
        random_state=42,
    )
    payload: Dict[str, Any] = {
        "best_k": best_k,
        "n_tweets": n_tweets,
        "k_min": k_lo,
        "k_max": k_hi,
        **viz_data,
    }
    os.makedirs(model_dir, exist_ok=True)
    with open(path, "wt", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("[VIZ-REBUILD] wrote cluster_viz_data.json", flush=True)
    return payload


def _build_user_multilabel_from_tweets(
    tweet_user_ids: Sequence[str],
    tweet_cluster_ids: np.ndarray,
    cluster_label_map: Dict[int, str],
    label_threshold: float,
    interest_labels: Sequence[str],
) -> Tuple[List[str], np.ndarray, List[Dict[str, float]]]:
    """
    每个用户：按「兴趣标签」聚合簇比例；Noise 簇不计入分子与分母（分母为有效微博条数）。
    比例>=threshold → 1。
    """
    by_user: Dict[str, List[int]] = defaultdict(list)
    for i, u in enumerate(tweet_user_ids):
        by_user[str(u).strip()].append(i)

    users_sorted = sorted(by_user.keys())
    n_labels = len(interest_labels)
    label_to_idx = {lab: i for i, lab in enumerate(interest_labels)}
    Y = np.zeros((len(users_sorted), n_labels), dtype=np.int32)
    ratios_list: List[Dict[str, float]] = []

    for ui, uid in enumerate(users_sorted):
        idxs = by_user[uid]
        counts = Counter()
        n_eff = 0
        for j in idxs:
            cid = int(tweet_cluster_ids[j])
            lab = cluster_label_map.get(cid, "其他")
            if lab == NOISE_LABEL or lab == "日常杂谈":
                continue
            n_eff += 1
            if lab in label_to_idx:
                counts[lab] += 1
        ratios: Dict[str, float] = {}
        for lab in interest_labels:
            ratios[lab] = counts.get(lab, 0) / float(n_eff) if n_eff else 0.0
        ratios_list.append(ratios)
        for lab in interest_labels:
            if ratios[lab] >= float(label_threshold):
                Y[ui, label_to_idx[lab]] = 1

    return users_sorted, Y, ratios_list


class TweetMultilabelGBDT:
    """
    每个兴趣维一个 GBDT；若该维全 0 或全 1（单类别），则跳过拟合用常数预测，避免 sklearn 报错。
    """

    def __init__(
        self,
        estimators: List[Optional[GradientBoostingClassifier]],
        constants: List[int],
    ):
        self.estimators = estimators
        self.constants = constants

    def predict(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        out = np.zeros((n, len(self.estimators)), dtype=np.int32)
        for j, (est, c) in enumerate(zip(self.estimators, self.constants)):
            if est is None:
                out[:, j] = np.int32(c)
            else:
                out[:, j] = est.predict(x).astype(np.int32)
        return out

    def predict_proba_per_label(self, x: np.ndarray) -> List[np.ndarray]:
        """与 MultiOutputClassifier.predict_proba 对齐：每列 (n, 2)"""
        n = x.shape[0]
        probs: List[np.ndarray] = []
        for j, (est, c) in enumerate(zip(self.estimators, self.constants)):
            if est is None:
                p1 = float(np.clip(c, 0, 1))
                probs.append(
                    np.column_stack([np.full(n, 1.0 - p1), np.full(n, p1)]).astype(np.float64)
                )
            else:
                pr = est.predict_proba(x)
                if pr.shape[1] == 1:
                    probs.append(np.column_stack([np.ones(n), np.zeros(n)]).astype(np.float64))
                else:
                    probs.append(pr.astype(np.float64))
        return probs


def _fit_multilabel_gbdt(
    x_train: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: int,
    n_estimators: int,
) -> TweetMultilabelGBDT:
    estimators: List[Optional[GradientBoostingClassifier]] = []
    constants: List[int] = []
    for j in range(y.shape[1]):
        col = y[:, j].astype(np.int32, copy=False)
        pos = int(col.sum())
        n = int(col.shape[0])
        if pos == 0:
            estimators.append(None)
            constants.append(0)
        elif pos >= n:
            estimators.append(None)
            constants.append(1)
        else:
            est = GradientBoostingClassifier(
                random_state=42,
                max_depth=int(max_depth),
                n_estimators=int(n_estimators),
                learning_rate=0.08,
            )
            est.fit(x_train, col)
            estimators.append(est)
            constants.append(0)
    return TweetMultilabelGBDT(estimators, constants)


def _resolve_local_model_dir(model_name: str) -> str:
    m = (model_name or "").strip()
    if not m:
        return m
    if os.path.isdir(m):
        return os.path.abspath(m)
    abs_m = os.path.abspath(m)
    if os.path.isdir(abs_m):
        return abs_m
    return m


def train_tweet_topic_pipeline(
    bundle_path: str,
    output_dir: str,
    *,
    k_min: int = 8,
    k_max: int = 15,
    label_threshold: float = 0.1,
    cluster_label_sim_threshold: float = 0.12,
    viz_only: bool = False,
    device: str = "cuda",
    model_name: str = "shibing624/text2vec-base-chinese",
    encode_batch_size: int = 64,
    min_tweet_chars: int = 5,
    gbdt_max_depth: int = 5,
    gbdt_n_estimators: int = 150,
    with_static_features: bool = True,
    max_tweets: Optional[int] = None,
) -> Dict:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model_name = _resolve_local_model_dir(model_name)

    cfg = TweetEmbeddingConfig(
        min_tweet_chars=min_tweet_chars,
        encode_batch_size=encode_batch_size,
        device=device,
        model_name=model_name,
    )
    print("[STEP1] encoding per-tweet vectors …", flush=True)
    tweet_user_ids, x_tweets, tweet_texts = extract_tweet_embeddings(
        bundle_path, cfg=cfg, max_tweets=max_tweets
    )
    n_tweets = x_tweets.shape[0]
    if n_tweets < 2:
        raise ValueError("有效微博不足，无法聚类。")

    # Strict order (aligned with kmeans_tweet_topic_debug.py):
    #   768-dim -> PCA(64) -> L2 normalize -> silhouette k in [k_min,k_max] -> KMeans
    pca_n = 64
    if n_tweets <= pca_n:
        raise ValueError(f"Need > PCA dim ({pca_n}) samples for PCA, got n_tweets={n_tweets}")

    x768 = x_tweets.astype(np.float32, copy=False)
    print("[STEP2] PCA(64) -> L2 normalize -> silhouette search k …", flush=True)
    tweet_pca = PCA(n_components=pca_n, random_state=42)
    x64 = tweet_pca.fit_transform(x768).astype(np.float32, copy=False)
    x64_norm = _normalize_rows(x64)

    k_hi = min(int(k_max), max(2, n_tweets))
    k_lo = max(2, int(k_min))
    if k_lo > k_hi:
        k_lo = k_hi

    idx_sub = _subsample_for_score(n_tweets, max_n=min(8000, n_tweets))
    x_sub = x64_norm[idx_sub]
    best_k = int(k_lo)
    best_score = -1.0
    k_search_results: List[Dict[str, Any]] = []
    for k in range(k_lo, k_hi + 1):
        if k <= 1 or k > len(x_sub):
            continue
        km_try = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_sub = km_try.fit_predict(x_sub)
        if len(set(labels_sub)) < 2:
            continue
        try:
            score = float(silhouette_score(x_sub, labels_sub, metric="cosine"))
        except Exception:
            score = -1.0
        print(f"[KSEARCH] k={k} silhouette={score}", flush=True)
        k_search_results.append({"k": k, "silhouette": score})
        if score > best_score:
            best_score = score
            best_k = int(k)
    if best_score < 0.0 and k_lo <= k_hi:
        best_k = int((k_lo + k_hi) // 2)
        print(f"[KSEARCH] fallback best_k={best_k} (silhouette unavailable)", flush=True)

    print(f"[STEP2b] KMeans fit k={best_k} …", flush=True)
    km = KMeans(n_clusters=int(best_k), random_state=42, n_init=10)
    km.fit(x64_norm)
    tweet_cluster_ids = km.predict(x64_norm).astype(np.int32)

    # 可视化数据：2D 散点 + 每簇样本文本
    print("[STEP2c] building cluster_viz_data.json …", flush=True)
    viz_data = _build_cluster_viz_data(
        x64,
        tweet_cluster_ids,
        tweet_texts,
        per_cluster_samples=10,
        random_state=42,
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "cluster_viz_data.json"), "wt", encoding="utf-8") as f:
        json.dump(
            {
                "best_k": int(best_k),
                "n_tweets": int(n_tweets),
                "k_min": int(k_lo),
                "k_max": int(k_hi),
                **viz_data,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    # 若仅用于聚类可视化，则在此提前返回（不做标签映射与 GBDT）
    if viz_only:
        with open(os.path.join(output_dir, "kmeans_k.json"), "wt", encoding="utf-8") as f:
            json.dump(
                {
                    "n_clusters": int(best_k),
                    "k_min": int(k_lo),
                    "k_max": int(k_hi),
                    "k_search_results": k_search_results,
                    "pca_n_components": int(pca_n),
                    "n_tweets": int(n_tweets),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return {
            "pipeline": "tweet_topic_multilabel_viz_only",
            "n_tweets": int(n_tweets),
            "n_clusters": int(best_k),
            "k_min": int(k_lo),
            "k_max": int(k_hi),
        }

    print("[STEP3] DeepSeek 命名簇标签（训练标签维度将由 AI 标签动态确定）…", flush=True)
    api_key = resolve_deepseek_api_key("")
    if not api_key:
        raise RuntimeError(
            "未配置 DeepSeek Key：请创建项目根目录 deepseek_api_key.txt 或设置 DEEPSEEK_API_KEY。"
        )
    llm_clusters = generate_tweet_topic_cluster_llm(
        output_dir,
        api_key=api_key,
        base_url=(os.environ.get("DEEPSEEK_BASE_URL", "") or "https://api.deepseek.com").strip(),
        model=(os.environ.get("DEEPSEEK_MODEL", "") or "deepseek-chat").strip(),
        force=False,
    )
    cluster_label_map: Dict[int, str] = {}
    for cid in sorted({int(x) for x in tweet_cluster_ids.tolist()}):
        item = llm_clusters.get(str(cid)) or {}
        raw_label = str(item.get("label", "")).strip() or f"簇{cid}"
        cluster_label_map[int(cid)] = raw_label

    interest_labels, noise_cluster_ids = _build_dynamic_labels(cluster_label_map)
    if not interest_labels:
        raise RuntimeError("DeepSeek 命名结果全为 Noise/无效标签，无法训练动态多标签 GBDT。")

    users_ml, Y, _ratios = _build_user_multilabel_from_tweets(
        tweet_user_ids,
        tweet_cluster_ids,
        {int(k): v for k, v in cluster_label_map.items()},
        label_threshold=label_threshold,
        interest_labels=interest_labels,
    )
    if len(users_ml) < 2:
        raise ValueError("聚类后有效用户不足（少于 2），无法训练 GBDT。")

    print("[STEP3] user multilabel from topic proportions (threshold=%s)" % label_threshold, flush=True)

    u_set = set(users_ml)
    uid_to_rows: Dict[str, List[int]] = defaultdict(list)
    for i, u in enumerate(tweet_user_ids):
        if str(u).strip() in u_set:
            uid_to_rows[str(u).strip()].append(i)

    mean_vecs: List[np.ndarray] = []
    for uid in users_ml:
        rows = uid_to_rows[uid]
        mean_vecs.append(x_tweets[rows].mean(axis=0))
    x_mean = np.stack(mean_vecs, axis=0).astype(np.float32, copy=False)

    if with_static_features:
        static_mat, static_names = static_features_for_users(bundle_path, users_ml)
        X_train = np.concatenate([x_mean, static_mat], axis=1)
    else:
        static_names = []
        X_train = x_mean

    print(
        f"[STEP4] GBDT per-label fit: samples={X_train.shape[0]} dim={X_train.shape[1]} labels={Y.sum(axis=0).tolist()}",
        flush=True,
    )
    clf = _fit_multilabel_gbdt(
        X_train,
        Y,
        max_depth=gbdt_max_depth,
        n_estimators=gbdt_n_estimators,
    )
    print("[GBDT] fitted TweetMultilabelGBDT", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "kmeans_tweet_model.pkl"), "wb") as f:
        pickle.dump(km, f)
    with open(os.path.join(output_dir, "tweet_pca_64.pkl"), "wb") as f:
        pickle.dump(tweet_pca, f)
    with open(os.path.join(output_dir, "gbdt_multilabel.pkl"), "wb") as f:
        pickle.dump(
            {
                "_gbdt_artifact_version": 1,
                "estimators": clf.estimators,
                "constants": clf.constants,
            },
            f,
        )
    with open(os.path.join(output_dir, "cluster_label_map.json"), "wt", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in cluster_label_map.items()}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "active_labels.json"), "wt", encoding="utf-8") as f:
        json.dump(
            {
                "active_labels": list(interest_labels),
                "noise_cluster_ids": [int(x) for x in noise_cluster_ids],
                "updated_at": int(time.time()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(output_dir, "kmeans_k.json"), "wt", encoding="utf-8") as f:
        json.dump(
            {
                "n_clusters": int(best_k),
                "k_min": int(k_lo),
                "k_max": int(k_hi),
                "k_search_results": k_search_results,
                "pca_n_components": int(pca_n),
                "n_tweets": int(n_tweets),
                "noise_cluster_ids": [int(x) for x in noise_cluster_ids],
                "cluster_label_sim_threshold": float(cluster_label_sim_threshold),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    pipeline_meta = {
        "pipeline": "tweet_topic_multilabel",
        "label_threshold": float(label_threshold),
        "cluster_label_sim_threshold": float(cluster_label_sim_threshold),
        "noise_cluster_ids": [int(x) for x in noise_cluster_ids],
        "k_min": int(k_lo),
        "k_max": int(k_hi),
        "interest_labels": list(interest_labels),
        "with_static_features": bool(with_static_features),
        "static_feature_names": static_names,
        "embed_dim": int(x_tweets.shape[1]),
        "pca_n_components": int(pca_n),
        "pca_order": "PCA(64)->L2->silhouette(k)->KMeans",
        "n_clusters": int(best_k),
        "gbdt_max_depth": int(gbdt_max_depth),
        "gbdt_n_estimators": int(gbdt_n_estimators),
        "device": device,
        "model_name": model_name,
        "min_tweet_chars": int(min_tweet_chars),
    }
    with open(os.path.join(output_dir, "training_pipeline.json"), "wt", encoding="utf-8") as f:
        json.dump(pipeline_meta, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "source_bundle_path.txt"), "wt", encoding="utf-8") as f:
        f.write(os.path.abspath(bundle_path))

    # 缓存训练期用户矩阵，便于 infer 在未改 bundle 时快速对齐（可选）
    np.save(os.path.join(output_dir, "train_user_mean_features.npy"), x_mean.astype(np.float32, copy=False))
    with open(os.path.join(output_dir, "train_user_ids.pkl"), "wb") as f:
        pickle.dump(users_ml, f)

    non_noise_vals = {v for v in cluster_label_map.values() if not _is_noise_label_name(v)}
    label_cov = len(non_noise_vals) / float(max(1, len(interest_labels)))
    best_sil: Optional[float] = None
    for row in k_search_results:
        if int(row.get("k", -1)) == int(best_k):
            v = row.get("silhouette")
            best_sil = float(v) if v is not None else None
            break
    metrics = {
        "pipeline": "tweet_topic_multilabel",
        "n_tweets": int(n_tweets),
        "n_users_train": int(len(users_ml)),
        "n_clusters": int(best_k),
        "silhouette_score": best_sil,
        "k_search_results": k_search_results,
        "noise_cluster_ids": [int(x) for x in noise_cluster_ids],
        "cluster_label_sim_threshold": float(cluster_label_sim_threshold),
        "label_coverage": float(label_cov),
        "label_threshold": float(label_threshold),
        "y_positive_counts": [int(x) for x in Y.sum(axis=0).tolist()],
        "device": device,
        "model_name": model_name,
    }
    with open(os.path.join(output_dir, "metrics.json"), "wt", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # 移除旧版 user-level 产物，避免 infer 误判
    for legacy in ("kmeans_model.pkl", "decision_tree_model.pkl", "user_features.npy", "user_ids.pkl", "cluster_ids.npy"):
        p = os.path.join(output_dir, legacy)
        if os.path.isfile(p):
            try:
                os.remove(p)
            except OSError:
                pass

    return metrics


def resolve_default_embed_model(project_root: str) -> str:
    env = (os.environ.get("WEIBO_EMBED_MODEL") or "").strip()
    if env:
        return env
    local = os.path.join(project_root, "models", "text2vec-base-chinese")
    if os.path.isdir(local):
        return os.path.abspath(local)
    return "shibing624/text2vec-base-chinese"


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser = argparse.ArgumentParser(description="Tweet-level KMeans + user multilabel GBDT.")
    parser.add_argument("--input", required=True, help="weibo_crawl_latest.json bundle path")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "output", "ml_artifacts"))
    parser.add_argument(
        "--k-min",
        type=int,
        default=8,
        help="话题簇下限（默认 8，避免宏观 2~3 类）",
    )
    parser.add_argument("--k-max", type=int, default=15, help="话题簇上限（默认 15）")
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.1,
        help="话题占比超过该值则计为正标签（K 较大时默认 0.1）",
    )
    parser.add_argument(
        "--cluster-label-sim-threshold",
        type=float,
        default=0.12,
        help="簇中心与标签向量余弦相似度低于此值则标为 Noise，不参与用户多标签比例",
    )
    parser.add_argument(
        "--viz-only",
        action="store_true",
        help="仅做推文聚类与可视化（cluster_viz_data.json），跳过标签映射与 GBDT 训练",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="sentence-transformers 模型路径或 HF 名；默认识别项目下 models/text2vec-base-chinese",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=32,
        help="RTX 3050 4GB 建议 24~48；过大易 OOM",
    )
    parser.add_argument("--min-tweet-chars", type=int, default=5)
    parser.add_argument("--gbdt-max-depth", type=int, default=5)
    parser.add_argument("--gbdt-n-estimators", type=int, default=150)
    parser.add_argument("--max-tweets", type=int, default=0, help="0=不限制；用于调试")
    parser.add_argument("--no-static-features", action="store_true", help="仅用语义均值向量，不加粉丝等")
    args = parser.parse_args()

    model_name = (args.model or "").strip() or resolve_default_embed_model(PROJECT_ROOT)
    max_tweets = int(args.max_tweets) if int(args.max_tweets) > 0 else None
    m = train_tweet_topic_pipeline(
        args.input,
        args.output_dir,
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        label_threshold=float(args.label_threshold),
        cluster_label_sim_threshold=float(args.cluster_label_sim_threshold),
        viz_only=bool(args.viz_only),
        device=args.device,
        model_name=model_name,
        encode_batch_size=int(args.encode_batch_size),
        min_tweet_chars=int(args.min_tweet_chars),
        gbdt_max_depth=int(args.gbdt_max_depth),
        gbdt_n_estimators=int(args.gbdt_n_estimators),
        with_static_features=not bool(args.no_static_features),
        max_tweets=max_tweets,
    )
    print(json.dumps(m, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
