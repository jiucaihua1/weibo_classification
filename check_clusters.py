"""
check_clusters.py

簇诊断脚本（聚类纯度/可解释性“看他们在聊什么”）：

只做两步：
1) 对 `output/user_features_text2vec.npy` 做 L2 标准化后运行/或复用 KMeans 得到 cluster_id；
2) 对每个 cluster 内的用户，基于 `output/cleaned_user_texts.jsonl` 抽取 top keywords，
   统计每个簇最常出现的词（不使用任何 KEYWORD_HINTS、不训练任何分类器）。

输出：每个簇的规模 + Top 关键词列表
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter
from typing import Dict, List, Tuple

import jieba.analyse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def _p(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def _load_json(path: str) -> dict:
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_cleaned_texts(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = str(obj.get("user_id", "")).strip()
            txt = str(obj.get("cleaned_text", "") or "").strip()
            if uid and txt:
                out[uid] = txt
    return out


def _determine_k(best_k_path: str, fallback_k: int) -> int:
    if best_k_path and os.path.isfile(best_k_path):
        try:
            m = _load_json(best_k_path)
            return int(m.get("best_k", fallback_k))
        except Exception:
            return fallback_k
    return fallback_k


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose KMeans clusters by top words inside each cluster.")
    ap.add_argument("--features", default=_p("output/user_features_text2vec.npy"))
    ap.add_argument("--user-ids", default=_p("output/user_ids_text2vec.pkl"))
    ap.add_argument("--cleaned-jsonl", default=_p("output/cleaned_user_texts.jsonl"))
    ap.add_argument("--ml-artifacts-dir", default=_p("output/ml_artifacts"))
    ap.add_argument("--k", type=int, default=0, help="Override k. 0 means auto-read best_k.")
    ap.add_argument("--top-words", type=int, default=20)
    ap.add_argument("--keywords-per-user", type=int, default=20, help="textrank topK per user inside each cluster.")
    ap.add_argument("--out-diagnose-json", type=str, default=_p("output/cluster_keywords_diagnose.json"))
    ap.add_argument(
        "--max-users-per-cluster",
        type=int,
        default=0,
        help="For speed: limit how many users from each cluster to inspect. 0 means no limit.",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.features):
        raise FileNotFoundError(f"features not found: {args.features}")
    if not os.path.isfile(args.user_ids):
        raise FileNotFoundError(f"user_ids not found: {args.user_ids}")
    if not os.path.isfile(args.cleaned_jsonl):
        raise FileNotFoundError(f"cleaned jsonl not found: {args.cleaned_jsonl}")

    X = np.load(args.features).astype(np.float32, copy=False)
    user_ids: List[str] = _load_pickle(args.user_ids)
    user_ids = [str(u) for u in user_ids]
    if X.ndim != 2 or X.shape[0] != len(user_ids):
        raise RuntimeError(f"shape mismatch: X={X.shape}, user_ids={len(user_ids)}")

    cleaned_texts = _load_cleaned_texts(args.cleaned_jsonl)

    X_norm = normalize(X, norm="l2")

    # Determine k:
    # - if user overrides --k>0: we force-fit a new KMeans(n_clusters=k) for diagnosis
    # - else: we reuse best_k from artifacts, and if possible reuse the trained model
    best_k_path = os.path.join(args.ml_artifacts_dir, "kmeans_k.json")
    k_override = int(args.k) if int(args.k) > 0 else 0
    k = k_override if k_override > 0 else _determine_k(best_k_path, fallback_k=6)
    if k <= 1:
        raise RuntimeError(f"Invalid k={k}")

    pca_model_path = os.path.join(args.ml_artifacts_dir, "pca_model.pkl")
    pca_model = None
    if os.path.isfile(pca_model_path):
        try:
            pca_model = _load_pickle(pca_model_path)
        except Exception:
            pca_model = None

    X_km = X_norm
    if pca_model is not None:
        try:
            X_km = pca_model.transform(X_norm).astype(np.float32, copy=False)
            X_km = normalize(X_km, norm="l2")
        except Exception:
            X_km = X_norm

    km_path = os.path.join(args.ml_artifacts_dir, "kmeans_model.pkl")
    if k_override > 0:
        # Force-fit diagnosis KMeans.
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_km).astype(int)
    else:
        # Reuse trained model if present; otherwise fit.
        if os.path.isfile(km_path):
            try:
                km = _load_pickle(km_path)
                labels = km.predict(X_km).astype(int)
            except Exception:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_km).astype(int)
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_km).astype(int)

    cluster_sizes = Counter(labels.tolist())
    print(f"[OK] Loaded users={len(user_ids)}, k={k} (actual distinct={len(cluster_sizes)})")

    # Diagnose: for each cluster, gather keywords across users' cleaned_text
    cluster_to_words: Dict[int, Counter] = {cid: Counter() for cid in cluster_sizes.keys()}
    cluster_to_count: Dict[int, int] = {}

    # Build cluster -> members once (O(n)) instead of repeatedly scanning user_ids.
    cluster_to_members: Dict[int, List[str]] = {cid: [] for cid in cluster_sizes.keys()}
    for uid, cid in zip(user_ids, labels):
        cluster_to_members[int(cid)].append(uid)

    # Iterate clusters sorted by size (more stable output)
    for cid, _ in sorted(cluster_sizes.items(), key=lambda x: -x[1]):
        members = cluster_to_members.get(cid, [])
        if args.max_users_per_cluster and args.max_users_per_cluster > 0:
            members = members[: int(args.max_users_per_cluster)]

        cluster_to_count[cid] = len(members)
        if not members:
            continue

        print(f"[DIAG] Cluster {cid}: inspecting {len(members)} users...")
        for idx, uid in enumerate(members, start=1):
            txt = cleaned_texts.get(uid)
            if not txt:
                continue
            kws = jieba.analyse.textrank(txt, topK=int(args.keywords_per_user), withWeight=False) or []
            for w in kws:
                w = str(w).strip()
                if not w:
                    continue
                # remove ultra-low value generic words
                if w in {"没有", "未知", "大家", "可以"}:
                    continue
                cluster_to_words[cid][w] += 1
            if idx % 50 == 0:
                print(f"[DIAG] Cluster {cid}: processed {idx}/{len(members)}...")

    print("\n" + "=" * 70)
    print("Cluster Keywords Diagnostic")
    print("=" * 70)

    diagnose: Dict[str, dict] = {}
    for cid in sorted(cluster_sizes.keys()):
        size = cluster_sizes.get(cid, 0)
        top = cluster_to_words[cid].most_common(int(args.top_words))
        top_str = "、".join([f"{w}({c})" for w, c in top]) if top else "—"
        print(f"\n[Cluster {cid}] size={size} inspected={cluster_to_count.get(cid, 0)}")
        print(f"Top words: {top_str}")

        diagnose[str(cid)] = {
            "cluster_size": int(size),
            "inspected_users": int(cluster_to_count.get(cid, 0)),
            "top_words": [{"word": w, "count": int(c)} for w, c in top],
        }

    out_json = args.out_diagnose_json
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "wt", encoding="utf-8") as f:
        json.dump(
            {
                "k": int(k),
                "actual_clusters": sorted([int(x) for x in cluster_sizes.keys()]),
                "diagnose": diagnose,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[Saved] diagnose json: {out_json}")


if __name__ == "__main__":
    main()

