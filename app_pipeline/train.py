import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier

from sentence_transformers import SentenceTransformer

from app_pipeline.feature_bert_textrank import FeatureExtractionConfig, extract_user_features


INTEREST_LABELS = ["科技", "财经", "汽车", "体育", "娱乐", "教育", "旅游", "美食"]
KEYWORD_HINTS = {
    "科技": {"ai", "算法", "芯片", "程序", "数码", "科技", "互联网"},
    "财经": {"股", "基金", "投资", "经济", "财报", "金融", "理财"},
    "汽车": {"汽车", "新能源", "电车", "驾驶", "油耗", "车型", "车企"},
    "体育": {"比赛", "球队", "足球", "篮球", "冠军", "运动", "联赛"},
    "娱乐": {"综艺", "电影", "明星", "演唱会", "娱乐", "剧集", "票房"},
    "教育": {"教育", "学校", "老师", "课程", "考试", "学生", "学习"},
    "旅游": {"旅行", "景点", "酒店", "攻略", "出发", "打卡", "游玩"},
    "美食": {"美食", "餐厅", "菜谱", "好吃", "火锅", "小吃", "探店"},
}


def _cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (n, dim), b: (m, dim)
    return: (n, m)
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization. Used so cosine-ish distances behave well with KMeans."""
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _load_features_and_ids(
    features_path: str,
    user_ids_path: str,
) -> Tuple[List[str], np.ndarray]:
    if not os.path.isfile(features_path):
        raise FileNotFoundError(f"features not found: {features_path}")
    if not os.path.isfile(user_ids_path):
        raise FileNotFoundError(f"user_ids not found: {user_ids_path}")

    x = np.load(features_path).astype(np.float32, copy=False)
    with open(user_ids_path, "rb") as f:
        user_ids = pickle.load(f)
    if not isinstance(user_ids, list):
        raise ValueError("user_ids.pkl must contain a list")
    if x.ndim != 2:
        raise ValueError(f"features must be 2D, got {x.shape}")
    if x.shape[0] != len(user_ids):
        raise ValueError(f"features rows {x.shape[0]} != user_ids len {len(user_ids)}")
    return [str(u) for u in user_ids], x


def map_clusters_to_labels_by_hint_embeddings(
    cluster_centers: np.ndarray,
    *,
    device: str,
    embed_model_name: str,
) -> Dict[int, str]:
    """
    将 KMeans 簇中心映射到兴趣标签：
    - 对每个兴趣标签，把 KEYWORD_HINTS 中的词拼成文本，encode 得到标签向量
    - 对每个簇中心向量做 cosine similarity，选择最相似的标签
    """
    st = SentenceTransformer(embed_model_name, device=device)
    label_texts: List[str] = []
    labels: List[str] = []
    for lab, words in KEYWORD_HINTS.items():
        labels.append(lab)
        # 用空格拼接，让模型把这些关键词当作一个短语语义
        label_texts.append(" ".join(sorted(words)))

    label_vecs = st.encode(
        label_texts,
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32, copy=False)

    sims = _cosine_similarity_matrix(cluster_centers.astype(np.float32, copy=False), label_vecs)
    mapping: Dict[int, str] = {}
    for cluster_id in range(sims.shape[0]):
        best_idx = int(np.argmax(sims[cluster_id]))
        best_label = labels[best_idx]
        mapping[cluster_id] = best_label
    return mapping


def train(
    bundle_path: str,
    output_dir: str,
    *,
    # read features produced by step3
    features_path: Optional[str] = None,
    user_ids_path: Optional[str] = None,
    # auto k
    k_min: int = 2,
    k_max: int = 10,
    # fallback options (if features not provided)
    top_k_keywords: int = 10,
    min_chars: int = 200,
    device: str = "cuda",
    encode_batch_size: int = 64,
    user_batch_size: int = 64,
    model_name: str = "shibing624/text2vec-base-chinese",
    # tree
    tree_max_depth: int = 8,
) -> Dict:
    # Make HF downloads use mirror by default (helps when model cache is missing).
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if features_path and user_ids_path:
        user_ids, x = _load_features_and_ids(features_path, user_ids_path)
    else:
        # fallback: re-extract (slow, but keeps script usable standalone)
        cfg = FeatureExtractionConfig(
            top_k=top_k_keywords,
            min_chars=min_chars,
            encode_batch_size=encode_batch_size,
            device=device,
            model_name=model_name,
            user_batch_size=user_batch_size,
        )
        user_ids, x = extract_user_features(bundle_path, cfg=cfg)

    if x.shape[0] < 2:
        raise ValueError("用户特征太少，无法聚类。")

    x_norm = _normalize_rows(x.astype(np.float32, copy=False))

    # auto k search by silhouette (cosine distance)
    k_max = min(int(k_max), int(x_norm.shape[0]))
    k_min = max(2, int(k_min))
    if k_max < k_min:
        k_max = k_min

    best_k = k_min
    best_score = -1.0
    best_km: Optional[KMeans] = None
    k_search_results: List[Dict[str, float]] = []

    for k in range(k_min, k_max + 1):
        if k <= 1:
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_ids = km.fit_predict(x_norm)
        if len(set(cluster_ids)) < 2:
            continue
        try:
            score = float(silhouette_score(x_norm, cluster_ids, metric="cosine"))
        except Exception:
            score = -1.0
        print(f"[KSEARCH] k={k} silhouette={score}", flush=True)
        k_search_results.append({"k": float(k), "silhouette_score": score})
        if score > best_score:
            best_score = score
            best_k = k
            best_km = km
            print(f"[KSEARCH] best_so_far_k={best_k} best_score={best_score}", flush=True)

    if best_km is None:
        # fallback: just pick k_min
        best_km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(x_norm)
        cluster_ids = best_km.labels_
    else:
        cluster_ids = best_km.predict(x_norm)

    print(f"[KSEARCH] best_k={best_k} best_score={best_score}", flush=True)

    cluster_label_map = map_clusters_to_labels_by_hint_embeddings(
        best_km.cluster_centers_,
        device=device,
        embed_model_name=model_name,
    )
    print("[LABEL] cluster_label_map computed", flush=True)
    coverage = len(set(cluster_label_map.values())) / float(len(INTEREST_LABELS))

    # decision tree for interpretability (pseudo-label = cluster_id)
    tree = DecisionTreeClassifier(
        max_depth=int(tree_max_depth),
        random_state=42,
        min_samples_leaf=2,
    )
    tree.fit(x_norm, cluster_ids)
    print("[TREE] decision_tree fitted", flush=True)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(best_km, f)
    with open(os.path.join(output_dir, "kmeans_k.json"), "wt", encoding="utf-8") as f:
        json.dump({"best_k": int(best_k), "best_silhouette": float(best_score)}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "cluster_label_map.json"), "wt", encoding="utf-8") as f:
        json.dump(cluster_label_map, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "decision_tree_model.pkl"), "wb") as f:
        pickle.dump(tree, f)
    print("[SAVE] clustering artifacts saved", flush=True)

    # persistence of clustering results
    user_cluster_map = {uid: int(cid) for uid, cid in zip(user_ids, cluster_ids)}
    with open(os.path.join(output_dir, "user_cluster.json"), "wt", encoding="utf-8") as f:
        json.dump(user_cluster_map, f, ensure_ascii=False, indent=2)
    np.save(os.path.join(output_dir, "cluster_ids.npy"), cluster_ids.astype(np.int32, copy=False))

    # cache features for infer stage
    np.save(os.path.join(output_dir, "user_features.npy"), x_norm.astype(np.float32, copy=False))
    with open(os.path.join(output_dir, "user_ids.pkl"), "wb") as f:
        pickle.dump(user_ids, f)
    with open(os.path.join(output_dir, "source_bundle_path.txt"), "wt", encoding="utf-8") as f:
        f.write(os.path.abspath(bundle_path))

    label_counter = Counter(cluster_label_map[int(c)] for c in cluster_ids)
    metrics = {
        "n_users": int(x_norm.shape[0]),
        "n_clusters": int(best_k),
        "silhouette_score": float(best_score),
        "label_coverage": float(coverage),
        "label_distribution": dict(label_counter),
        "decision_tree_max_depth": int(tree_max_depth),
        "embed_dim": int(x_norm.shape[1]),
        "k_search_results": k_search_results,
        "device": device,
        "model_name": model_name,
        "features_loaded": bool(features_path and user_ids_path),
    }
    with open(os.path.join(output_dir, "metrics.json"), "wt", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


if __name__ == "__main__":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    DEFAULT_FEATURES = os.path.join(PROJECT_ROOT, "output", "user_features_text2vec.npy")
    DEFAULT_IDS = os.path.join(PROJECT_ROOT, "output", "user_ids_text2vec.pkl")
    parser = argparse.ArgumentParser(description="KMeans cluster on text2vec user features, with auto-k.")
    parser.add_argument("--input", required=True, help="bundle path for evidence (e.g. output/weibo_crawl_latest.json)")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "output", "ml_artifacts"), help="Model output directory.")
    parser.add_argument("--features", default=DEFAULT_FEATURES, help="user_features_text2vec.npy")
    parser.add_argument("--user-ids", default=DEFAULT_IDS, help="user_ids_text2vec.pkl")
    parser.add_argument("--k-min", type=int, default=2, help="min k for auto selection")
    parser.add_argument("--k-max", type=int, default=10, help="max k for auto selection")
    parser.add_argument("--tree-max-depth", type=int, default=8, help="decision tree max depth")

    # fallback options (used only if features not provided / missing)
    parser.add_argument("--top-k", type=int, default=10, help="TextRank topK keywords per user (fallback mode)")
    parser.add_argument("--min-chars", type=int, default=200, help="Drop users if cleaned text length < threshold (fallback mode)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda for RTX3050 acceleration.")
    parser.add_argument("--encode-batch-size", type=int, default=64, help="sentence-transformers encode batch_size (fallback mode)")
    parser.add_argument("--user-batch-size", type=int, default=64, help="How many users per encode flush (fallback mode)")
    args = parser.parse_args()

    result = train(
        args.input,
        args.output_dir,
        features_path=args.features,
        user_ids_path=args.user_ids,
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        tree_max_depth=int(args.tree_max_depth),
        device=args.device,
        top_k_keywords=int(args.top_k),
        min_chars=int(args.min_chars),
        encode_batch_size=int(args.encode_batch_size),
        user_batch_size=int(args.user_batch_size),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
