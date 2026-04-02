import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List

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
    unified_path: str,
    output_dir: str,
    n_clusters: int = 8,
    *,
    top_k_keywords: int = 10,
    min_chars: int = 200,
    device: str = "cuda",
    encode_batch_size: int = 64,
    user_batch_size: int = 64,
    model_name: str = "shibing624/text2vec-base-chinese",
) -> Dict:
    """
    覆盖替换：用 TextRank + text2vec 用户向量替换 TF-IDF。
    """
    cfg = FeatureExtractionConfig(
        top_k=top_k_keywords,
        min_chars=min_chars,
        encode_batch_size=encode_batch_size,
        device=device,
        model_name=model_name,
        user_batch_size=user_batch_size,
    )
    user_ids, x = extract_user_features(unified_path, cfg=cfg)
    if x.shape[0] < 2:
        raise ValueError("用户特征太少，无法聚类。")

    k = min(n_clusters, x.shape[0])
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = km.fit_predict(x)

    # label 映射
    cluster_label_map = map_clusters_to_labels_by_hint_embeddings(
        km.cluster_centers_,
        device=device,
        embed_model_name=model_name,
    )

    score = silhouette_score(x, cluster_ids) if len(set(cluster_ids)) >= 2 else -1.0
    coverage = len(set(cluster_label_map.values())) / float(len(INTEREST_LABELS))

    # 训练决策树（伪标签来自 KMeans）用于可解释分类
    tree = DecisionTreeClassifier(
        max_depth=8,
        random_state=42,
        min_samples_leaf=2,
    )
    tree.fit(x, cluster_ids)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(km, f)
    with open(os.path.join(output_dir, "cluster_label_map.json"), "wt", encoding="utf-8") as f:
        json.dump(cluster_label_map, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "decision_tree_model.pkl"), "wb") as f:
        pickle.dump(tree, f)

    with open(os.path.join(output_dir, "user_cluster.json"), "wt", encoding="utf-8") as f:
        json.dump({uid: int(cid) for uid, cid in zip(user_ids, cluster_ids)}, f, ensure_ascii=False, indent=2)

    # 缓存用户特征，推断阶段可直接复用（避免再跑 TextRank + BERT）
    np.save(os.path.join(output_dir, "user_features.npy"), x.astype(np.float32, copy=False))
    with open(os.path.join(output_dir, "user_ids.pkl"), "wb") as f:
        pickle.dump(user_ids, f)
    with open(os.path.join(output_dir, "source_bundle_path.txt"), "wt", encoding="utf-8") as f:
        f.write(os.path.abspath(unified_path))

    label_counter = Counter(cluster_label_map[int(c)] for c in cluster_ids)
    metrics = {
        "n_users": int(x.shape[0]),
        "n_clusters": int(k),
        "silhouette_score": float(score),
        "label_coverage": float(coverage),
        "label_distribution": dict(label_counter),
        "decision_tree_max_depth": 8,
        "embed_dim": int(x.shape[1]),
        "top_k_keywords": int(top_k_keywords),
        "min_chars": int(min_chars),
        "device": device,
        "model_name": model_name,
    }
    with open(os.path.join(output_dir, "metrics.json"), "wt", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TextRank + text2vec + KMeans hybrid interest classifier.")
    parser.add_argument("--input", required=True, help="Path to weibo_crawl_latest.json bundle.")
    parser.add_argument("--output-dir", default="output/ml_artifacts", help="Model output directory.")
    parser.add_argument("--clusters", type=int, default=8, help="Number of KMeans clusters.")
    parser.add_argument("--top-k", type=int, default=10, help="TextRank topK keywords per user.")
    parser.add_argument("--min-chars", type=int, default=200, help="Drop users if cleaned text length < threshold.")
    parser.add_argument("--device", type=str, default="cuda", help="cuda for RTX3050 acceleration.")
    parser.add_argument("--encode-batch-size", type=int, default=64, help="sentence-transformers encode batch_size.")
    parser.add_argument("--user-batch-size", type=int, default=64, help="How many users per encode flush.")
    args = parser.parse_args()
    result = train(
        args.input,
        args.output_dir,
        n_clusters=args.clusters,
        top_k_keywords=args.top_k,
        min_chars=args.min_chars,
        device=args.device,
        encode_batch_size=args.encode_batch_size,
        user_batch_size=args.user_batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
