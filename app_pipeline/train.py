import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from app_pipeline.data_io import load_training_records
from app_pipeline.preprocess import clean_text, tokenize_zh


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


def build_user_documents(records: List[Dict]) -> Dict[str, str]:
    by_user = defaultdict(list)
    for item in records:
        cleaned = clean_text(item["text"])
        if cleaned:
            by_user[item["user_id"]].append(cleaned)
    return {uid: " ".join(parts) for uid, parts in by_user.items() if parts}


def map_clusters_to_labels(vectorizer: TfidfVectorizer, model: KMeans, topn: int = 30) -> Dict[int, str]:
    vocab = np.array(vectorizer.get_feature_names_out())
    mapping = {}
    for cluster_id, center in enumerate(model.cluster_centers_):
        top_indices = center.argsort()[::-1][:topn]
        top_terms = set(vocab[top_indices].tolist())
        label_scores = {label: len(top_terms.intersection(words)) for label, words in KEYWORD_HINTS.items()}
        best_label = max(label_scores, key=label_scores.get)
        if label_scores[best_label] == 0:
            best_label = INTEREST_LABELS[cluster_id % len(INTEREST_LABELS)]
        mapping[cluster_id] = best_label
    return mapping


def train(unified_path: str, output_dir: str, n_clusters: int = 8) -> Dict:
    records = load_training_records(unified_path)
    user_docs = build_user_documents(records)
    if len(user_docs) < 1:
        raise ValueError("用户文本为空，无法训练。")

    user_ids = list(user_docs.keys())
    docs = [user_docs[uid] for uid in user_ids]
    max_df = 0.9 if len(docs) > 1 else 1.0
    vectorizer = TfidfVectorizer(tokenizer=tokenize_zh, lowercase=False, min_df=1, max_df=max_df)
    x = vectorizer.fit_transform(docs)

    k = min(n_clusters, len(user_ids))
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_ids = model.fit_predict(x)
    cluster_label_map = map_clusters_to_labels(vectorizer, model)

    unique_labels = len(set(cluster_ids))
    score = silhouette_score(x, cluster_ids) if unique_labels >= 2 and len(user_ids) > unique_labels else -1.0
    coverage = len(set(cluster_label_map.values())) / float(len(INTEREST_LABELS))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(output_dir, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(output_dir, "cluster_label_map.json"), "wt", encoding="utf-8") as f:
        json.dump(cluster_label_map, f, ensure_ascii=False, indent=2)
    with open(os.path.join(output_dir, "user_cluster.json"), "wt", encoding="utf-8") as f:
        json.dump({uid: int(cid) for uid, cid in zip(user_ids, cluster_ids)}, f, ensure_ascii=False, indent=2)

    label_counter = Counter(cluster_label_map[int(c)] for c in cluster_ids)
    metrics = {
        "n_users": len(user_ids),
        "n_records": len(records),
        "n_clusters": k,
        "silhouette_score": float(score),
        "label_coverage": float(coverage),
        "label_distribution": dict(label_counter),
    }
    with open(os.path.join(output_dir, "metrics.json"), "wt", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TF-IDF + KMeans hybrid interest classifier.")
    parser.add_argument("--input", required=True, help="Path to unified jsonl or weibo_crawl_*.json bundle.")
    parser.add_argument("--output-dir", default="output/ml_artifacts", help="Model output directory.")
    parser.add_argument("--clusters", type=int, default=8, help="Number of KMeans clusters.")
    args = parser.parse_args()
    result = train(args.input, args.output_dir, n_clusters=args.clusters)
    print(json.dumps(result, ensure_ascii=False, indent=2))
