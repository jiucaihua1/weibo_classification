import argparse
import json
import os
import pickle

import jieba
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE

# 常见中文停用词（可按需扩充）
_STOP_WORDS = set(
    ["的", "了", "是", "我", "在", "和", "就", "都", "也", "有", "与", "及", "等", "为", "对", "被","自己","就是"]
)


def _load_user_texts_from_cleaned_jsonl(path: str) -> dict[str, str]:
    by_uid: dict[str, str] = {}
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = str(obj.get("user_id", "")).strip()
            text = str(obj.get("cleaned_text", "") or "").strip()
            if uid and text:
                by_uid[uid] = text
    return by_uid


def get_top_keywords(text_list: list[str], top_n: int = 10, stop_words: set[str] | None = None) -> list[str]:
    stop = stop_words if stop_words is not None else _STOP_WORDS
    words_list: list[str] = []
    for text in text_list:
        words = [w for w in jieba.cut(str(text)) if w not in stop and len(w) > 1]
        words_list.append(" ".join(words))
    if not words_list or not any(words_list):
        return []
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(words_list)
    feature_names = vectorizer.get_feature_names_out()
    sum_tfidf = np.asarray(tfidf_matrix.sum(axis=0)).ravel()
    words_freq = [(feature_names[i], float(sum_tfidf[i])) for i in range(len(feature_names))]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return [word for word, _ in words_freq[:top_n]]


def main() -> None:
    ap = argparse.ArgumentParser(description="PCA + t-SNE visualize + KMeans clustering for user embeddings.")
    ap.add_argument(
        "--features",
        default=os.path.join("output", "text2vec_merged", "user_features_text2vec.npy"),
        help="Path to user feature matrix .npy (n_users x dim).",
    )
    ap.add_argument(
        "--user-ids",
        default=os.path.join("output", "text2vec_merged", "user_ids_text2vec.pkl"),
        help="Path to user id list .pkl (aligned with features rows).",
    )
    ap.add_argument("--pca-dim", type=int, default=50)
    ap.add_argument("--tsne-perplexity", type=float, default=30.0)
    ap.add_argument("--clusters", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=os.path.join("output", "viz_cluster_tsne.png"),
        help="Output image path (.png).",
    )
    ap.add_argument(
        "--texts-jsonl",
        default=os.path.join("output", "cleaned_user_texts_merged.jsonl"),
        help="cleaned_user_texts jsonl: fields user_id, cleaned_text (for per-cluster TF-IDF keywords).",
    )
    ap.add_argument(
        "--skip-keywords",
        action="store_true",
        help="Do not print TF-IDF keywords per cluster.",
    )
    ap.add_argument("--keyword-top-n", type=int, default=10)
    args = ap.parse_args()

    # 中文字体（若系统无 SimHei，会自动回退）
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    X_features = np.load(args.features)
    with open(args.user_ids, "rb") as f:
        user_ids = pickle.load(f)

    if X_features.ndim != 2:
        raise RuntimeError(f"Unexpected feature shape: {X_features.shape}")
    if len(user_ids) != X_features.shape[0]:
        raise RuntimeError(f"user_ids length mismatch: ids={len(user_ids)} features={X_features.shape[0]}")

    print(f"Loaded features: {X_features.shape}, users={len(user_ids)}")

    print("1. 正在进行 PCA 降维 (滤除噪音)...")
    pca_dim = min(int(args.pca_dim), X_features.shape[1], max(2, X_features.shape[0] - 1))
    pca = PCA(n_components=pca_dim, random_state=int(args.seed))
    X_pca = pca.fit_transform(X_features)

    print("2. 正在进行 t-SNE 降维 (为了画图)...")
    # t-SNE 的 perplexity 需要小于样本数
    perplexity = float(args.tsne_perplexity)
    perplexity = max(2.0, min(perplexity, max(2.0, (X_pca.shape[0] - 1) / 3.0)))
    tsne = TSNE(n_components=2, random_state=int(args.seed), perplexity=perplexity, init="pca", learning_rate="auto")
    X_2d = tsne.fit_transform(X_pca)

    print("3. 正在进行 K-Means 聚类...")
    n_clusters = max(2, int(args.clusters))
    kmeans = KMeans(n_clusters=n_clusters, random_state=int(args.seed), n_init="auto")
    labels = kmeans.fit_predict(X_pca)

    print("4. 正在绘制聚类分布图...")
    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap("tab10", n_clusters)
    for i in range(n_clusters):
        mask = labels == i
        plt.scatter(
            X_2d[mask, 0],
            X_2d[mask, 1],
            c=[cmap(i)],
            label=f"兴趣群体 {i} (n={int(mask.sum())})",
            alpha=0.65,
            s=28,
            edgecolors="none",
        )

    plt.title("微博用户兴趣特征聚类分布图（t-SNE 可视化）")
    plt.xlabel("特征维度 1")
    plt.ylabel("特征维度 2")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.35)

    out_path = os.path.normpath(args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"[OK] Saved plot: {out_path}")

    # --- 5. 按聚类标签，用 TF-IDF + 结巴从「用户拼接文本」解释每个群体 ---
    if not args.skip_keywords:
        texts_path = os.path.normpath(args.texts_jsonl)
        if not os.path.isfile(texts_path):
            print(f"[WARN] 未找到文本文件，跳过关键词解释: {texts_path}")
        else:
            uid_to_text = _load_user_texts_from_cleaned_jsonl(texts_path)
            # 等价于 df: user_id, text, cluster_label
            rows: list[tuple[str, str, int]] = []
            for idx, uid in enumerate(user_ids):
                uid_s = str(uid).strip()
                t = uid_to_text.get(uid_s, "")
                if t:
                    rows.append((uid_s, t, int(labels[idx])))

            if not rows:
                print("[WARN] 没有任何用户能在 jsonl 中对齐到文本，跳过关键词解释。")
            else:
                print(f"\n5. 各兴趣群体核心关键词 (TF-IDF, 文本来自 {texts_path}, n={len(rows)} 用户有文)…")
                for i in range(n_clusters):
                    cluster_texts = [r[1] for r in rows if r[2] == i]
                    if not cluster_texts:
                        print(f"群体 {i}: （无可用文本）")
                        continue
                    top_words = get_top_keywords(cluster_texts, top_n=int(args.keyword_top_n))
                    print(f"群体 {i} 的核心关键词是: {top_words}")


if __name__ == "__main__":
    main()

