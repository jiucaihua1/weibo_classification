import argparse
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans


def main() -> None:
    ap = argparse.ArgumentParser(description="PCA + t-SNE visualize + KMeans clustering for user embeddings.")
    ap.add_argument(
        "--features",
        default=os.path.join("output", "text2vec_20260402224047", "user_features_text2vec.npy"),
        help="Path to user feature matrix .npy (n_users x dim).",
    )
    ap.add_argument(
        "--user-ids",
        default=os.path.join("output", "text2vec_20260402224047", "user_ids_text2vec.pkl"),
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


if __name__ == "__main__":
    main()

