"""
K-Means 多标签（动态跨界门槛）

流程：
1) 距离矩阵：每个用户到各簇中心的欧氏距离（sklearn KMeans.transform）
2) 首要兴趣：距离最小的簇
3) 专属门槛：min_distance * tolerance_ratio（如 1.5）
4) 多标签：所有 distance <= 门槛 的簇编号
5) 写出 JSONL（可视为表的一列 multilabel_cluster_ids）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from app_pipeline.cluster_category_names import map_cluster_keywords_to_names
from app_pipeline.cluster_llm_labels import (
    multilabel_jsonl_fingerprint,
    resolve_deepseek_api_key,
    try_load_cached_cluster_llm,
)
from app_pipeline.cluster_keywords import (
    keywords_per_primary_cluster,
    load_user_texts_from_cleaned_jsonl,
    pick_default_cleaned_jsonl,
)
from app_pipeline.train import _load_features_and_ids, _normalize_rows


def _fit_or_load_kmeans(
    x: np.ndarray,
    *,
    n_clusters: int,
    seed: int,
    model_path: str | None,
) -> tuple[KMeans, int]:
    """
    Returns (fitted_kmeans, n_clusters_effective).
    若从文件加载，簇数以模型为准。
    """
    if model_path:
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"kmeans model not found: {model_path}")
        with open(model_path, "rb") as f:
            km = pickle.load(f)
        if not hasattr(km, "cluster_centers_"):
            raise ValueError("loaded object is not a fitted KMeans")
        k_model = int(km.cluster_centers_.shape[0])
        if int(n_clusters) != k_model:
            print(f"[INFO] --clusters={n_clusters} 与模型不一致，使用模型簇数 k={k_model}")
        if int(km.cluster_centers_.shape[1]) != int(x.shape[1]):
            raise ValueError(
                f"model dim={km.cluster_centers_.shape[1]} != features dim={x.shape[1]}"
            )
        return km, k_model
    km = KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init="auto")
    km.fit(x)
    return km, int(n_clusters)


def multilabel_from_distances(
    dist_matrix: np.ndarray,
    *,
    tolerance_ratio: float,
) -> tuple[list[list[int]], list[int], list[float], list[list[float]]]:
    """
    dist_matrix: (n_users, n_clusters) 每行到各簇心距离
    returns: labels_list, primary_list, threshold_list, distances_as_lists
    """
    if dist_matrix.ndim != 2:
        raise ValueError(f"dist_matrix must be 2D, got {dist_matrix.shape}")
    n_users, k = dist_matrix.shape
    ratio = float(tolerance_ratio)
    if ratio <= 0:
        raise ValueError("tolerance_ratio must be positive")

    all_labels: list[list[int]] = []
    primaries: list[int] = []
    thresholds: list[float] = []
    dist_rows: list[list[float]] = []

    for i in range(n_users):
        d = dist_matrix[i].astype(np.float64, copy=False)
        min_d = float(np.min(d))
        primary = int(np.argmin(d))
        thresh = min_d * ratio
        picked = sorted(int(j) for j in range(k) if float(d[j]) <= thresh)
        all_labels.append(picked)
        primaries.append(primary)
        thresholds.append(thresh)
        dist_rows.append([float(x) for x in d.tolist()])
    return all_labels, primaries, thresholds, dist_rows


def main() -> int:
    ap = argparse.ArgumentParser(description="KMeans distance-based multi-label assignment.")
    ap.add_argument(
        "--features",
        default=os.path.join("output", "text2vec_merged", "user_features_text2vec.npy"),
    )
    ap.add_argument(
        "--user-ids",
        default=os.path.join("output", "text2vec_merged", "user_ids_text2vec.pkl"),
    )
    ap.add_argument("--clusters", type=int, default=5, help="KMeans K（与模型一致）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--kmeans-model",
        default="",
        help="若指定则加载已有 kmeans_model.pkl，否则在当前特征上重新 fit",
    )
    ap.add_argument(
        "--tolerance-ratio",
        type=float,
        default=1.2,
        help="跨界门槛 = 最小距离 * 该比例（如 1.2、1.5）",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        default=True,
        help="对特征按行 L2 归一化（与 app_pipeline/train.py 一致，默认开启）",
    )
    ap.add_argument(
        "--no-normalize",
        action="store_true",
        help="关闭行 L2 归一化",
    )
    ap.add_argument(
        "--out-jsonl",
        default=os.path.join("output", "kmeans_multilabel_users.jsonl"),
    )
    ap.add_argument(
        "--out-csv",
        default=os.path.join("output", "kmeans_multilabel_users.csv"),
    )
    ap.add_argument("--skip-csv", action="store_true")
    ap.add_argument(
        "--texts-jsonl",
        default="",
        help="含 user_id/cleaned_text 的 jsonl；留空则尝试 output/cleaned_user_texts_merged.jsonl 与 cleaned_user_texts.jsonl",
    )
    ap.add_argument(
        "--skip-keywords",
        action="store_true",
        help="不根据文本生成簇关键词、不写进 meta",
    )
    ap.add_argument("--keyword-top-n", type=int, default=10, help="每簇 TF-IDF 关键词条数")
    ap.add_argument(
        "--llm-labels",
        action="store_true",
        help="调用 DeepSeek 为每簇生成 4 字兴趣标签（Key：环境变量或项目根 deepseek_api_key.txt；从 unified_*.jsonl 抽样）",
    )
    ap.add_argument(
        "--deepseek-api-key",
        default="",
        help="若空则读 DEEPSEEK_API_KEY 或 deepseek_api_key.txt",
    )
    ap.add_argument(
        "--deepseek-key-file",
        default="",
        help="Key 文件路径（默认项目根目录 deepseek_api_key.txt）",
    )
    ap.add_argument(
        "--deepseek-base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
    )
    ap.add_argument(
        "--deepseek-model",
        default=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat").strip(),
    )
    ap.add_argument(
        "--llm-force",
        action="store_true",
        help="即使聚类文件指纹未变也重新调用 DeepSeek（默认会沿用 meta 中已保存的标签）",
    )
    args = ap.parse_args()

    ds_key = resolve_deepseek_api_key(args.deepseek_api_key, key_file=args.deepseek_key_file)

    normalize = bool(args.normalize) and not bool(args.no_normalize)
    user_ids, x = _load_features_and_ids(str(args.features), str(args.user_ids))
    if normalize:
        x = _normalize_rows(x)

    model_path = (args.kmeans_model or "").strip() or None
    km, n_clusters_eff = _fit_or_load_kmeans(
        x,
        n_clusters=int(args.clusters),
        seed=int(args.seed),
        model_path=model_path,
    )

    # 第一步：距离矩阵 (n_users, n_clusters) — 每列为到该簇心的欧氏距离
    dist_matrix = km.transform(x)
    labels_multi, primaries, thresholds, dist_rows = multilabel_from_distances(
        dist_matrix,
        tolerance_ratio=float(args.tolerance_ratio),
    )

    out_jsonl = os.path.normpath(str(args.out_jsonl))
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    meta: dict[str, Any] = {
        "n_users": len(user_ids),
        "n_clusters": n_clusters_eff,
        "tolerance_ratio": float(args.tolerance_ratio),
        "normalize_rows": normalize,
        "used_pretrained_kmeans": model_path is not None,
        "kmeans_model_path": model_path or "",
        "features": os.path.normpath(str(args.features)),
        "output_jsonl": out_jsonl,
    }

    texts_path = (args.texts_jsonl or "").strip()
    if not texts_path:
        out_dir = os.path.dirname(out_jsonl)
        if not out_dir or out_dir == ".":
            texts_path = pick_default_cleaned_jsonl("output")
        else:
            texts_path = pick_default_cleaned_jsonl(out_dir)
        if not texts_path and os.path.isdir("output"):
            texts_path = pick_default_cleaned_jsonl("output")

    cluster_kw: dict[int, list[str]] = {}
    if not args.skip_keywords:
        if not texts_path or not os.path.isfile(texts_path):
            print(f"[WARN] 未找到清洗文本 jsonl，跳过簇关键词: {texts_path or '(未配置)'}")
        else:
            uid_to_text = load_user_texts_from_cleaned_jsonl(texts_path)
            cluster_kw = keywords_per_primary_cluster(
                list(user_ids),
                primaries,
                uid_to_text,
                n_clusters=n_clusters_eff,
                top_n=int(args.keyword_top_n),
            )
            meta["texts_jsonl_for_keywords"] = os.path.normpath(texts_path)
            meta["cluster_keywords"] = {str(c): list(words) for c, words in sorted(cluster_kw.items())}
            meta["cluster_keyword_summaries"] = {
                str(c): " · ".join(words[:3]) if words else ""
                for c, words in sorted(cluster_kw.items())
            }
            # 词典匹配仅作兜底；若随后成功生成 cluster_llm_labels 会覆盖展示逻辑
            if not (args.llm_labels and ds_key):
                meta["cluster_display_names"] = map_cluster_keywords_to_names(meta["cluster_keywords"])
            print(f"\n各簇核心关键词 (TF-IDF, 文本来自 {texts_path}, 按主簇聚合)…")
            for c in range(n_clusters_eff):
                words = cluster_kw.get(c, [])
                if not words:
                    print(f"簇 {c}: （无可用文本）")
                else:
                    print(f"簇 {c} 的核心关键词是: {words}")
    else:
        print("[INFO] 已 --skip-keywords，不生成簇关键词。")

    with open(out_jsonl, "wt", encoding="utf-8") as f:
        for i, uid in enumerate(user_ids):
            row = {
                "user_id": uid,
                "primary_cluster": primaries[i],
                "multilabel_cluster_ids": labels_multi[i],
                "min_distance": float(np.min(dist_matrix[i])),
                "threshold": thresholds[i],
                "distances_to_centroids": dist_rows[i],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta_path_pre = os.path.join(os.path.dirname(out_jsonl) or ".", "kmeans_multilabel_meta.json")
    if args.llm_labels:
        if not ds_key:
            print(
                "[WARN] --llm-labels 已指定但未找到 Key：请设 DEEPSEEK_API_KEY，"
                "或在项目根创建 deepseek_api_key.txt（见 deepseek_api_key.txt.example）。"
            )
        else:
            try:
                from app_pipeline.cluster_llm_labels import generate_cluster_llm_labels

                out_dir_abs = os.path.dirname(os.path.abspath(out_jsonl))
                if not out_dir_abs:
                    out_dir_abs = os.getcwd()
                fp = multilabel_jsonl_fingerprint(out_jsonl)
                model_s = str(args.deepseek_model)
                cached = None if args.llm_force else try_load_cached_cluster_llm(meta_path_pre, fp, model_s)
                if cached is not None:
                    meta.update(cached)
                    meta.pop("cluster_display_names", None)
                    print("\n[LLM] 多标签结果文件指纹未变，沿用 meta 中已有 DeepSeek 标签（未调用 API）。")
                else:
                    print("\n[LLM] DeepSeek 生成簇四字标签（原始微博来自 unified_*.jsonl）…")
                    labels = generate_cluster_llm_labels(
                        list(user_ids),
                        primaries,
                        output_dir=out_dir_abs,
                        n_clusters=n_clusters_eff,
                        api_key=ds_key,
                        seed=int(args.seed),
                        base_url=str(args.deepseek_base_url),
                        model=model_s,
                    )
                    meta["cluster_llm_labels"] = labels
                    meta["cluster_llm_provider"] = "deepseek"
                    meta["cluster_llm_model"] = model_s
                    meta["cluster_llm_base_url"] = str(args.deepseek_base_url)
                    meta["cluster_llm_source_fingerprint"] = fp
                    meta.pop("cluster_display_names", None)
            except Exception as exc:
                print(f"[ERROR] LLM 簇标签失败: {exc}")
                if meta.get("cluster_keywords") and "cluster_display_names" not in meta:
                    meta["cluster_display_names"] = map_cluster_keywords_to_names(meta["cluster_keywords"])

    if not args.skip_csv:
        out_csv = os.path.normpath(str(args.out_csv))
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        k = dist_matrix.shape[1]
        dist_cols = [f"dist_cluster_{j}" for j in range(k)]
        with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            header = (
                ["user_id", "primary_cluster", "multilabel_cluster_ids", "min_distance", "threshold"]
                + dist_cols
            )
            w.writerow(header)
            for i, uid in enumerate(user_ids):
                ml = labels_multi[i]
                ml_str = json.dumps(ml, ensure_ascii=False)
                w.writerow(
                    [uid, primaries[i], ml_str, float(np.min(dist_matrix[i])), thresholds[i]]
                    + dist_rows[i]
                )
        meta["output_csv"] = out_csv

    meta_path = meta_path_pre
    with open(meta_path, "wt", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)
    meta["meta_file"] = meta_path

    print("[DONE] KMeans multi-label")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
