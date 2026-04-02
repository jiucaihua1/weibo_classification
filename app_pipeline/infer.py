import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from app_pipeline.data_io import iter_crawl_bundle_users
from app_pipeline.feature_bert_textrank import FeatureExtractionConfig, extract_user_features
from app_pipeline.preprocess import clean_text


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


def _is_verified_v(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    user = raw.get("user")
    if isinstance(user, dict):
        return bool(user.get("verified") is True)
    return bool(raw.get("verified") is True)


def _text_is_retweet(text: str) -> bool:
    if not text:
        return False
    t = str(text).strip()
    return t == "转发微博" or t.startswith("转发微博")


def infer(
    unified_path: str,
    model_dir: str,
    output_file: str,
) -> List[Dict]:
    """
    覆盖替换：用 TextRank + text2vec 用户向量 做 KMeans 聚类并输出兴趣画像。
    """
    with open(os.path.join(model_dir, "kmeans_model.pkl"), "rb") as f:
        km = pickle.load(f)
    with open(os.path.join(model_dir, "cluster_label_map.json"), "rt", encoding="utf-8") as f:
        cluster_map = {int(k): v for k, v in json.load(f).items()}

    # 若存在决策树模型，则用它做“分类/可解释预测”（树是用 KMeans 的伪标签训练出来的）
    tree_path = os.path.join(model_dir, "decision_tree_model.pkl")
    tree = None
    if os.path.isfile(tree_path):
        try:
            with open(tree_path, "rb") as f:
                tree = pickle.load(f)
        except Exception:
            tree = None

    # 尝试加载训练阶段缓存的用户特征（推断阶段不重复做 TextRank + BERT）
    cached_ids_path = os.path.join(model_dir, "user_ids.pkl")
    cached_feat_path = os.path.join(model_dir, "user_features.npy")
    cached_src_path = os.path.join(model_dir, "source_bundle_path.txt")

    input_abs = os.path.abspath(unified_path)
    use_cache = (
        os.path.isfile(cached_ids_path)
        and os.path.isfile(cached_feat_path)
        and os.path.isfile(cached_src_path)
    )
    user_ids: List[str] = []
    x: np.ndarray
    if use_cache:
        try:
            with open(cached_src_path, "rt", encoding="utf-8") as f:
                cached_src = f.read().strip()
            if cached_src == input_abs:
                import pickle as _pickle

                with open(cached_ids_path, "rb") as f:
                    user_ids = _pickle.load(f)
                x = np.load(cached_feat_path).astype(np.float32, copy=False)
            else:
                use_cache = False
        except Exception:
            use_cache = False

    if not use_cache:
        # 尝试从 metrics.json 里读取提取参数，保证与训练一致
        top_k_keywords = 10
        min_chars = 200
        device = "cuda"
        model_name = "shibing624/text2vec-base-chinese"
        if os.path.isfile(os.path.join(model_dir, "metrics.json")):
            try:
                with open(os.path.join(model_dir, "metrics.json"), "rt", encoding="utf-8") as f:
                    m = json.load(f)
                top_k_keywords = int(m.get("top_k_keywords", top_k_keywords))
                min_chars = int(m.get("min_chars", min_chars))
                device = str(m.get("device", device))
                model_name = str(m.get("model_name", model_name))
            except Exception:
                pass

        cfg = FeatureExtractionConfig(
            top_k=top_k_keywords,
            min_chars=min_chars,
            device=device,
            model_name=model_name,
            encode_batch_size=64,
            user_batch_size=64,
        )
        user_ids, x = extract_user_features(unified_path, cfg=cfg)

    if not user_ids:
        raise RuntimeError("infer: user_ids empty")

    # 预测簇（训练阶段对特征做了行归一化，所以推断也要一致）
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
    if tree is not None:
        cluster_ids = tree.predict(x_norm).astype(int)
    else:
        cluster_ids = km.predict(x_norm).astype(int)
    centroids = km.cluster_centers_.astype(np.float32, copy=False)
    c_norm = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12)
    cos_scores = np.sum(x_norm * c_norm[cluster_ids], axis=1)  # (n_users,)

    idx_map = {uid: i for i, uid in enumerate(user_ids)}

    print(f"[INFER] total_feature_users={len(idx_map)}", flush=True)

    # 证据抽取：流式遍历 bundle users，匹配到已通过过滤的 user_id
    profiles: List[Dict] = []
    processed_outputs = 0
    target_total = max(1, len(idx_map))
    for user_block in iter_crawl_bundle_users(unified_path):
        if not isinstance(user_block, dict):
            continue
        uid = str(user_block.get("user_id", "")).strip()
        if not uid or uid not in idx_map:
            continue

        records = user_block.get("records") or []
        if not isinstance(records, list) or not records:
            continue

        # 若用户在特征提取阶段是被过滤掉的，这里可能仍会出现；
        # 但我们只输出 idx_map 中的用户，且用相同规则过滤证据。
        texts: List[str] = []
        source_stats: Counter = Counter()
        evidence: List[Dict[str, Any]] = []
        verified = False

        for rec in records:
            if not isinstance(rec, dict):
                continue
            raw = rec.get("raw") or {}
            if _is_verified_v(raw):
                verified = True
                break
            if rec.get("source_type") != "tweet":
                continue
            text = rec.get("text") or ""
            if _text_is_retweet(text):
                continue
            cleaned = clean_text(str(text))
            if not cleaned:
                continue
            texts.append(cleaned)
            source_stats[rec.get("source_type", "tweet")] += 1
            evidence.append(
                {
                    "created_at": rec.get("created_at"),
                    "source_type": rec.get("source_type"),
                    "text": cleaned,
                    "item_id": rec.get("item_id"),
                    "url": (raw.get("url") if isinstance(raw, dict) else None),
                }
            )

        if verified or not texts:
            continue

        evidence.sort(key=lambda r: _safe_int(r.get("item_id") or 0, 0), reverse=True)
        evidence = evidence[:20]

        i = idx_map[uid]
        cluster_id = int(cluster_ids[i])
        label = cluster_map.get(cluster_id, "其他")
        score = float(cos_scores[i])

        profiles.append(
            {
                "user_id": uid,
                "top_interest": label,
                "cluster_id": cluster_id,
                "interest_scores": {label: score},
                "source_stats": dict(source_stats),
                "sample_size": len(texts),
                "evidence": evidence,
            }
        )
        processed_outputs += 1
        if processed_outputs % 20 == 0:
            print(f"[INFER] profiles={processed_outputs}/{target_total}", flush=True)

    profiles.sort(key=lambda x: x["sample_size"], reverse=True)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wt", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    return profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer user interest profiles.")
    parser.add_argument("--input", required=True, help="Path to unified jsonl or weibo_crawl_*.json bundle.")
    parser.add_argument("--model-dir", default="output/ml_artifacts", help="Model artifact directory.")
    parser.add_argument("--output", default="output/user_interest_profiles.json", help="Output profile json file.")
    args = parser.parse_args()
    result = infer(args.input, args.model_dir, args.output)
    print(f"inferred_users={len(result)}")
