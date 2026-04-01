import argparse
import json
import os
import pickle
from collections import Counter, defaultdict
from typing import Dict, List

from app_pipeline.data_io import load_infer_records
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


def infer(unified_path: str, model_dir: str, output_file: str) -> List[Dict]:
    with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    with open(os.path.join(model_dir, "kmeans_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(model_dir, "cluster_label_map.json"), "rt", encoding="utf-8") as f:
        cluster_map = {int(k): v for k, v in json.load(f).items()}

    records = load_infer_records(unified_path)
    by_user_text = defaultdict(list)
    by_user_source = defaultdict(Counter)
    for row in records:
        text = clean_text(row.get("text", ""))
        if text:
            by_user_text[row["user_id"]].append(text)
        by_user_source[row["user_id"]][row.get("source_type", "unknown")] += 1

    profiles = []
    for user_id, texts in by_user_text.items():
        doc = " ".join(texts)
        x = vectorizer.transform([doc])
        cluster_id = int(model.predict(x)[0])
        label = cluster_map.get(cluster_id, "其他")
        center = model.cluster_centers_[cluster_id]
        score = float(x.multiply(center).sum())
        evidence = _extract_evidence_rows(records, user_id, limit=20)
        profiles.append(
            {
                "user_id": user_id,
                "top_interest": label,
                "cluster_id": cluster_id,
                "interest_scores": {label: score},
                "source_stats": dict(by_user_source[user_id]),
                "sample_size": len(texts),
                "evidence": evidence,
            }
        )

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
