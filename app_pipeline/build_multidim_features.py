import argparse
import glob
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"


def _latest_unified_files(limit: int) -> list[Path]:
    files = sorted(
        [Path(p) for p in glob.glob(str(OUTPUT_DIR / "unified_*.jsonl"))],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return list(reversed(files[: max(1, limit)]))


def _safe_bool(value) -> int:
    return 1 if bool(value) else 0


def _safe_num(value) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def build_features(unified_paths: list[Path], out_dir: Path, embed_model_name: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # user_id -> profile fields (from user spider first, fallback tweet user snippets)
    profiles: dict[str, dict] = {}
    # user_id -> authored tweet texts
    text_pool: dict[str, list[str]] = defaultdict(list)
    # user_id -> interaction counters
    interactions: dict[str, dict] = defaultdict(
        lambda: {
            "tweets_count": 0,
            "retweet_posts_count": 0,
            "sum_reposts_count": 0.0,
            "sum_comments_count": 0.0,
            "sum_attitudes_count": 0.0,
            "sum_video_online_numbers": 0.0,
            "sum_pic_num": 0.0,
            "comment_rows_count": 0,
        }
    )

    for path in unified_paths:
        with path.open("rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                uid = str(row.get("user_id", "")).strip()
                if not uid:
                    continue
                raw = row.get("raw") or {}
                source_type = str(row.get("source_type", "")).strip().lower()

                # 1) 基本面用户画像：优先 user spider，次选 tweet 的 user 子对象
                if source_type == "user":
                    profiles[uid] = {
                        "user_id": uid,
                        "followers_count": _safe_num(raw.get("followers_count")),
                        "friends_count": _safe_num(raw.get("friends_count")),
                        "statuses_count": _safe_num(raw.get("statuses_count")),
                        "verified": _safe_bool(raw.get("verified")),
                        "verified_reason": str(raw.get("verified_reason", "") or ""),
                        "description": str(raw.get("description", "") or ""),
                        "gender": str(raw.get("gender", "") or ""),
                        "location": str(raw.get("location", "") or ""),
                        "mbrank": _safe_num(raw.get("mbrank")),
                        "mbtype": _safe_num(raw.get("mbtype")),
                    }
                elif uid not in profiles:
                    user = raw.get("user") or {}
                    if isinstance(user, dict):
                        profiles[uid] = {
                            "user_id": uid,
                            "followers_count": _safe_num(user.get("followers_count")),
                            "friends_count": _safe_num(user.get("friends_count")),
                            "statuses_count": _safe_num(user.get("statuses_count")),
                            "verified": _safe_bool(user.get("verified")),
                            "verified_reason": str(user.get("verified_reason", "") or ""),
                            "description": str(user.get("description", "") or ""),
                            "gender": str(user.get("gender", "") or ""),
                            "location": str(user.get("location", "") or ""),
                            "mbrank": _safe_num(user.get("mbrank")),
                            "mbtype": _safe_num(user.get("mbtype")),
                        }

                # 2) 文本语义数据：用户历史推文文本池
                if source_type == "tweet":
                    text = str(raw.get("content", "") or row.get("text", "") or "").strip()
                    if text:
                        text_pool[uid].append(text)

                    # 3) 互动数据：转发/评论/点赞等计数
                    stats = interactions[uid]
                    stats["tweets_count"] += 1
                    stats["retweet_posts_count"] += _safe_bool(raw.get("is_retweet"))
                    stats["sum_reposts_count"] += _safe_num(raw.get("reposts_count"))
                    stats["sum_comments_count"] += _safe_num(raw.get("comments_count"))
                    stats["sum_attitudes_count"] += _safe_num(raw.get("attitudes_count"))
                    stats["sum_video_online_numbers"] += _safe_num(raw.get("video_online_numbers"))
                    stats["sum_pic_num"] += _safe_num(raw.get("pic_num"))
                elif source_type == "comment":
                    interactions[uid]["comment_rows_count"] += 1

    # Write fundamental features
    fundamental_path = out_dir / "fundamental_features.jsonl"
    with fundamental_path.open("wt", encoding="utf-8") as f:
        for uid in sorted(profiles.keys()):
            f.write(json.dumps(profiles[uid], ensure_ascii=False) + "\n")

    # Write text corpus (clean minimal noise + keep raw concat)
    text_corpus_path = out_dir / "text_corpus.jsonl"
    with text_corpus_path.open("wt", encoding="utf-8") as f:
        for uid in sorted(text_pool.keys()):
            texts = text_pool[uid]
            text_concat = " ".join(t.strip() for t in texts if t.strip())
            f.write(
                json.dumps(
                    {
                        "user_id": uid,
                        "tweet_count": len(texts),
                        "text_concat": text_concat,
                        "texts": texts[:200],  # cap to avoid huge lines
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # Optional embeddings for text semantics
    text_emb_npy_path = ""
    text_emb_uid_path = ""
    if embed_model_name:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not available; cannot build embeddings.")
        model = SentenceTransformer(embed_model_name)
        ordered_uids = sorted(text_pool.keys())
        docs = [" ".join(text_pool[uid]) for uid in ordered_uids]
        vectors = model.encode(docs, show_progress_bar=True, normalize_embeddings=True)
        vec_arr = np.asarray(vectors, dtype=np.float32)
        text_emb_npy_path = str(out_dir / "text_semantic_embeddings.npy")
        text_emb_uid_path = str(out_dir / "text_semantic_user_ids.json")
        np.save(text_emb_npy_path, vec_arr)
        with open(text_emb_uid_path, "wt", encoding="utf-8") as f:
            json.dump(ordered_uids, f, ensure_ascii=False, indent=2)

    # Write interaction features
    interaction_path = out_dir / "interaction_features.jsonl"
    with interaction_path.open("wt", encoding="utf-8") as f:
        for uid in sorted(interactions.keys()):
            stat = interactions[uid]
            tweets = max(1, int(stat["tweets_count"]))
            row = {
                "user_id": uid,
                "tweets_count": int(stat["tweets_count"]),
                "retweet_posts_count": int(stat["retweet_posts_count"]),
                "avg_reposts_count": stat["sum_reposts_count"] / tweets,
                "avg_comments_count": stat["sum_comments_count"] / tweets,
                "avg_attitudes_count": stat["sum_attitudes_count"] / tweets,
                "avg_video_online_numbers": stat["sum_video_online_numbers"] / tweets,
                "avg_pic_num": stat["sum_pic_num"] / tweets,
                "comment_rows_count": int(stat["comment_rows_count"]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_unified_files": [str(p) for p in unified_paths],
        "n_users_profile": len(profiles),
        "n_users_text": len(text_pool),
        "n_users_interaction": len(interactions),
        "fundamental_features": str(fundamental_path),
        "text_corpus": str(text_corpus_path),
        "interaction_features": str(interaction_path),
        "text_semantic_embeddings": text_emb_npy_path,
        "text_semantic_user_ids": text_emb_uid_path,
    }
    meta_path = out_dir / "feature_build_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    meta["meta_file"] = str(meta_path)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build 3-way training features from Weibo unified crawl files."
    )
    parser.add_argument(
        "--latest-unified-count",
        type=int,
        default=3,
        help="Use latest N unified_*.jsonl files as input.",
    )
    parser.add_argument(
        "--input-unified",
        nargs="*",
        default=[],
        help="Explicit unified files; if set, overrides --latest-unified-count",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR / "features"),
        help="Output directory for feature files.",
    )
    parser.add_argument(
        "--embed-model",
        default="",
        help="Optional sentence-transformers model name, e.g. shibing624/text2vec-base-chinese",
    )
    args = parser.parse_args()

    if args.input_unified:
        unified_paths = [Path(p) for p in args.input_unified if Path(p).is_file()]
    else:
        unified_paths = _latest_unified_files(args.latest_unified_count)
    if not unified_paths:
        print("[ERROR] No unified input files found.")
        return 2

    out_dir = Path(args.output_dir)
    meta = build_features(unified_paths, out_dir, args.embed_model.strip())
    print("[DONE] Built multi-dimensional feature dataset.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
