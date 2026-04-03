"""
app_pipeline/step1_bert_features.py

命令行工具：从 cleaned_user_texts.jsonl 生成用户级 TextRank + text2vec 向量（.npy / .pkl）。
Web「数据清洗」任务不再调用本模块；主聚类流水线为 train_tweet_topic（单条微博向量）。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import List, Optional

import numpy as np
import jieba.analyse


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_project_root(), path))


def _check_cuda(device: str) -> None:
    if device != "cuda":
        return
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot use device='cuda'.")
    except ImportError as e:
        raise RuntimeError("torch not installed; required for device='cuda'.") from e


def extract_user_features_from_cleaned_jsonl(
    cleaned_input_path: str,
    outdir: str,
    *,
    top_k: int = 10,
    device: str = "cuda",
    model_name: str = "shibing624/text2vec-base-chinese",
    encode_batch_size: int = 64,
    require_full_topk: bool = False,
    max_users: Optional[int] = None,
    hf_endpoint: str = "https://hf-mirror.com",
) -> dict:
    """
    从 cleaned_user_texts.jsonl 生成 user_features_text2vec.npy / user_ids_text2vec.pkl。

    重要：不做额外的数据清洗/过滤逻辑，只对 cleaned_text 做 TextRank。
    """
    cleaned_input_path = _resolve_path(cleaned_input_path)
    outdir = _resolve_path(outdir)
    os.makedirs(outdir, exist_ok=True)

    if not os.path.isfile(cleaned_input_path):
        raise FileNotFoundError(f"cleaned input not found: {cleaned_input_path}")

    # Ensure HF downloads use mirror. Force-set (not setdefault) so it affects runtime init.
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        os.environ["HUGGINGFACE_HUB_BASE_URL"] = hf_endpoint

    _check_cuda(device)

    user_ids: List[str] = []
    keyword_counts: List[int] = []
    flat_keywords: List[str] = []

    print(
        f"[START] cleaned_input={cleaned_input_path}, top_k={top_k}, max_users={max_users}",
        flush=True,
    )
    with open(cleaned_input_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = str(obj.get("user_id", "")).strip()
            cleaned_text = str(obj.get("cleaned_text", "") or "").strip()
            if not uid or not cleaned_text:
                continue

            kws = jieba.analyse.textrank(cleaned_text, topK=top_k, withWeight=False) or []
            kws = [str(x).strip() for x in kws if str(x).strip()]

            if require_full_topk:
                if len(kws) < top_k:
                    continue
                kws = kws[:top_k]
            else:
                kws = kws[:top_k]
                if len(kws) <= 0:
                    continue

            user_ids.append(uid)
            keyword_counts.append(len(kws))
            flat_keywords.extend(kws)

            if max_users is not None and len(user_ids) >= max_users:
                break

    if not user_ids:
        raise RuntimeError("No users found in cleaned_user_texts.jsonl after keyword extraction.")

    n_users = len(user_ids)
    total_kw = len(flat_keywords)
    print(
        f"[STEP A] users={n_users}, total_keywords={total_kw}, require_full_topk={bool(require_full_topk)}",
        flush=True,
    )

    # Re-import/instantiate after env variables are set.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_name), device=str(device))
    print(f"[STEP B] encoding keywords batch_size={int(encode_batch_size)}", flush=True)
    emb = model.encode(
        flat_keywords,
        batch_size=int(encode_batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32, copy=False)

    if emb.ndim != 2:
        raise RuntimeError(f"unexpected embedding shape: {emb.shape}")
    dim = int(emb.shape[1])
    if emb.shape[0] != total_kw:
        raise RuntimeError(f"embedding count mismatch: got={emb.shape[0]}, expected={total_kw}")

    user_features = np.zeros((n_users, dim), dtype=np.float32)
    start = 0
    for i, cnt in enumerate(keyword_counts):
        end = start + cnt
        user_features[i] = emb[start:end].mean(axis=0)
        start = end

    # debug mode: don't overwrite full-data outputs
    if max_users is None:
        features_path = os.path.join(outdir, "user_features_text2vec.npy")
        ids_path = os.path.join(outdir, "user_ids_text2vec.pkl")
        meta_path = os.path.join(outdir, "step3_text2vec_meta.json")
    else:
        features_path = os.path.join(outdir, f"user_features_text2vec_max{max_users}.npy")
        ids_path = os.path.join(outdir, f"user_ids_text2vec_max{max_users}.pkl")
        meta_path = os.path.join(outdir, f"step3_text2vec_meta_max{max_users}.json")

    np.save(features_path, user_features)
    with open(ids_path, "wb") as f:
        pickle.dump(user_ids, f)

    meta = {
        "cleaned_input": cleaned_input_path,
        "output_features": features_path,
        "output_user_ids": ids_path,
        "n_users": int(user_features.shape[0]),
        "top_k": int(top_k),
        "device": str(device),
        "model_name": str(model_name),
        "encode_batch_size": int(encode_batch_size),
        "require_full_topk": bool(require_full_topk),
        "embed_dim": int(user_features.shape[1]),
        "keyword_count_min": int(min(keyword_counts)) if keyword_counts else 0,
        "keyword_count_max": int(max(keyword_counts)) if keyword_counts else 0,
        "max_users": max_users,
    }
    with open(meta_path, "wt", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-file BERT feature extraction (TextRank -> text2vec -> mean pooling).")
    ap.add_argument(
        "--cleaned-input",
        type=str,
        default="output/cleaned_user_texts.jsonl",
        help="Path to cleaned_user_texts.jsonl (generated by clean_user_text.py).",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="output",
        help="Output directory for artifacts.",
    )
    ap.add_argument("--top-k", type=int, default=10, help="TextRank topK keywords per user.")
    ap.add_argument("--device", type=str, default="cuda", help="Device for sentence-transformers; requirement says use 'cuda'.")
    ap.add_argument("--model-name", type=str, default="shibing624/text2vec-base-chinese", help="SentenceTransformer model name.")
    ap.add_argument("--encode-batch-size", type=int, default=64, help="sentence-transformers encode batch_size (keywords per batch).")
    ap.add_argument(
        "--require-full-topk",
        action="store_true",
        help="If true, skip users whose keyword count < top_k (strict).",
    )
    ap.add_argument("--max-users", type=int, default=0, help="For debug: max users to process (0 means no limit).")
    ap.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com", help="HF endpoint mirror to speed up downloads.")
    args = ap.parse_args()

    max_users: Optional[int] = int(args.max_users) if int(args.max_users) > 0 else None
    meta = extract_user_features_from_cleaned_jsonl(
        args.cleaned_input,
        args.outdir,
        top_k=int(args.top_k),
        device=str(args.device),
        model_name=str(args.model_name),
        encode_batch_size=int(args.encode_batch_size),
        require_full_topk=bool(args.require_full_topk),
        max_users=max_users,
        hf_endpoint=str(args.hf_endpoint) if args.hf_endpoint else "https://hf-mirror.com",
    )
    print("[OK] saved")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return

    cleaned_input_path = _resolve_path(args.cleaned_input)
    outdir = _resolve_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    if not os.path.isfile(cleaned_input_path):
        raise FileNotFoundError(f"cleaned input not found: {cleaned_input_path}")

    # Ensure HF downloads use mirror.
    # Note: some components read the endpoint env var at import / client-init time,
    # so we force-set (not setdefault) and also provide HUGGINGFACE_HUB_BASE_URL.
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
        os.environ["HUGGINGFACE_HUB_BASE_URL"] = args.hf_endpoint

    _check_cuda(args.device)

    max_users: Optional[int] = int(args.max_users) if int(args.max_users) > 0 else None
    top_k = int(args.top_k)

    user_ids: List[str] = []
    keyword_counts: List[int] = []
    flat_keywords: List[str] = []

    print(f"[START] cleaned_input={cleaned_input_path}, top_k={top_k}")
    with open(cleaned_input_path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = str(obj.get("user_id", "")).strip()
            cleaned_text = str(obj.get("cleaned_text", "") or "").strip()
            if not uid or not cleaned_text:
                continue

            kws = jieba.analyse.textrank(cleaned_text, topK=top_k, withWeight=False) or []
            kws = [str(x).strip() for x in kws if str(x).strip()]

            if args.require_full_topk:
                if len(kws) < top_k:
                    continue
                kws = kws[:top_k]
            else:
                # 允许少于 top_k：但仍然只取前 top_k，之后对实际数量做 mean pooling
                kws = kws[:top_k]
                if len(kws) <= 0:
                    continue

            user_ids.append(uid)
            keyword_counts.append(len(kws))
            flat_keywords.extend(kws)

            if max_users is not None and len(user_ids) >= max_users:
                break

    if not user_ids:
        raise RuntimeError("No users found in cleaned_user_texts.jsonl after keyword extraction.")

    n_users = len(user_ids)
    total_kw = len(flat_keywords)
    print(f"[STEP A] users={n_users}, total_keywords={total_kw}, require_full_topk={bool(args.require_full_topk)}")

    # Re-create model after env variables are set.
    # Re-import after env variables are set to ensure HF endpoint takes effect.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(args.model_name), device=str(args.device))
    print(f"[STEP B] encoding keywords batch_size={int(args.encode_batch_size)}")
    emb = model.encode(
        flat_keywords,
        batch_size=int(args.encode_batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32, copy=False)

    if emb.ndim != 2:
        raise RuntimeError(f"unexpected embedding shape: {emb.shape}")
    dim = int(emb.shape[1])
    if emb.shape[0] != total_kw:
        raise RuntimeError(f"embedding count mismatch: got={emb.shape[0]}, expected={total_kw}")

    # mean pooling per user
    user_features = np.zeros((n_users, dim), dtype=np.float32)
    start = 0
    for i, cnt in enumerate(keyword_counts):
        end = start + cnt
        user_features[i] = emb[start:end].mean(axis=0)
        start = end

    x = user_features

    # When doing debug with --max-users, do NOT overwrite the default full-data outputs.
    if max_users is None:
        features_path = os.path.join(outdir, "user_features_text2vec.npy")
        ids_path = os.path.join(outdir, "user_ids_text2vec.pkl")
        meta_path = os.path.join(outdir, "step3_text2vec_meta.json")
    else:
        features_path = os.path.join(outdir, f"user_features_text2vec_max{max_users}.npy")
        ids_path = os.path.join(outdir, f"user_ids_text2vec_max{max_users}.pkl")
        meta_path = os.path.join(outdir, f"step3_text2vec_meta_max{max_users}.json")

    np.save(features_path, x)
    with open(ids_path, "wb") as f:
        pickle.dump(user_ids, f)

    meta = {
        "cleaned_input": cleaned_input_path,
        "output_features": features_path,
        "output_user_ids": ids_path,
        "n_users": int(x.shape[0]),
        "top_k": int(args.top_k),
        "device": str(args.device),
        "model_name": str(args.model_name),
        "encode_batch_size": int(args.encode_batch_size),
        "require_full_topk": bool(args.require_full_topk),
        "embed_dim": int(x.shape[1]),
        "keyword_count_min": int(min(keyword_counts)) if keyword_counts else 0,
        "keyword_count_max": int(max(keyword_counts)) if keyword_counts else 0,
        "max_users": max_users,
    }
    with open(meta_path, "wt", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] saved")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

