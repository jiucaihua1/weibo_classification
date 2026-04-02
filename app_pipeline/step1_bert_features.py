"""
app_pipeline/step1_bert_features.py

将「TextRank 关键词抽取 + BERT(text2vec) 向量化 + Mean Pooling + 持久化」合并为一个脚本：

输入：
  - weibo_crawl_latest.json（bundle：顶层包含 users[].records[]）

输出（默认写到 output/ 目录）：
  - user_features_text2vec.npy        float32 (n_users, 768)
  - user_ids_text2vec.pkl            list[str]，与 features 行严格一一对应
  - step3_text2vec_meta.json         元信息/参数

说明：
  - 具体清洗/过滤/关键词抽取/BERT 编码逻辑复用 app_pipeline/feature_bert_textrank.py 的实现；
    这里的作用是把它变成“一键可跑”的单文件脚本，避免你再单独跑 step2 + step3。
  - GPU：默认 device='cuda'；并默认把 HF 下载镜像指向 hf-mirror.com。
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Optional

import numpy as np

from app_pipeline.feature_bert_textrank import FeatureExtractionConfig, extract_user_features


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-file BERT feature extraction (TextRank -> text2vec -> mean pooling).")
    ap.add_argument(
        "--input",
        type=str,
        default="output/weibo_crawl_latest.json",
        help="Path to weibo_crawl_latest.json bundle.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="output",
        help="Output directory for artifacts.",
    )
    ap.add_argument("--top-k", type=int, default=10, help="TextRank topK keywords per user.")
    ap.add_argument("--min-chars", type=int, default=200, help="Drop users if cleaned long_text length < threshold.")
    ap.add_argument("--device", type=str, default="cuda", help="Device for sentence-transformers; requirement says use 'cuda'.")
    ap.add_argument("--model-name", type=str, default="shibing624/text2vec-base-chinese", help="SentenceTransformer model name.")
    ap.add_argument("--encode-batch-size", type=int, default=64, help="sentence-transformers encode batch_size (keywords per batch).")
    ap.add_argument("--user-batch-size", type=int, default=64, help="How many users per encode flush.")
    ap.add_argument(
        "--require-full-topk",
        action="store_true",
        help="If true, skip users whose keyword count < top_k (strict).",
    )
    ap.add_argument("--max-users", type=int, default=0, help="For debug: max users to process (0 means no limit).")
    ap.add_argument("--hf-endpoint", type=str, default="https://hf-mirror.com", help="HF endpoint mirror to speed up downloads.")
    args = ap.parse_args()

    bundle_path = _resolve_path(args.input)
    outdir = _resolve_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    if not os.path.isfile(bundle_path):
        raise FileNotFoundError(f"input not found: {bundle_path}")

    # Ensure HF downloads use mirror.
    if args.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", args.hf_endpoint)

    _check_cuda(args.device)

    cfg = FeatureExtractionConfig(
        top_k=int(args.top_k),
        min_chars=int(args.min_chars),
        encode_batch_size=int(args.encode_batch_size),
        device=str(args.device),
        model_name=str(args.model_name),
        user_batch_size=int(args.user_batch_size),
        require_full_topk=bool(args.require_full_topk),
    )

    max_users: Optional[int] = int(args.max_users) if int(args.max_users) > 0 else None

    print(f"[START] bundle={bundle_path}")
    user_ids, x = extract_user_features(bundle_path, cfg=cfg, max_users=max_users)

    # Ensure dtype/shape stable.
    x = x.astype(np.float32, copy=False)
    if x.ndim != 2:
        raise RuntimeError(f"unexpected features ndim: {x.shape}")
    if len(user_ids) != x.shape[0]:
        raise RuntimeError(f"user_ids len mismatch: {len(user_ids)} vs features rows {x.shape[0]}")

    features_path = os.path.join(outdir, "user_features_text2vec.npy")
    ids_path = os.path.join(outdir, "user_ids_text2vec.pkl")
    meta_path = os.path.join(outdir, "step3_text2vec_meta.json")

    np.save(features_path, x)
    with open(ids_path, "wb") as f:
        pickle.dump(user_ids, f)

    meta = {
        "input": bundle_path,
        "output_features": features_path,
        "output_user_ids": ids_path,
        "n_users": int(x.shape[0]),
        "top_k": int(args.top_k),
        "min_chars": int(args.min_chars),
        "device": str(args.device),
        "model_name": str(args.model_name),
        "encode_batch_size": int(args.encode_batch_size),
        "user_batch_size": int(args.user_batch_size),
        "require_full_topk": bool(args.require_full_topk),
        "embed_dim": int(x.shape[1]),
    }
    with open(meta_path, "wt", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] saved")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

