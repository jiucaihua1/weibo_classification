"""
step3_keyword_embeddings.py

对 step2_textrank_keywords.py 输出的：
  output/user_keywords_top10.jsonl
进行向量化（轻量中文 text2vec base）：

1) 解析 jsonl：每行一个用户
   {"user_id": "...", "keywords": ["kw1", ..., "kw10"]}

2) sentence-transformers 加载模型：
   shibing624/text2vec-base-chinese
   device='cuda'

3) 将每个用户的 10 个关键词分别 encode 得到 10 个词向量（embedding）
   再对这 10 个向量做 Mean Pooling，得到该用户一个特征向量

4) 保存：
   - output/user_features_text2vec.npy  (float32, shape=(n_users, dim))
   - output/user_ids_text2vec.pkl
   - output/step3_text2vec_meta.json（可选元信息）
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any, Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=str,
        default="output/user_keywords_top10.jsonl",
        help="input jsonl path",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="output",
        help="output directory",
    )
    ap.add_argument("--model-name", type=str, default="shibing624/text2vec-base-chinese")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch-size", type=int, default=64, help="encode batch_size (keywords per batch)")
    ap.add_argument("--top-k", type=int, default=10, help="expected keyword count per user")
    ap.add_argument("--require-topk", action="store_true", help="if true, skip users whose keywords != top-k")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(_project_root(), path))


def load_user_keywords_jsonl(path: str, *, top_k: int, require_topk: bool) -> Tuple[List[str], List[List[str]]]:
    user_ids: List[str] = []
    user_keywords: List[List[str]] = []

    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj: Dict[str, Any] = json.loads(line)
            uid = str(obj.get("user_id", "")).strip()
            kws = obj.get("keywords") or []
            if not uid:
                continue
            if not isinstance(kws, list):
                continue
            kws = [str(x).strip() for x in kws if str(x).strip()]
            if require_topk:
                if len(kws) != top_k:
                    continue
            else:
                if len(kws) < top_k:
                    continue
            kws = kws[:top_k]
            user_ids.append(uid)
            user_keywords.append(kws)
    return user_ids, user_keywords


def main() -> None:
    args = parse_args()
    input_path = _resolve_path(args.input)
    outdir = _resolve_path(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    out_features = os.path.join(outdir, "user_features_text2vec.npy")
    out_ids = os.path.join(outdir, "user_ids_text2vec.pkl")
    out_meta = os.path.join(outdir, "step3_text2vec_meta.json")

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"input not found: {input_path}")

    user_ids, user_keywords = load_user_keywords_jsonl(
        input_path,
        top_k=int(args.top_k),
        require_topk=bool(args.require_topk),
    )

    if not user_ids:
        raise RuntimeError("No users to encode. Try lowering filters.")

    flat_texts: List[str] = []
    for kws in user_keywords:
        flat_texts.extend(kws)

    # total = n_users * top_k
    n_users = len(user_ids)
    top_k = int(args.top_k)
    if len(flat_texts) != n_users * top_k:
        raise RuntimeError(f"unexpected flat_texts length={len(flat_texts)}, expected={n_users*top_k}")

    # Lazy import
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model_name, device=args.device)

    # encode 得到 embedding shape=(n_users*top_k, dim)
    emb = model.encode(
        flat_texts,
        batch_size=int(args.batch_size),
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype(np.float32, copy=False)

    if emb.ndim != 2:
        raise RuntimeError(f"unexpected embedding shape: {emb.shape}")

    dim = int(emb.shape[1])
    emb = emb.reshape(n_users, top_k, dim)
    # Mean pooling over the 10 keyword vectors
    user_features = emb.mean(axis=1).astype(np.float32, copy=False)  # (n_users, dim)

    np.save(out_features, user_features)
    with open(out_ids, "wb") as f:
        pickle.dump(user_ids, f)

    meta = {
        "input": input_path,
        "output_features": out_features,
        "output_user_ids": out_ids,
        "model_name": args.model_name,
        "device": args.device,
        "batch_size": int(args.batch_size),
        "n_users": n_users,
        "top_k": top_k,
        "embedding_dim": dim,
    }
    with open(out_meta, "wt", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] saved user features")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

