import argparse
import glob
import json
import os
import random
import re
import sys
from datetime import datetime
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import jieba
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize as l2_normalize
from sentence_transformers import SentenceTransformer

# Reuse the same TF-IDF + jieba keyword logic used elsewhere in this repo.
# When running this script directly, `sys.path[0]` points to `test_sandbox/`,
# so we must inject project root for `import app_pipeline...` to work.
_sandbox_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_sandbox_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# PowerShell / Windows terminal 有时不是 UTF-8，会导致控制台打印中文/emoji/奇怪 token 乱码。
# 我们尽量把 stdout 固定成 UTF-8，并且关键词也会落盘到 UTF-8 文件里给你直接查看。
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _is_retweet_text(text: str) -> bool:
    if not text:
        return False
    t = str(text).strip()
    return t == "转发微博" or t.startswith("转发微博")


def _iter_unified_jsonl_paths(input_glob: str) -> List[str]:
    paths = sorted(glob.glob(input_glob), key=os.path.getmtime, reverse=True)
    return paths


def _clean_tweet(text: str, *, min_chars: int) -> Optional[str]:
    """
    暴力白名单清洗：
    - 去 URL / @ / #话题#
    - 删除控制字符、零宽字符
    - 删除 emoji/符号区字符
    - 只保留中文、英文字母、数字和基础标点/空白；其它全部替换为空格
    - 最终压缩空白，并按 min_chars 过滤
    """
    t = str(text or "")
    if not t.strip():
        return None

    # remove zero-width / BOM / similar invisible chars
    t = re.sub(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]", "", t)

    # remove url / mentions / hashtags (non-greedy for #...#)
    t = re.sub(r"http[s]?://\S+|www\.\S+", " ", t)
    t = re.sub(r"@\S+", " ", t)
    t = re.sub(r"#.*?#", " ", t)

    # rough emoji range cleanup (kept as a best-effort)
    t = re.sub(
        "[\U0001F300-\U0001F6FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000027BF"
        "\uFE0F]+",
        " ",
        t,
    )

    # whitelist: Chinese / letters / digits / basic punctuation / whitespace
    # basic punct: ，。！？、；：“”‘’（）()【】[]{}、,.!?;:
    t = re.sub(
        r"[^"
        r"\u4e00-\u9fff"
        r"a-zA-Z0-9"
        r"，。！？、；：“”‘’（）()【】\[\]{}"
        r",.!?;:\s"
        r"]+",
        " ",
        t,
    )

    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return None
    if len(t) <= int(min_chars):
        return None
    return t


def _load_tweets_from_unified_jsonl(path: str, *, min_chars: int, max_tweets: Optional[int]) -> Tuple[List[str], List[int]]:
    """
    从 unified jsonl 抽取：
    - source_type == 'tweet'
    - 非转发
    - 清洗后 len > min_chars

    返回:
      texts: cleaned tweet texts
      row_spans: 记录每条样本在原始 jsonl 中的大致行号（用于调试；无则为0）
    """
    texts: List[str] = []
    row_spans: List[int] = []

    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_tweets is not None and len(texts) >= int(max_tweets):
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if str(obj.get("source_type", "")).strip() != "tweet":
                continue
            text = obj.get("text") or ""
            if _is_retweet_text(str(text)):
                continue

            cleaned = _clean_tweet(str(text), min_chars=min_chars)
            if not cleaned:
                continue

            texts.append(cleaned)
            row_spans.append(i)

    return texts, row_spans


def _load_tweet_samples_from_unified_jsonl(
    path: str,
    *,
    min_chars: int,
    max_tweets: Optional[int],
) -> List[dict]:
    """
    返回样本列表（与 embedding / labels 的顺序严格对齐）：
    - unified_file
    - line_no
    - user_id
    - item_id
    - text_raw
    - cleaned_text
    """
    samples: List[dict] = []
    with open(path, "rt", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_tweets is not None and len(samples) >= int(max_tweets):
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if str(obj.get("source_type", "")).strip() != "tweet":
                continue

            text_raw = obj.get("text") or ""
            if _is_retweet_text(str(text_raw)):
                continue

            cleaned = _clean_tweet(str(text_raw), min_chars=min_chars)
            if not cleaned:
                continue

            samples.append(
                {
                    "unified_file": os.path.normpath(path),
                    "line_no": int(i),
                    "user_id": str(obj.get("user_id", "")).strip(),
                    "item_id": str(obj.get("item_id", "")).strip(),
                    "text_raw": str(text_raw),
                    "cleaned_text": cleaned,
                }
            )
    return samples


def _encode_in_batches(model: SentenceTransformer, texts: Sequence[str], *, batch_size: int, device: str) -> np.ndarray:
    # SentenceTransformer already handles device, but keep parameter for API clarity.
    _ = device
    if not texts:
        raise ValueError("No texts to encode.")
    vectors: List[np.ndarray] = []
    n = len(texts)
    bs = max(1, int(batch_size))
    for start in range(0, n, bs):
        chunk = texts[start : start + bs]
        vec = model.encode(
            chunk,
            batch_size=bs,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        vectors.append(vec.astype(np.float32, copy=False))
        if (start // bs) % 10 == 0:
            print(f"[EMBED] {min(start + bs, n)}/{n}", flush=True)
    return np.concatenate(vectors, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug KMeans on per-tweet embeddings with PCA(64) + L2 normalization.")
    ap.add_argument(
        "--input-glob",
        default=os.path.join("output", "unified_*.jsonl"),
        help="Input unified jsonl glob (default: output/unified_*.jsonl).",
    )
    ap.add_argument("--latest-n", type=int, default=2, help="Take latest N unified_*.jsonl files.")
    ap.add_argument("--max-tweets", type=int, default=0, help="0=use all; otherwise limit tweets per run (after filtering).")
    ap.add_argument("--k", type=int, default=12, choices=[12], help="K for KMeans (fixed to 12).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--model-path", default=os.path.join("models", "text2vec-base-chinese"), help="Local model dir.")
    ap.add_argument("--embed-batch-size", type=int, default=64)
    ap.add_argument("--min-chars", type=int, default=5, help="Keep cleaned tweet text if len > min_chars.")
    ap.add_argument("--silhouette-max-samples", type=int, default=5000, help="Silhouette computed on a subsample if too large.")
    ap.add_argument("--samples-per-cluster", type=int, default=10, help="Randomly print N original tweets for each cluster.")
    ap.add_argument(
        "--convergence-runs",
        type=int,
        default=1,
        help="Rerun KMeans multiple times with different random_state to check label stability. 1=just one run.",
    )
    ap.add_argument(
        "--save-cleaned-jsonl",
        default=os.path.join("output", "kmeans_tweet_topic_debug_cleaned.jsonl"),
        help="Save filtered+cleaned tweets into a jsonl for manual inspection.",
    )
    ap.add_argument(
        "--save-with-cluster-jsonl",
        default=os.path.join("output", "kmeans_tweet_topic_debug_labeled.jsonl"),
        help="Save filtered tweets with cluster_id appended.",
    )
    ap.add_argument(
        "--no-save",
        action="store_true",
        help="Disable writing jsonl outputs.",
    )
    args = ap.parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    input_paths = _iter_unified_jsonl_paths(str(args.input_glob))[: max(1, int(args.latest_n))]
    if not input_paths:
        raise RuntimeError(f"No unified files matched: {args.input_glob}")

    print(f"[LOAD] unified files: {input_paths}", flush=True)
    max_tweets = None if int(args.max_tweets) <= 0 else int(args.max_tweets)

    all_samples: List[dict] = []
    for p in input_paths:
        per_file_max = None if max_tweets is None else max(0, int(max_tweets) - len(all_samples))
        if per_file_max is not None and per_file_max <= 0:
            break
        samples = _load_tweet_samples_from_unified_jsonl(
            p, min_chars=int(args.min_chars), max_tweets=per_file_max
        )
        all_samples.extend(samples)
        if max_tweets is not None and len(all_samples) >= int(max_tweets):
            all_samples = all_samples[: int(max_tweets)]
            break

    if not all_samples:
        raise RuntimeError("No tweets left after filtering/cleaning.")
    all_texts: List[str] = [s["cleaned_text"] for s in all_samples]
    print(
        f"[DATA] tweets={len(all_texts)} (source_type=tweet, non-retweet, cleaned_len>{args.min_chars})",
        flush=True,
    )

    if not bool(args.no_save):
        out_cleaned = os.path.normpath(str(args.save_cleaned_jsonl))
        os.makedirs(os.path.dirname(out_cleaned) or ".", exist_ok=True)
        with open(out_cleaned, "wt", encoding="utf-8") as f:
            for s in all_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"[SAVE] cleaned jsonl: {out_cleaned} (n={len(all_samples)})", flush=True)

    # Load local model.
    if not os.path.isdir(str(args.model_path)):
        raise RuntimeError(f"Model path not found: {args.model_path}")
    print(f"[MODEL] loading sentence-transformers from: {args.model_path} on device={args.device}", flush=True)
    model = SentenceTransformer(str(args.model_path), device=str(args.device))

    print("[STEP1] Encoding per-tweet vectors ...", flush=True)
    X = _encode_in_batches(
        model,
        all_texts,
        batch_size=int(args.embed_batch_size),
        device=str(args.device),
    )
    if X.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape: {X.shape}")
    if X.shape[1] != 768:
        print(f"[WARN] embedding dim != 768, got dim={X.shape[1]} (still continue).", flush=True)

    # Strict order:
    #   768-dim -> PCA(64) -> L2 normalize -> KMeans
    print("[STEP2] PCA(64) on embeddings (before L2 normalize) ...", flush=True)
    if X.shape[0] <= 64:
        raise RuntimeError(f"Need at least 65 samples for PCA(64), got n={X.shape[0]}")
    pca = PCA(n_components=64, random_state=int(args.seed))
    X64 = pca.fit_transform(X).astype(np.float32, copy=False)

    print("[STEP3] L2 normalize (required) ...", flush=True)
    Xn = l2_normalize(X64, norm="l2").astype(np.float32, copy=False)

    k = int(args.k)
    convergence_runs = max(1, int(args.convergence_runs))

    print(f"[STEP4] KMeans clustering (K={k}, runs={convergence_runs}) ...", flush=True)

    n = Xn.shape[0]
    idx = None
    if n > int(args.silhouette_max_samples):
        idx = np.random.choice(n, size=int(args.silhouette_max_samples), replace=False)
    X_for = Xn[idx] if idx is not None else Xn

    run_labels: List[np.ndarray] = []
    run_sils: List[float] = []
    run_inertias: List[float] = []

    for r in range(convergence_runs):
        rs = int(args.seed) + r
        kmeans = KMeans(n_clusters=k, random_state=rs, n_init=10)
        labels_r = kmeans.fit_predict(Xn).astype(int)
        run_labels.append(labels_r)

        labels_for = labels_r[idx] if idx is not None else labels_r
        sil_r = float(silhouette_score(X_for, labels_for, metric="cosine"))
        run_sils.append(sil_r)
        run_inertias.append(float(getattr(kmeans, "inertia_", float("nan"))))

        cluster_sizes_r = np.bincount(labels_r, minlength=k).astype(int).tolist()
        print(
            f"[RUN {r+1}/{convergence_runs}] seed={rs} silhouette={sil_r:.6f} inertia={run_inertias[-1]:.3f} sizes={cluster_sizes_r}",
            flush=True,
        )

    # Use first run as primary for downstream TF-IDF explanation + labeled outputs.
    labels = run_labels[0]
    sil = run_sils[0]
    print(f"[RESULT] Primary Silhouette Score: {sil:.6f}", flush=True)

    if convergence_runs > 1:
        print("[CHECK] Label stability across runs (ARI) ...", flush=True)
        base_labels = run_labels[0]
        for r in range(1, convergence_runs):
            ari = float(adjusted_rand_score(base_labels, run_labels[r]))
            print(f"  ARI(run1, run{r+1}) = {ari:.4f}", flush=True)
        print(f"[CHECK] silhouette range: {min(run_sils):.6f} ~ {max(run_sils):.6f}", flush=True)

    # Cluster sizes
    cluster_sizes = np.bincount(labels, minlength=int(args.k)).astype(int)
    print("[RESULT] Cluster sizes:", flush=True)
    for cid in range(int(args.k)):
        print(f"  cluster {cid}: n={int(cluster_sizes[cid])}", flush=True)

    # Print random tweet samples per cluster (original text, before whitelist cleaning).
    samples_per_cluster = max(1, int(args.samples_per_cluster))
    rng = random.Random(int(args.seed))
    print(f"[STEP5] Print {samples_per_cluster} raw tweets per cluster ...", flush=True)
    labels_list = labels.tolist()
    for cid in range(int(args.k)):
        idxs = [i for i, lab in enumerate(labels_list) if int(lab) == int(cid)]
        rng.shuffle(idxs)
        take = min(len(idxs), samples_per_cluster)
        print(f"\n===== cluster {cid} (n={int(cluster_sizes[cid])}) sample={take} =====", flush=True)
        for j in range(take):
            s = all_samples[idxs[j]]
            # Print only the tweet content for fast human inspection.
            text_raw = str(s.get("text_raw", "")).strip()
            if not text_raw:
                text_raw = str(s.get("cleaned_text", "")).strip()
            print(f"[{j+1}/{take}] {text_raw}\n", flush=True)

    if not bool(args.no_save):
        out_labeled = os.path.normpath(str(args.save_with_cluster_jsonl))
        os.makedirs(os.path.dirname(out_labeled) or ".", exist_ok=True)
        with open(out_labeled, "wt", encoding="utf-8") as f:
            for s, cid in zip(all_samples, labels.tolist()):
                s2 = dict(s)
                s2["cluster_id"] = int(cid)
                f.write(json.dumps(s2, ensure_ascii=False) + "\n")
        print(f"[SAVE] labeled jsonl: {out_labeled} (n={len(all_samples)})", flush=True)


if __name__ == "__main__":
    main()

