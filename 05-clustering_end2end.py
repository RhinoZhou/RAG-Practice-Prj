"""
聚类全流程实验（Embedding → UMAP → 聚类 → 可视化 → 评估）

作者: Ph.D. Rhino
输入：JSONL(每行对象含 text/title/content) 或 JSON 数组字符串
输出：
- outputs/emb_xy.npz
- outputs/clusters.png
- outputs/cluster_labels.npz
- outputs/cluster_summaries.json
- outputs/cluster_eval.json
- 额外：kmeans_curve.json/png、labels_with_text.csv、kmeans_cluster_sizes.csv、hdbscan_cluster_sizes.csv、keywords_preview.txt、(可选)hdbscan_scores.npz/hdbscan_outliers.npz
"""

from __future__ import annotations

import json
import os
import sys
import math
import argparse
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


def ensure_packages_installed() -> None:
    pkgs = [
        "numpy",
        "pandas",
        "scikit-learn",
        "sentence-transformers",
        "umap-learn",
        "matplotlib",
        "seaborn",
    ]
    def ok(p: str) -> bool:
        try:
            __import__(p.split("[")[0].replace("-", "_"))
            return True
        except Exception:
            return False
    miss = [p for p in pkgs if not ok(p)]
    if miss:
        print(f"[setup] install: {miss}")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", *miss], check=True)


ensure_packages_installed()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    pairwise_distances_argmin_min,
)
import umap
try:
    import hdbscan  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False
from sentence_transformers import SentenceTransformer


def read_texts(path: str) -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            try:
                arr = json.load(f)
                for it in arr:
                    if isinstance(it, str) and it.strip():
                        texts.append(it.strip())
            except Exception:
                pass
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for k in ("text", "title", "content"):
                    if k in obj and isinstance(obj[k], str) and obj[k].strip():
                        texts.append(obj[k].strip())
                        break
    return texts


def batch_embed(texts: List[str], model_name: str, batch_size: int) -> np.ndarray:
    model = SentenceTransformer(model_name)
    out: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        arr = model.encode(texts[i:i+batch_size], show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        out.append(arr)
    return np.vstack(out)


def reduce_umap(emb: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, random_state: int = 42) -> np.ndarray:
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    return reducer.fit_transform(emb)


@dataclass
class ClusterResult:
    method: str
    labels: np.ndarray
    n_clusters: int
    noise_ratio: float
    metrics: Dict[str, Optional[float]]


def evaluate(emb: np.ndarray, labels: np.ndarray) -> Dict[str, Optional[float]]:
    m = labels >= 0
    if m.sum() < 3 or len(np.unique(labels[m])) < 2:
        return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
    def safe(fn):
        try:
            return float(fn(emb[m], labels[m]))
        except Exception:
            return None
    return {
        "silhouette": safe(lambda X, y: silhouette_score(X, y, metric="cosine")),
        "davies_bouldin": safe(davies_bouldin_score),
        "calinski_harabasz": safe(calinski_harabasz_score),
    }


def kmeans_auto_k(emb: np.ndarray, k_min: int, k_max: int, random_state: int = 42) -> Tuple[ClusterResult, List[Dict[str, float]]]:
    curve: List[Dict[str, float]] = []
    best = (-1.0, None, None)
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(emb)
        try:
            sil = float(silhouette_score(emb, labels, metric="cosine"))
        except Exception:
            sil = float("nan")
        try:
            db = float(davies_bouldin_score(emb, labels))
        except Exception:
            db = float("nan")
        try:
            ch = float(calinski_harabasz_score(emb, labels))
        except Exception:
            ch = float("nan")
        curve.append({"k": float(k), "silhouette": sil, "davies_bouldin": db, "calinski_harabasz": ch})
        if (np.isnan(sil) and best[0] < 0) or (not np.isnan(sil) and sil > best[0]):
            best = (sil if not np.isnan(sil) else -1.0, k, labels)
    assert best[2] is not None
    labels = best[2]
    return ClusterResult("kmeans", labels, int(len(np.unique(labels))), 0.0, evaluate(emb, labels)), curve


def run_hdbscan_or_dbscan(emb: np.ndarray, min_cluster_size: int) -> ClusterResult:
    if HAS_HDBSCAN:
        cl = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = cl.fit_predict(emb)
        method = "hdbscan"
    else:
        cl = DBSCAN(eps=0.8, min_samples=max(5, min_cluster_size), metric="euclidean")
        labels = cl.fit_predict(emb)
        method = "dbscan"
    n_clusters = int(len(np.unique(labels[labels >= 0])))
    noise_ratio = float(np.mean(labels < 0))
    return ClusterResult(method, labels, n_clusters, noise_ratio, evaluate(emb, labels))


def tfidf_keywords(texts: List[str], labels: np.ndarray, top_k: int = 8) -> Dict[int, List[str]]:
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    tfidf = vec.fit_transform(texts)
    names = np.array(vec.get_feature_names_out())
    out: Dict[int, List[str]] = {}
    for c in sorted(np.unique(labels)):
        if c < 0:
            continue
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            out[int(c)] = []
            continue
        mean_scores = np.asarray(tfidf[idx].mean(axis=0)).ravel()
        top = np.argsort(mean_scores)[-top_k:][::-1]
        out[int(c)] = names[top].tolist()
    return out


def representatives(emb: np.ndarray, labels: np.ndarray, texts: List[str]) -> Dict[int, Dict[str, object]]:
    reps: Dict[int, Dict[str, object]] = {}
    for c in sorted(np.unique(labels)):
        if c < 0:
            continue
        idx = np.where(labels == c)[0]
        if idx.size == 0:
            continue
        centroid = emb[idx].mean(axis=0, keepdims=True)
        closest, dist = pairwise_distances_argmin_min(centroid, emb[idx], metric="euclidean")
        gid = int(idx[int(closest[0])])
        reps[int(c)] = {"index": gid, "text": texts[gid], "distance": float(dist[0])}
    return reps


def boundary_indices(emb: np.ndarray, labels: np.ndarray) -> List[int]:
    from sklearn.metrics import silhouette_samples
    m = labels >= 0
    if m.sum() < 3 or len(np.unique(labels[m])) < 2:
        return []
    try:
        s = silhouette_samples(emb[m], labels[m], metric="cosine")
    except Exception:
        return []
    return [int(np.where(m)[0][i]) for i in np.where(s <= 0.0)[0]]


def plot(xy: np.ndarray, labels_k: np.ndarray, labels_h: np.ndarray, out_path: str) -> None:
    plt.figure(figsize=(14, 6))
    cmap = "tab20"
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=labels_k, palette=cmap, s=12, linewidth=0)
    plt.title("KMeans (auto-k)")
    plt.legend([], [], frameon=False)
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=xy[:, 0], y=xy[:, 1], hue=labels_h, palette=cmap, s=12, linewidth=0)
    plt.title("HDBSCAN/DBSCAN")
    plt.legend([], [], frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Clustering end-to-end pipeline")
    p.add_argument("--input", type=str, default=os.path.join("data", "titles_clean.jsonl"))
    p.add_argument("--outdir", type=str, default="outputs")
    p.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=20)
    p.add_argument("--min_cluster_size", type=int, default=5)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"[load] reading: {args.input}")
    texts = read_texts(args.input)
    if not texts:
        raise RuntimeError("No valid texts found in input.")
    print(f"[load] total texts: {len(texts)}")

    print(f"[embed] model: {args.model}")
    emb = batch_embed(texts, args.model, args.batch_size)
    print(f"[embed] embeddings: {emb.shape}")

    print("[umap] reducing to 2D ...")
    xy = reduce_umap(emb)
    np.savez_compressed(os.path.join(args.outdir, "emb_xy.npz"), x=xy[:, 0], y=xy[:, 1])

    print("[cluster] KMeans auto-k ...")
    ck, k_curve = kmeans_auto_k(emb, args.k_min, args.k_max)
    print(f"[cluster] KMeans -> k={ck.n_clusters}, metrics={ck.metrics}")
    with open(os.path.join(args.outdir, "kmeans_curve.json"), "w", encoding="utf-8") as f:
        json.dump(k_curve, f, ensure_ascii=False, indent=2)
    try:
        dfc = pd.DataFrame(k_curve)
        plt.figure(figsize=(8, 4))
        plt.plot(dfc["k"], dfc["silhouette"], marker="o", label="silhouette")
        plt.twinx()
        plt.plot(dfc["k"], dfc["davies_bouldin"], color="orange", marker="x", label="db-index")
        plt.title("KMeans metric curve")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "kmeans_curve.png"), dpi=200)
        plt.close()
    except Exception:
        pass

    print("[cluster] HDBSCAN/DBSCAN ...")
    ch = run_hdbscan_or_dbscan(emb, args.min_cluster_size)
    print(f"[cluster] {ch.method.upper()} -> k={ch.n_clusters}, noise={ch.noise_ratio:.2f}, metrics={ch.metrics}")

    plot_path = os.path.join(args.outdir, "clusters.png")
    plot(xy, ck.labels, ch.labels, plot_path)

    np.savez_compressed(os.path.join(args.outdir, "cluster_labels.npz"), kmeans_labels=ck.labels, hdbscan_labels=ch.labels)

    # 标签明细 & 簇大小
    df_labels = pd.DataFrame({"index": np.arange(len(texts)), "text": texts, "kmeans": ck.labels, "hdbscan": ch.labels})
    df_labels.to_csv(os.path.join(args.outdir, "labels_with_text.csv"), index=False, encoding="utf-8-sig")
    def size_s(labels: np.ndarray) -> pd.Series:
        s = pd.Series(labels)
        s = s[s >= 0]
        return s.value_counts().sort_index()
    pd.DataFrame({"cluster": size_s(ck.labels).index, "size": size_s(ck.labels).values}).to_csv(os.path.join(args.outdir, "kmeans_cluster_sizes.csv"), index=False)
    pd.DataFrame({"cluster": size_s(ch.labels).index, "size": size_s(ch.labels).values}).to_csv(os.path.join(args.outdir, "hdbscan_cluster_sizes.csv"), index=False)

    eval_json = {"kmeans": {"n_clusters": ck.n_clusters, "noise_ratio": ck.noise_ratio, **ck.metrics}, "hdbscan": {"n_clusters": ch.n_clusters, "noise_ratio": ch.noise_ratio, **ch.metrics}}
    with open(os.path.join(args.outdir, "cluster_eval.json"), "w", encoding="utf-8") as f:
        json.dump(eval_json, f, ensure_ascii=False, indent=2)

    print("[summary] keywords / representatives / boundary ...")
    sums: Dict[str, Dict[str, object]] = {}
    for name, labels in (("kmeans", ck.labels), ("hdbscan", ch.labels)):
        sums[name] = {
            "keywords": {str(k): v for k, v in tfidf_keywords(texts, labels, top_k=8).items()},
            "representatives": {str(k): v for k, v in representatives(emb, labels, texts).items()},
            "boundary_indices": boundary_indices(emb, labels),
        }
    with open(os.path.join(args.outdir, "cluster_summaries.json"), "w", encoding="utf-8") as f:
        json.dump(sums, f, ensure_ascii=False, indent=2)

    # 关键词预览
    try:
        lines: List[str] = []
        for name, labels in (("kmeans", ck.labels), ("hdbscan", ch.labels)):
            cnt = pd.Series(labels)
            cnt = cnt[cnt >= 0].value_counts().sort_values(ascending=False)
            kw = tfidf_keywords(texts, labels, top_k=6)
            lines.append(f"== {name} ==")
            for cid in cnt.index[: min(8, len(cnt))]:
                lines.append(f"cluster {cid} (n={int(cnt[cid])}): {', '.join(kw.get(int(cid), []))}")
        with open(os.path.join(args.outdir, "keywords_preview.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        pass

    print("[done] Artifacts saved in outputs/ .")


if __name__ == "__main__":
    main()


