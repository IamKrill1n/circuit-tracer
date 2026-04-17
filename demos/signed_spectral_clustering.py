"""
Run spectral clustering variants on signed directed graph JSON files.

Implements two approaches:
1) Functional-profile similarity + standard Laplacian
2) Signed Laplacian directly on symmetrized signed weights
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


def _kmeans_numpy(x: np.ndarray, k: int, max_iter: int = 200, n_init: int = 8) -> tuple[np.ndarray, float]:
    """Simple NumPy k-means with multiple restarts."""
    n, d = x.shape
    if k <= 0 or k > n:
        raise ValueError(f"Invalid k={k} for n={n}")

    best_labels: np.ndarray | None = None
    best_inertia = float("inf")
    rng = np.random.default_rng(42)

    for _ in range(n_init):
        centers = x[rng.choice(n, size=k, replace=False)].copy()
        labels = np.zeros(n, dtype=np.int64)
        prev_labels = None

        for _it in range(max_iter):
            dist2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = np.argmin(dist2, axis=1)
            if prev_labels is not None and np.array_equal(prev_labels, labels):
                break
            prev_labels = labels.copy()
            for c in range(k):
                members = x[labels == c]
                if len(members) == 0:
                    centers[c] = x[rng.integers(0, n)]
                else:
                    centers[c] = members.mean(axis=0)

        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()

    assert best_labels is not None
    return best_labels, best_inertia


def _load_graph_json(graph_path: Path) -> tuple[np.ndarray, list[str]]:
    with graph_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("nodes", [])
    links = data.get("links", [])
    node_ids = [n["node_id"] for n in nodes]
    id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    n = len(node_ids)
    adj = np.zeros((n, n), dtype=np.float64)
    for link in links:
        src = link["source"]
        tgt = link["target"]
        weight = float(link.get("weight", 0.0))
        if src in id_to_index and tgt in id_to_index:
            src_idx = id_to_index[src]
            tgt_idx = id_to_index[tgt]
            # row=target, col=source
            adj[tgt_idx, src_idx] = weight
    return adj, node_ids


def _symmetrize_weights(adj_tgt_src: np.ndarray) -> np.ndarray:
    """Convert directed matrix (row=target,col=source) to symmetric signed matrix."""
    w = adj_tgt_src.astype(np.float64, copy=True)
    return 0.5 * (w + w.T)


def _functional_similarity_from_profiles(
    w_sym: np.ndarray,
    sigma: float,
    normalize_profiles: bool = False,
) -> np.ndarray:
    """
    Build similarity from augmented profile vectors:
    x_i = [W[i,:], W[:,i]]
    S_ij = exp(-||x_i-x_j||^2 / (2 sigma^2))
    """
    x = np.concatenate([w_sym, w_sym.T], axis=1)
    if normalize_profiles:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        x = x / norms

    sq = ((x[:, None, :] - x[None, :, :]) ** 2).sum(axis=2)
    s = np.exp(-sq / (2.0 * sigma * sigma))
    np.fill_diagonal(s, 1.0)
    return 0.5 * (s + s.T)


def _embedding_from_similarity_laplacian(s: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Standard unnormalized Laplacian embedding from non-negative S."""
    d = np.diag(np.sum(s, axis=1))
    l = d - s
    eigvals, eigvecs = np.linalg.eigh(l)
    idx = np.argsort(eigvals)[:k]
    emb = eigvecs[:, idx]
    return emb, eigvals[idx]


def _embedding_from_signed_laplacian(w_sym: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Signed Laplacian L = D_abs - W where D_abs_i = sum_j |w_ij|."""
    d_abs = np.diag(np.sum(np.abs(w_sym), axis=1))
    l_signed = d_abs - w_sym
    eigvals, eigvecs = np.linalg.eigh(l_signed)
    idx = np.argsort(eigvals)[:k]
    emb = eigvecs[:, idx]
    return emb, eigvals[idx]


def _cluster_counts(labels: np.ndarray) -> dict[str, int]:
    out: dict[str, int] = {}
    for c in np.unique(labels):
        out[str(int(c))] = int((labels == c).sum())
    return out


@dataclass
class RunResult:
    graph_file: str
    n_nodes: int
    method: str
    k: int
    sigma: float | None
    normalize_profiles: bool
    inertia: float
    selected_eigenvalues: list[float]
    cluster_sizes: dict[str, int]
    labels_by_node_id: dict[str, int]
    embedding_path: str


def run_setting(
    graph_path: Path,
    output_dir: Path,
    method: str,
    k: int,
    sigma: float | None,
    normalize_profiles: bool,
) -> RunResult:
    adj, node_ids = _load_graph_json(graph_path)
    w_sym = _symmetrize_weights(adj)
    n = int(w_sym.shape[0])
    if k >= n:
        raise ValueError(f"k={k} must be < number of nodes ({n})")

    if method == "functional_rbf":
        if sigma is None or sigma <= 0:
            raise ValueError("sigma must be positive for functional_rbf")
        s = _functional_similarity_from_profiles(w_sym, sigma=sigma, normalize_profiles=normalize_profiles)
        emb, eigvals = _embedding_from_similarity_laplacian(s, k=k)
    elif method == "signed_laplacian":
        emb, eigvals = _embedding_from_signed_laplacian(w_sym, k=k)
    else:
        raise ValueError(f"Unknown method: {method}")

    labels, inertia = _kmeans_numpy(emb, k=k)

    base_name = graph_path.stem
    sigma_tag = "none" if sigma is None else f"{sigma:g}"
    norm_tag = "norm1" if normalize_profiles else "norm0"
    run_id = f"{base_name}__{method}__k{k}__sigma{sigma_tag}__{norm_tag}"
    emb_path = output_dir / f"{run_id}__embedding.npy"
    np.save(emb_path, emb)

    label_map = {nid: int(labels[i]) for i, nid in enumerate(node_ids)}
    return RunResult(
        graph_file=str(graph_path),
        n_nodes=int(n),
        method=method,
        k=int(k),
        sigma=sigma,
        normalize_profiles=normalize_profiles,
        inertia=float(inertia),
        selected_eigenvalues=[float(v) for v in eigvals.tolist()],
        cluster_sizes=_cluster_counts(labels),
        labels_by_node_id=label_map,
        embedding_path=str(emb_path),
    )


def _write_outputs(results: list[RunResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "spectral_run_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    csv_path = output_dir / "spectral_run_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_file",
                "n_nodes",
                "method",
                "k",
                "sigma",
                "normalize_profiles",
                "inertia",
                "selected_eigenvalues",
                "cluster_sizes",
                "embedding_path",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "graph_file": r.graph_file,
                    "n_nodes": r.n_nodes,
                    "method": r.method,
                    "k": r.k,
                    "sigma": r.sigma,
                    "normalize_profiles": r.normalize_profiles,
                    "inertia": r.inertia,
                    "selected_eigenvalues": json.dumps(r.selected_eigenvalues),
                    "cluster_sizes": json.dumps(r.cluster_sizes),
                    "embedding_path": r.embedding_path,
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Signed/functional spectral clustering runner.")
    parser.add_argument(
        "--graph-files",
        nargs="+",
        required=True,
        help="Graph JSON files to process.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["functional_rbf", "signed_laplacian"],
        choices=["functional_rbf", "signed_laplacian"],
        help="Spectral methods to run.",
    )
    parser.add_argument("--k-values", nargs="+", type=int, default=[3, 5, 7], help="k values.")
    parser.add_argument(
        "--sigma-values",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0],
        help="RBF sigmas (used by functional_rbf).",
    )
    parser.add_argument(
        "--normalize-profiles",
        action="store_true",
        help="L2-normalize profile vectors before RBF distance.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="demos/spectral_outputs",
        help="Directory for all outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    for gf in args.graph_files:
        graph_path = Path(gf)
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")
        for method in args.methods:
            for k in args.k_values:
                if method == "functional_rbf":
                    for sigma in args.sigma_values:
                        res = run_setting(
                            graph_path=graph_path,
                            output_dir=output_dir,
                            method=method,
                            k=k,
                            sigma=sigma,
                            normalize_profiles=args.normalize_profiles,
                        )
                        results.append(res)
                        print(
                            f"[ok] {graph_path.name} {method} k={k} sigma={sigma} "
                            f"inertia={res.inertia:.4f}"
                        )
                else:
                    res = run_setting(
                        graph_path=graph_path,
                        output_dir=output_dir,
                        method=method,
                        k=k,
                        sigma=None,
                        normalize_profiles=False,
                    )
                    results.append(res)
                    print(f"[ok] {graph_path.name} {method} k={k} inertia={res.inertia:.4f}")

    _write_outputs(results, output_dir)
    print(f"\nSaved {len(results)} runs to {output_dir}")


if __name__ == "__main__":
    main()
