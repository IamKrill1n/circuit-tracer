from __future__ import annotations

from typing import Any

import numpy as np

from summarization.supernode_graph import SummarizationGraph


def _unwrap_sng(
    sng: SummarizationGraph | dict[str, Any],
) -> tuple[list[str], np.ndarray, dict]:
    if isinstance(sng, SummarizationGraph):
        return sng.sn_names, sng.sn_adj, sng.node_by_name()
    return list(sng["sn_names"]), np.asarray(sng["sn_adj"], dtype=np.float64), {}


def shortcut_analysis(
    sng: SummarizationGraph | dict[str, Any],
    final_supernodes: dict[str, list[str]] | list[list[str]] | None = None,
    min_edge_weight: float = 1e-6,
) -> dict[str, Any]:
    del final_supernodes
    sn_names, sn_adj, _ = _unwrap_sng(sng)
    k = len(sn_names)
    edges = []
    tot = 0.0
    shortcut_tot = 0.0

    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            w = float(sn_adj[i, j])
            if w < min_edge_weight:
                continue
            best_med = 0.0
            med_name = None
            for b in range(k):
                if b in (i, j):
                    continue
                mediation = min(float(sn_adj[i, b]), float(sn_adj[b, j]))
                if mediation > best_med:
                    best_med = mediation
                    med_name = sn_names[b]
            ratio = w / (w + best_med + 1e-12)
            is_shortcut = ratio < 0.5
            tot += w
            if is_shortcut:
                shortcut_tot += w
            edges.append(
                {
                    "src": sn_names[i],
                    "tgt": sn_names[j],
                    "weight": w,
                    "shortcut_ratio": ratio,
                    "is_shortcut": is_shortcut,
                    "best_mediator": med_name,
                    "mediation_strength": best_med,
                }
            )
    edges.sort(key=lambda x: -x["weight"])
    return {
        "edges": edges,
        "n_shortcuts": sum(1 for e in edges if e["is_shortcut"]),
        "n_direct": sum(1 for e in edges if not e["is_shortcut"]),
        "n_total": len(edges),
        "global_shortcut_frac": float(shortcut_tot / (tot + 1e-12)),
    }
