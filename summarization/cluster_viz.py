"""Plotly visualization helpers for clustered supernode graphs."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from summarization.utils import _parse_layer
from summarization.flow_analysis import _classify_sn


def _sn_kind(sn_name: str) -> str:
    return _classify_sn(sn_name)


def _sn_title(sn: str, members: list[str], attr: dict[str, dict[str, Any]] | None) -> str:
    if not members:
        return sn
    if attr is None:
        return f"{sn} ({len(members)} nodes)"
    previews: list[str] = []
    for nid in members[:5]:
        clerp = str(attr.get(nid, {}).get("clerp", "") or "").strip()
        if clerp:
            previews.append(f"{nid}: {clerp[:80]}")
        else:
            previews.append(str(nid))
    more = f" … +{len(members) - 5} more" if len(members) > 5 else ""
    return sn + "<br>" + "<br>".join(previews) + more


def _parse_ctx_idx(attr: dict[str, dict[str, Any]] | None, node_id: str) -> int:
    if attr is None:
        return 0
    raw = attr.get(node_id, {}).get("ctx_idx", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _layer_and_ctx_for_supernode(
    sn: str,
    members: list[str],
    attr: dict[str, dict[str, Any]] | None,
) -> tuple[int, float]:
    items = members if members else [sn]
    layers = [_parse_layer(attr or {}, nid) for nid in items]
    ctx_idx = [_parse_ctx_idx(attr, nid) for nid in items]
    return (min(layers) if layers else 0, float(np.mean(ctx_idx) if ctx_idx else 0.0))


def _sn_label(sn: str, members: list[str], attr: dict[str, dict[str, Any]] | None) -> str:
    if attr is None:
        return sn
    for nid in members:
        clerp = str(attr.get(nid, {}).get("clerp", "") or "").strip()
        if clerp:
            return clerp[:70] + ("..." if len(clerp) > 70 else "")
    return sn


def _layered_layout(
    sn_names: list[str],
    final_supernodes: dict[str, list[str]],
    attr: dict[str, dict[str, Any]] | None,
) -> dict[str, tuple[float, float]]:
    if not sn_names:
        return {}
    grouped: dict[int, list[tuple[str, float]]] = {}
    for sn in sn_names:
        layer, ctx_mean = _layer_and_ctx_for_supernode(sn, final_supernodes.get(sn, []), attr)
        grouped.setdefault(layer, []).append((sn, ctx_mean))
    ordered_layers = sorted(grouped.keys())
    layer_index = {layer: idx for idx, layer in enumerate(ordered_layers)}
    pos: dict[str, tuple[float, float]] = {}
    for layer in ordered_layers:
        items = sorted(grouped[layer], key=lambda p: (p[1], p[0]))
        for x_idx, (sn, _) in enumerate(items):
            pos[sn] = (float(x_idx), float(layer_index[layer]))
    return pos


def _edge_style(weight: float, max_abs_w: float) -> tuple[float, str]:
    scale = abs(weight) / max(max_abs_w, 1e-9)
    width = 0.5 + 9.0 * scale
    alpha = 0.15 + 0.80 * scale
    if weight >= 0:
        color = f"rgba(33,120,78,{alpha:.3f})"
    else:
        color = f"rgba(203,24,29,{alpha:.3f})"
    return width, color


def supernode_graph_figure(
    sng: dict[str, Any],
    final_supernodes: dict[str, list[str]],
    attr: dict[str, dict[str, Any]] | None = None,
    title: str = "Cluster graph (supernodes)",
    seed: int = 42,
) -> go.Figure:
    """
    Build an interactive Plotly figure: supernodes as markers, directed edges as arrows.

    `sng` is the dict returned by `build_supernode_graph`.
    """
    sn_names: list[str] = list(sng["sn_names"])
    sn_adj = np.asarray(sng["sn_adj"], dtype=np.float64)
    sn_inf = np.asarray(sng["sn_inf"], dtype=np.float64) if sng.get("sn_inf") is not None else None

    g = nx.DiGraph()
    k = len(sn_names)
    for sn in sn_names:
        g.add_node(sn)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            w = float(sn_adj[i, j])
            if w != 0.0:
                g.add_edge(sn_names[i], sn_names[j], weight=w)

    pos = _layered_layout(sn_names, final_supernodes, attr)

    node_x = [pos[sn][0] for sn in sn_names if sn in pos]
    node_y = [pos[sn][1] for sn in sn_names if sn in pos]
    names_in_pos = [sn for sn in sn_names if sn in pos]

    colors: list[str] = []
    for sn in names_in_pos:
        kind = _sn_kind(sn)
        if kind == "emb":
            colors.append("#4CAF50")
        elif kind == "logit":
            colors.append("#FF9800")
        else:
            colors.append("#2196F3")

    sizes: list[float] = []
    for sn in names_in_pos:
        m = len(final_supernodes.get(sn, []))
        sizes.append(18 + min(32, 4 * max(m, 1)))

    labels = [_sn_label(sn, final_supernodes.get(sn, []), attr) for sn in names_in_pos]
    hover = [_sn_title(sn, final_supernodes.get(sn, []), attr) for sn in names_in_pos]

    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            w = float(sn_adj[i, j])
            if w == 0.0:
                continue
            u, v = sn_names[i], sn_names[j]
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            g.add_edge(u, v, weight=w)

    fig = go.Figure()

    max_abs_w = max((abs(float(data["weight"])) for _, _, data in g.edges(data=True)), default=1.0)
    for u, v, data in g.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = float(data["weight"])
        width, color = _edge_style(w, max_abs_w)
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(width=width, color=color),
                hovertemplate=f"{u} -> {v}<br>weight={w:.4f}<extra></extra>",
                showlegend=False,
            )
        )

    marker_kwargs: dict[str, Any] = dict(
        size=sizes,
        color=colors,
        line=dict(width=1, color="#333"),
    )
    if sn_inf is not None and len(sn_inf) == len(sn_names):
        vals = [float(sn_inf[sn_names.index(sn)]) for sn in names_in_pos]
        marker_kwargs = dict(
            size=sizes,
            color=vals,
            colorscale="Viridis",
            colorbar=dict(title="To-logit sum"),
            line=dict(width=1, color="#333"),
        )

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="middle center",
            textfont=dict(size=10, color="black"),
            marker=marker_kwargs,
            hovertext=hover,
            hoverinfo="text",
            name="Supernodes",
        )
    )

    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
        height=700,
    )
    fig.update_yaxes(autorange=True)
    return fig
