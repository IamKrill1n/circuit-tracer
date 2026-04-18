"""Plotly visualization helpers for clustered supernode graphs."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from circuit_tracer.subgraph.cluster import _parse_layer
from circuit_tracer.subgraph.flow_analysis import _classify_sn


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


def _layout_from_graph(
    g: nx.DiGraph,
    sn_names: list[str],
    final_supernodes: dict[str, list[str]],
    attr: dict[str, dict[str, Any]] | None,
    seed: int,
) -> dict[str, tuple[float, float]]:
    for sn in sn_names:
        g.add_node(sn)
    if g.number_of_nodes() == 0:
        return {}
    if g.number_of_edges() == 0:
        # Fall back to layer ordering when there are no edges
        order = sorted(
            sn_names,
            key=lambda s: min(_parse_layer(attr or {}, n) for n in final_supernodes.get(s, [s])),
        )
        return {sn: (i * 1.2, 0.0) for i, sn in enumerate(order)}
    return nx.spring_layout(g, seed=seed, k=2.0 / np.sqrt(max(len(sn_names), 1)))


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
    for sn in sn_names:
        g.add_node(sn)
    k = len(sn_names)
    for i in range(k):
        for j in range(k):
            if i == j:
                continue
            w = float(sn_adj[i, j])
            if w != 0.0:
                g.add_edge(sn_names[i], sn_names[j], weight=w)

    pos = _layout_from_graph(g, sn_names, final_supernodes, attr, seed)

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

    hover = [_sn_title(sn, final_supernodes.get(sn, []), attr) for sn in names_in_pos]

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    max_w = max((float(sn_adj[i, j]) for i in range(k) for j in range(k) if i != j), default=1.0)
    max_w = max(max_w, 1e-9)

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
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

    fig = go.Figure()

    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=2, color="rgba(80,80,80,0.55)"),
                hoverinfo="skip",
                name="Edges",
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
            text=names_in_pos,
            textposition="top center",
            textfont=dict(size=10),
            marker=marker_kwargs,
            hovertext=hover,
            hoverinfo="text",
            name="Supernodes",
        )
    )

    fig.update_layout(
        title=title,
        showlegend=False,
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor="white",
        height=700,
    )
    return fig
