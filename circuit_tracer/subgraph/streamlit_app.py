from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import torch

from circuit_tracer.subgraph.cluster import cluster_graph
from circuit_tracer.subgraph.flow_analysis import build_supernode_graph, supernodes_to_mapping
from circuit_tracer.subgraph.prune import PruneGraph, load_prune_graph, prune_graph_pipeline
from circuit_tracer.subgraph.utils import _parse_layer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_DIRS = [REPO_ROOT / "demos" / "temp_graph_files", REPO_ROOT / "demos" / "graph_files"]
DEFAULT_PRUNED_DIR = REPO_ROOT / "demos" / "subgraph"


def _list_paths(paths: list[Path], suffix: str) -> list[Path]:
    all_files: list[Path] = []
    for directory in paths:
        if directory.exists():
            all_files.extend(sorted(directory.glob(f"*{suffix}")))
    return all_files


def _parse_token_weights(raw_value: str) -> list[float] | None:
    cleaned = raw_value.strip()
    if not cleaned:
        return None
    return [float(value.strip()) for value in cleaned.split(",") if value.strip()]


def _save_uploaded_file(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(data)
        return handle.name


def _supernode_kind(name: str) -> str:
    if "EMB" in name:
        return "embedding"
    if "LOGIT" in name:
        return "logit"
    return "middle"


def _build_supernode_graph_nx(
    supernode_map: dict[str, list[str]],
    supernode_flow: dict[str, Any],
    attr: dict[str, dict[str, Any]],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    sn_names = supernode_flow["sn_names"]
    sn_adj = np.asarray(supernode_flow["sn_adj"], dtype=np.float64)
    sn_inf = np.asarray(supernode_flow["sn_inf"], dtype=np.float64)

    for index, sn_name in enumerate(sn_names):
        members = supernode_map.get(sn_name, [])
        min_layer = min((_parse_layer(attr, node_id) for node_id in members), default=-1)
        graph.add_node(
            sn_name,
            kind=_supernode_kind(sn_name),
            members=members,
            member_count=len(members),
            min_layer=min_layer,
            influence=float(sn_inf[index]),
        )

    for src_idx, src_name in enumerate(sn_names):
        for dst_idx, dst_name in enumerate(sn_names):
            if src_idx == dst_idx:
                continue
            weight = float(sn_adj[src_idx, dst_idx])
            if weight == 0.0:
                continue
            graph.add_edge(src_name, dst_name, weight=weight)

    return graph


def _layered_layout(graph: nx.DiGraph) -> dict[str, tuple[float, float]]:
    grouped: dict[int, list[str]] = {}
    for node_name, data in graph.nodes(data=True):
        grouped.setdefault(int(data.get("min_layer", -1)), []).append(node_name)

    positions: dict[str, tuple[float, float]] = {}
    for x_index, layer in enumerate(sorted(grouped)):
        names = sorted(grouped[layer], key=lambda name: (graph.nodes[name]["kind"], name))
        for y_index, name in enumerate(names):
            y_pos = y_index - (len(names) - 1) / 2.0
            positions[name] = (float(x_index), -float(y_pos))
    return positions


def _plot_graph(graph: nx.DiGraph) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(14, 8))
    if not graph.nodes:
        ax.text(0.5, 0.5, "No nodes in clustered graph", ha="center", va="center")
        ax.axis("off")
        return fig

    pos = _layered_layout(graph)
    color_map = {"embedding": "#4C78A8", "middle": "#54A24B", "logit": "#F58518"}
    node_colors = [color_map.get(graph.nodes[node]["kind"], "#9C9C9C") for node in graph.nodes]
    node_sizes = [500 + 90 * graph.nodes[node]["member_count"] for node in graph.nodes]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95,
        linewidths=1.0,
        edgecolors="#2F2F2F",
        ax=ax,
    )
    labels = {name: f"{name}\n({graph.nodes[name]['member_count']})" for name in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)

    edges = list(graph.edges(data=True))
    if edges:
        widths = [1.0 + 6.0 * min(abs(edge_data["weight"]), 1.0) for _, _, edge_data in edges]
        edge_colors = ["#2CA02C" if edge_data["weight"] > 0 else "#D62728" for _, _, edge_data in edges]
        nx.draw_networkx_edges(
            graph,
            pos,
            arrows=True,
            arrowstyle="-|>",
            width=widths,
            edge_color=edge_colors,
            connectionstyle="arc3,rad=0.08",
            alpha=0.75,
            ax=ax,
        )

    ax.set_title("Final Cluster Graph (nodes = supernodes, edges = supernode flow)")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _build_node_table(graph: nx.DiGraph) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for node_name, data in graph.nodes(data=True):
        rows.append(
            {
                "supernode": node_name,
                "type": data["kind"],
                "member_count": data["member_count"],
                "influence": round(float(data["influence"]), 6),
                "min_layer": data["min_layer"],
                "members": ", ".join(data["members"]),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["supernode", "type", "member_count", "influence", "min_layer", "members"])
    return pd.DataFrame(rows).sort_values(by=["min_layer", "supernode"]).reset_index(drop=True)


def _build_edge_table(graph: nx.DiGraph) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for src, dst, edge_data in graph.edges(data=True):
        rows.append({"source": src, "target": dst, "weight": round(float(edge_data["weight"]), 6)})
    if not rows:
        return pd.DataFrame(columns=["source", "target", "weight"])
    return pd.DataFrame(rows).sort_values(by="weight", ascending=False).reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Cluster Graph Visualizer", layout="wide")
    st.title("Final Cluster Graph Visualizer")
    st.caption(
        "Load a full JSON graph or a pruned graph (.pt), run the clustering pipeline, "
        "and inspect the final supernode graph."
    )

    source_mode = st.sidebar.radio("Input graph type", options=["Full JSON graph", "Pruned graph (.pt)"])
    json_files = _list_paths(DEFAULT_JSON_DIRS, ".json")
    pruned_files = _list_paths([DEFAULT_PRUNED_DIR], ".pt")

    if source_mode == "Full JSON graph" and not json_files:
        st.warning("No JSON files found in demos/temp_graph_files or demos/graph_files. Upload one to continue.")
    if source_mode == "Pruned graph (.pt)" and not pruned_files:
        st.warning("No .pt files found in demos/subgraph. Upload one to continue.")

    selected_path: str | None
    uploaded_file = None
    if source_mode == "Full JSON graph":
        selected_path = st.sidebar.selectbox(
            "Choose existing JSON file",
            options=["(none)"] + [str(path) for path in json_files],
        )
        uploaded_file = st.sidebar.file_uploader("Or upload JSON graph", type=["json"])
    else:
        selected_path = st.sidebar.selectbox(
            "Choose existing pruned graph file",
            options=["(none)"] + [str(path) for path in pruned_files],
        )
        uploaded_file = st.sidebar.file_uploader("Or upload .pt pruned graph", type=["pt"])

    st.sidebar.subheader("Clustering parameters")
    target_k = st.sidebar.slider("target_k", min_value=1, max_value=20, value=7)
    max_layer_span = st.sidebar.slider("max_layer_span", min_value=1, max_value=16, value=4)
    max_sn_enabled = st.sidebar.checkbox("Set max_sn budget", value=False)
    max_sn = st.sidebar.slider("max_sn", min_value=1, max_value=30, value=12) if max_sn_enabled else None
    gamma = st.sidebar.slider("gamma", min_value=0.0, max_value=1.0, value=1.0)
    mediation_penalty = st.sidebar.slider("mediation_penalty", min_value=0.0, max_value=1.0, value=0.1)
    enforce_dag = st.sidebar.checkbox("enforce_dag", value=True)
    similarity_mode = st.sidebar.selectbox(
        "similarity_mode",
        options=["edge", "scores", "relevance", "influence", "uniform"],
        index=0,
    )

    prune_kwargs: dict[str, Any] = {}
    if source_mode == "Full JSON graph":
        st.sidebar.subheader("Pruning parameters")
        prune_kwargs["logit_weights"] = st.sidebar.selectbox("logit_weights", ["target", "probs"], index=0)
        prune_kwargs["node_threshold"] = st.sidebar.slider("node_threshold", min_value=0.0, max_value=1.0, value=0.8)
        prune_kwargs["edge_threshold"] = st.sidebar.slider("edge_threshold", min_value=0.0, max_value=1.0, value=0.98)
        prune_kwargs["alpha"] = st.sidebar.slider("alpha", min_value=0.0, max_value=1.0, value=0.5)
        prune_kwargs["keep_all_tokens_and_logits"] = st.sidebar.checkbox("keep_all_tokens_and_logits", value=True)
        prune_kwargs["combined_scores_method"] = st.sidebar.selectbox(
            "combined_scores_method", options=["geometric", "arithmetic", "harmonic"], index=0
        )
        prune_kwargs["normalization"] = st.sidebar.selectbox("normalization", ["min_max", "rank"], index=0)
        token_weights_raw = st.sidebar.text_input("token_weights (optional comma-separated floats)", value="")
        try:
            prune_kwargs["token_weights"] = _parse_token_weights(token_weights_raw)
        except ValueError as exc:
            st.sidebar.error(f"Invalid token_weights: {exc}")
            return

    run_clicked = st.sidebar.button("Run clustering pipeline", type="primary")
    if not run_clicked:
        st.info("Choose an input graph and click **Run clustering pipeline**.")
        return

    try:
        if uploaded_file is not None:
            suffix = ".json" if source_mode == "Full JSON graph" else ".pt"
            input_path = _save_uploaded_file(uploaded_file.getvalue(), suffix=suffix)
            st.write(f"Using uploaded file: `{uploaded_file.name}`")
        elif selected_path and selected_path != "(none)":
            input_path = selected_path
            st.write(f"Using file: `{input_path}`")
        else:
            st.error("Select a file or upload one before running the pipeline.")
            return

        with st.spinner("Running prune/cluster pipeline..."):
            if source_mode == "Full JSON graph":
                prune_graph = prune_graph_pipeline(json_path=input_path, **prune_kwargs)
            else:
                prune_graph = load_prune_graph(input_path)

            supernodes = cluster_graph(
                prune_graph,
                target_k=target_k,
                max_layer_span=max_layer_span,
                max_sn=max_sn,
                gamma=gamma,
                mediation_penalty=mediation_penalty,
                similarity_mode=similarity_mode,
                enforce_dag=enforce_dag,
            )
            supernode_map = supernodes_to_mapping(prune_graph, supernodes)
            supernode_flow = build_supernode_graph(prune_graph, supernode_map, enforce_dag=enforce_dag)
            graph = _build_supernode_graph_nx(supernode_map, supernode_flow, prune_graph.attr)

        left, right = st.columns(2)
        left.metric("Pruned nodes", prune_graph.num_nodes)
        left.metric("Pruned edges", prune_graph.num_edges)
        right.metric("Supernodes", len(graph.nodes))
        right.metric("Supernode edges", len(graph.edges))

        figure = _plot_graph(graph)
        st.pyplot(figure, clear_figure=True)
        st.caption(
            "Node colors: blue = embedding, green = middle cluster, orange = logit. "
            "Edge colors: green = positive weight, red = negative weight."
        )

        st.subheader("Supernode details")
        st.dataframe(_build_node_table(graph), use_container_width=True)

        st.subheader("Edge details")
        st.dataframe(_build_edge_table(graph), use_container_width=True)
    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
