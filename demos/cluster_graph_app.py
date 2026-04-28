"""
Streamlit UI: load a full graph JSON or a saved PruneGraph (.pt), run prune (if needed),
cluster to supernodes, and visualize the final cluster graph with Plotly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

# Repo root (parent of `demos/`)
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from summarization.cluster import build_supernode_graph, cluster_graph, supernodes_to_mapping
from summarization.cluster_viz import supernode_graph_figure
from summarization.prune import load_prune_graph, prune_graph_pipeline

_DEFAULT_JSON_DIR = _ROOT / "demos" / "temp_graph_files"
_DEFAULT_PRUNE_DIR = _ROOT / "demos" / "subgraph"


def _list_json_files() -> list[Path]:
    if not _DEFAULT_JSON_DIR.is_dir():
        return []
    return sorted(_DEFAULT_JSON_DIR.glob("*.json"))


def _list_pt_files() -> list[Path]:
    if not _DEFAULT_PRUNE_DIR.is_dir():
        return []
    return sorted(_DEFAULT_PRUNE_DIR.glob("*.pt"))


@st.cache_data(show_spinner=False)
def _run_prune_cached(
    json_path: str,
    logit_weights: str,
    token_weights_json: str | None,
    node_threshold: float,
    edge_threshold: float,
    keep_all_tokens: bool,
    combined_scores_method: str,
    alpha: float,
) -> bytes:
    """Serialize PruneGraph via torch.save to bytes for caching."""
    import io

    import torch

    tw = json.loads(token_weights_json) if token_weights_json else None
    pg = prune_graph_pipeline(
        json_path=json_path,
        logit_weights=logit_weights,  # type: ignore[arg-type]
        token_weights=tw,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
        keep_all_tokens_and_logits=keep_all_tokens,
        combined_scores_method=combined_scores_method,
        alpha=alpha,
    )
    buf = io.BytesIO()
    torch.save(pg.to_dict(), buf)
    return buf.getvalue()


def _prune_from_cache(blob: bytes):
    import io

    import torch

    from summarization.prune import PruneGraph

    buf = io.BytesIO(blob)
    payload = torch.load(buf, map_location="cpu")
    return PruneGraph.from_dict(payload)


def main() -> None:
    st.set_page_config(page_title="Cluster graph viewer", layout="wide")
    st.title("Cluster graph visualization")
    st.markdown(
        "Load a **full attribution graph** (JSON) or a **pruned graph** (`.pt`). "
        "The app runs **prune → cluster → supernode graph**, then shows **supernodes** "
        "as nodes and **aggregated flows** as edges."
    )

    mode = st.radio(
        "Input",
        ("Full JSON graph", "Pruned graph (.pt)"),
        horizontal=True,
    )

    json_path = ""
    pt_path = ""

    if mode == "Full JSON graph":
        files = _list_json_files()
        c1, c2 = st.columns(2)
        with c1:
            if files:
                choice = st.selectbox(
                    "JSON under `demos/temp_graph_files/`",
                    options=[str(p) for p in files],
                    format_func=lambda p: Path(p).name,
                )
                json_path = choice
            else:
                st.info(f"No JSON files in `{_DEFAULT_JSON_DIR}`. Paste a path below.")
        with c2:
            json_path = st.text_input("Or path to graph JSON", value=json_path)
    else:
        files = _list_pt_files()
        c1, c2 = st.columns(2)
        with c1:
            if files:
                choice = st.selectbox(
                    "PruneGraph under `demos/subgraph/`",
                    options=[str(p) for p in files],
                    format_func=lambda p: Path(p).name,
                )
                pt_path = choice
            else:
                st.info(f"No `.pt` files in `{_DEFAULT_PRUNE_DIR}`. Paste a path below.")
        with c2:
            pt_path = st.text_input("Or path to PruneGraph `.pt`", value=pt_path)

    with st.expander("Pruning options (full JSON only)", expanded=mode == "Full JSON graph"):
        logit_weights = st.selectbox("Logit weights", ("target", "probs"), index=0)
        token_weights_raw = st.text_area(
            "Token weights (JSON list, optional)",
            value="",
            placeholder='e.g. [0, 0, 0.5, 0.5] — leave empty for uniform',
            height=68,
        )
        node_th = st.slider("Node threshold (cumulative)", 0.0, 1.0, 0.8, 0.01)
        edge_th = st.slider("Edge threshold (cumulative)", 0.0, 1.0, 0.98, 0.01)
        keep_all = st.checkbox("Keep all tokens and logits", value=True)
        combined = st.selectbox("Combined scores", ("geometric", "arithmetic", "harmonic"))
        alpha = st.slider("Alpha (influence vs relevance blend)", 0.0, 1.0, 0.5, 0.05)

    with st.expander("Clustering options", expanded=True):
        auto_k = st.checkbox("Pick k automatically (`find_best_k`)", value=False)
        target_k = st.number_input("Target k (middle supernodes)", min_value=1, max_value=50, value=7)
        max_layer_span = st.number_input("Max layer span", min_value=1, max_value=32, value=4)
        max_sn = st.number_input("Max supernodes cap (0 = no cap)", min_value=0, max_value=100, value=0)
        mean_method = st.selectbox("Mean method", ("geo", "harm", "arith"), index=2)
        similarity_mode = st.selectbox("Similarity mode", ("edge", "node"), index=0)
        mediation = st.slider("Mediation penalty", 0.0, 1.0, 0.1, 0.05)
        enforce_dag = st.checkbox("Enforce DAG constraints on clusters", value=True)

    run = st.button("Run pipeline & visualize", type="primary")

    if not run:
        return

    try:
        if mode == "Full JSON graph":
            path = Path(json_path).expanduser()
            if not path.is_file():
                st.error(f"JSON not found: {path}")
                return
            tw_key = token_weights_raw.strip() or "null"
            blob = _run_prune_cached(
                str(path),
                logit_weights,
                tw_key,
                float(node_th),
                float(edge_th),
                keep_all,
                combined,
                float(alpha),
            )
            prune_graph = _prune_from_cache(blob)
        else:
            path = Path(pt_path).expanduser()
            if not path.is_file():
                st.error(f"PruneGraph file not found: {path}")
                return
            prune_graph = load_prune_graph(str(path))
    except Exception as e:
        st.exception(e)
        return

    st.success(
        f"Pruned graph: **{prune_graph.num_nodes}** nodes, **{prune_graph.num_edges}** directed edges."
    )

    try:
        k_use = int(target_k)
        if auto_k:
            from summarization.auto_grouping import find_best_k

            best_k, _sweep = find_best_k(
                prune_graph,
                max_layer_span=int(max_layer_span),
                k_min_override=None,
                k_max_override=None,
                max_sn=int(max_sn) if max_sn > 0 else None,
                mean_method=str(mean_method),
                mediation_penalty=float(mediation),
                similarity_mode=str(similarity_mode),
                enforce_dag=enforce_dag,
            )
            k_use = int(best_k)
            st.info(f"Auto k = **{k_use}**")

        supernodes = cluster_graph(
            prune_graph,
            target_k=k_use,
            max_layer_span=int(max_layer_span),
            max_sn=int(max_sn) if max_sn > 0 else None,
            mean_method=str(mean_method),
            mediation_penalty=float(mediation),
            similarity_mode=str(similarity_mode),
            enforce_dag=enforce_dag,
        )
        mapped = supernodes_to_mapping(prune_graph, supernodes)
        sng = build_supernode_graph(prune_graph, mapped, enforce_dag=False)
    except Exception as e:
        st.exception(e)
        return

    st.caption(f"**{len(mapped)}** supernodes after clustering.")

    title = f"Cluster graph — {path.name}"
    fig = supernode_graph_figure(
        sng,
        mapped,
        attr=prune_graph.attr,
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Supernode members"):
        rows = [{"supernode": k, "n_members": len(v), "members": ", ".join(v[:12]) + (" …" if len(v) > 12 else "")} for k, v in mapped.items()]
        st.dataframe(rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
